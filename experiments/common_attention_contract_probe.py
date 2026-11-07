"""Installed attention primitive tests; direct operator graphs are not model full graphs."""

import base64
import hashlib
import json
import sys
from pathlib import Path

CASES = ("prefill", "mixed", "shared_decode", "changed_metadata_replay")


def specifications(name):
    if name == "prefill":
        return [256], [256], False, False
    if name == "mixed":
        return [1, 256, 1], [511, 256, 511], False, False
    if name == "shared_decode":
        return [1, 1], [511, 511], True, False
    if name == "changed_metadata_replay":
        return [2], [510], False, True
    raise ValueError("unknown declared primitive case")


def decode(record, torch):
    raw = base64.b64decode(record["bytes_base64"], validate=True)
    if record["dtype"] != "torch.float16" or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("captured operand checksum/dtype mismatch")
    return (
        torch.frombuffer(bytearray(raw), dtype=torch.float16)
        .clone()
        .cuda()
        .reshape(record["shape"])
    )


def digest(value, torch):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite primitive evidence")
    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def materialize(name, state, torch):
    lengths, computed, shared, changed = specifications(name)
    qrow = decode(state["query"], torch).reshape(16, 128)
    keys, values = decode(state["keys"], torch), decode(state["values"], torch)
    m = sum(lengths)
    # Fixed generous storage lets graph replay change page IDs without reallocating.
    final = torch.zeros((2, 96, 16, 2, 128), dtype=torch.float16, device="cuda")
    table = torch.zeros((3, 32), dtype=torch.int32, device="cuda")
    q = torch.empty_strided((m + 4, 16, 128), (2560, 128, 1), dtype=torch.float16, device="cuda")
    q.copy_((-qrow if changed else qrow).expand(m + 4, -1, -1))
    k = torch.zeros((m + 4, 2, 128), dtype=torch.float16, device="cuda")
    v = torch.zeros_like(k)
    slots, seq_ids, positions, starts = [], [], [], [0]
    for request, (count, past) in enumerate(zip(lengths, computed, strict=True)):
        pages = list(range(request * 32, (request + 1) * 32))
        if shared:
            pages = list(range(31)) + [31 + request]
        if changed:
            pages = list(reversed(range(64, 96)))
        table[request].copy_(torch.tensor(pages, dtype=torch.int32, device="cuda"))
        for position in range(past + count):
            block, offset = pages[position // 16], position % 16
            final[0, block, offset].copy_(keys[position])
            final[1, block, offset].copy_(values[position])
        for position in range(past, past + count):
            row = len(slots)
            slots.append(pages[position // 16] * 16 + position % 16)
            seq_ids.append(request)
            positions.append(position)
            k[row].copy_(keys[position])
            v[row].copy_(values[position])
        starts.append(len(slots))
    if len(set(slots)) != len(slots):
        raise ValueError("query writes share physical slots")
    initial = final.clone()
    for slot in slots:
        initial[:, slot // 16, slot % 16].zero_()
    return dict(
        q=q,
        k=k,
        v=v,
        initial=initial,
        final=final,
        table=table,
        slots=torch.tensor(slots, dtype=torch.int64, device="cuda"),
        ids=torch.tensor(seq_ids, dtype=torch.int32, device="cuda"),
        positions=torch.tensor(positions, dtype=torch.int64, device="cuda"),
        lengths=lengths,
        computed=computed,
        starts=starts,
        m=m,
    )


def run(config, result, save):
    import torch
    import vllm
    from vllm import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from cloud_engine.common_attention import (
        ATTENTION_ID,
        bind_impl,
        install_metadata,
        validate_impl,
    )
    from cloud_engine.kernel import ragged_attention_direct
    from cloud_engine.numerical_policy import POLICY_ID
    from cloud_engine.precision import configure_numerical_policy_precision, precision_status

    configure_numerical_policy_precision()
    torch.set_grad_enabled(False)
    fixture_raw = Path(config["fixture_path"]).read_bytes()
    fixture = json.loads(fixture_raw)
    state = fixture["states"]["custom-batch"]
    args = EngineArgs(
        model=config["model_dir"],
        tokenizer=config["model_dir"],
        dtype="half",
        max_model_len=4096,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
        enforce_eager=False,
        disable_log_stats=True,
        seed=0,
    )
    vconfig = args.create_engine_config()
    result.update(
        policy_id=POLICY_ID,
        attention_id=ATTENTION_ID,
        fixture_sha256=hashlib.sha256(fixture_raw).hexdigest(),
        precision=precision_status(),
        runtime={"torch": torch.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda},
        scope="Actual native builder and installed primitive; direct operator fullgraph/CUDAgraph, not unsupported FA2 full-attention model graphs. Full pair follows separately.",
        source_sha256={
            "common_attention": hashlib.sha256(
                Path(sys.modules[bind_impl.__module__].__file__).read_bytes()
            ).hexdigest(),
            "kernel": hashlib.sha256(
                Path(sys.modules[ragged_attention_direct.__module__].__file__).read_bytes()
            ).hexdigest(),
        },
        cases=[],
    )
    with set_current_vllm_config(vconfig):
        install_metadata()
        spec = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=128, dtype=torch.float16, use_mla=False
        )
        builder = FlashAttentionMetadataBuilder(spec, vconfig, torch.device("cuda:0"))
        impl = FlashAttentionImpl(16, 128, 128**-0.5, 2, None, None, "auto")
        validate_impl(impl)
        record = bind_impl(impl)
        result["installed_callable_active"] = impl.forward is record[1]
        result["native_builder_class"] = type(builder).__name__
        result["full_attention_model_graph_supported"] = builder.full_cudagraph_supported
        layer = torch.nn.Module()
        layer.register_buffer("_k_scale", torch.tensor(1.0, device="cuda"))
        layer.register_buffer("_v_scale", torch.tensor(1.0, device="cuda"))
        compiled = torch.compile(impl.forward, fullgraph=True)
        # Pinned PIECEWISE warmup passes no attention metadata, preserving output/cache.
        dummy = materialize("shared_decode", state, torch)
        dummy_out = torch.full((6, 16, 128), 7.0, dtype=torch.float16, device="cuda")
        old_cache = dummy["initial"].clone()
        returned = impl.forward(
            layer, dummy["q"], dummy["k"], dummy["v"], dummy["initial"], None, output=dummy_out
        )
        result["profiling_none_preserves_alias_cache_output"] = bool(
            returned is dummy_out
            and torch.equal(dummy["initial"], old_cache)
            and bool((dummy_out == 7).all())
        )
        save()
        dynamic = None
        for name in CASES:
            data = materialize(name, state, torch)
            m = data["m"]
            if name != "changed_metadata_replay":
                q, k, v = data["q"].clone(), data["k"].clone(), data["v"].clone()
                # Preserve the source-derived fused-QKV row stride explicitly.
                q = torch.empty_strided(
                    data["q"].shape, data["q"].stride(), dtype=torch.float16, device="cuda"
                ).copy_(data["q"])
                source_cache = data["initial"].clone()
                cache = source_cache.clone()
                table, slots = data["table"].clone(), data["slots"].clone()
                starts = torch.tensor(data["starts"], dtype=torch.int32, device="cuda")
                seq_lens = torch.tensor(
                    [a + b for a, b in zip(data["lengths"], data["computed"], strict=True)],
                    dtype=torch.int32,
                    device="cuda",
                )
                output = torch.full((m + 4, 16, 128), 7.0, dtype=torch.float16, device="cuda")
            else:
                (
                    q,
                    k,
                    v,
                    source_cache,
                    cache,
                    table,
                    slots,
                    starts,
                    seq_lens,
                    output,
                    graph,
                    graph_result,
                    pointers,
                ) = dynamic
                q.copy_(data["q"])
                k.copy_(data["k"])
                v.copy_(data["v"])
                source_cache.copy_(data["initial"])
                table.copy_(data["table"])
                slots.copy_(data["slots"])
                starts[:2].copy_(torch.tensor(data["starts"], dtype=torch.int32, device="cuda"))
                seq_lens[0] = 512
            reqs = len(data["lengths"])
            common = CommonAttentionMetadata(
                query_start_loc=starts[: reqs + 1],
                query_start_loc_cpu=torch.tensor(data["starts"], dtype=torch.int32),
                seq_lens=seq_lens[:reqs],
                seq_lens_cpu=torch.tensor(
                    [a + b for a, b in zip(data["lengths"], data["computed"], strict=True)],
                    dtype=torch.int32,
                ),
                num_computed_tokens_cpu=torch.tensor(data["computed"], dtype=torch.int32),
                num_reqs=reqs,
                num_actual_tokens=m,
                max_query_len=max(data["lengths"]),
                block_table_tensor=table[:reqs],
                slot_mapping=slots,
            )
            metadata = builder.build(0, common)
            actual_ids = metadata.cloud_query_seq_ids
            actual_positions = metadata.cloud_positions
            if not torch.equal(actual_ids, data["ids"]) or not torch.equal(
                actual_positions, data["positions"]
            ):
                raise ValueError("actual builder produced stale logical metadata")
            expected = ragged_attention_direct(
                q[:m],
                data["final"][0],
                data["final"][1],
                table[:reqs],
                data["ids"],
                data["positions"],
                128**-0.5,
            )
            case = {
                "name": name,
                "tokens": m,
                "requests": reqs,
                "metadata_ids": actual_ids.cpu().tolist(),
                "metadata_positions": actual_positions.cpu().tolist(),
                "write_slots": slots.cpu().tolist(),
                "query_sha256": digest(q, torch),
                "expected_output_sha256": digest(expected, torch),
                "expected_cache_sha256": digest(data["final"], torch),
            }
            if name == "prefill":
                case["historical_custom_target_exact"] = bool(
                    torch.equal(
                        expected[-1].reshape(2048), decode(state["output"], torch).reshape(2048)
                    )
                )
                if case["historical_custom_target_exact"] is not True:
                    result["cases"].append(case)
                    save()
                    raise ValueError(
                        "existing custom kernel did not reproduce actual target output"
                    )
            result["cases"].append(case)
            save()

            def reset(cache=cache, source_cache=source_cache, output=output):
                cache.copy_(source_cache)
                output.fill_(7.0)

            def check(returned, cache=cache, output=output, data=data, expected=expected, m=m):
                return {
                    "output_alias": returned is output,
                    "output_exact": bool(torch.equal(output[:m], expected)),
                    "cache_append_and_untouched_exact": bool(torch.equal(cache, data["final"])),
                    "padding_unchanged": bool((output[m:] == 7).all()),
                    "output_sha256": digest(output[:m], torch),
                    "cache_sha256": digest(cache, torch),
                }

            reset()
            case["eager"] = check(impl.forward(layer, q, k, v, cache, metadata, output=output))
            save()
            reset()
            case["compiled"] = check(compiled(layer, q, k, v, cache, metadata, output=output))
            save()
            if name != "changed_metadata_replay":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        reset()
                        compiled(layer, q, k, v, cache, metadata, output=output)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    reset()
                    graph_result = compiled(layer, q, k, v, cache, metadata, output=output)
                pointers = [
                    x.data_ptr()
                    for x in (
                        q,
                        k,
                        v,
                        source_cache,
                        cache,
                        table,
                        slots,
                        metadata.cloud_query_seq_ids,
                        metadata.cloud_positions,
                    )
                ]
            else:
                current = [
                    x.data_ptr()
                    for x in (
                        q,
                        k,
                        v,
                        source_cache,
                        cache,
                        table,
                        slots,
                        metadata.cloud_query_seq_ids,
                        metadata.cloud_positions,
                    )
                ]
                case["graph_storage_before"] = pointers
                case["graph_storage_after"] = current
                case["stable_graph_storage"] = current == pointers
            case["graph_storage"] = pointers
            output.fill_(float("nan"))
            cache.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            case["graph"] = check(graph_result)
            save()
            if name == "shared_decode":
                dynamic = (
                    q,
                    k,
                    v,
                    source_cache,
                    cache,
                    table,
                    slots,
                    starts,
                    seq_lens,
                    output,
                    graph,
                    graph_result,
                    pointers,
                )
            for phase in ("eager", "compiled", "graph"):
                if not all(
                    case[phase][key]
                    for key in (
                        "output_alias",
                        "output_exact",
                        "cache_append_and_untouched_exact",
                        "padding_unchanged",
                    )
                ):
                    raise ValueError("installed attention primitive contract failed")
            if name == "changed_metadata_replay" and case["stable_graph_storage"] is not True:
                raise ValueError("dynamic metadata replay changed storage addresses")
        if not result["profiling_none_preserves_alias_cache_output"]:
            raise ValueError("actual profiling path changed output/cache")
    result["complete"] = True


def main():
    config_path, output = map(Path, sys.argv[1:])
    if output.exists():
        raise ValueError("refusing evidence overwrite")
    config = json.loads(config_path.read_text())
    result = {
        "schema": "common-attention-contract-v1",
        "complete": False,
        "error": None,
        "config": config,
        "gate_cleared": False,
    }

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False))
        temporary.replace(output)

    save()
    try:
        run(config, result, save)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    save()


if __name__ == "__main__":
    main()
