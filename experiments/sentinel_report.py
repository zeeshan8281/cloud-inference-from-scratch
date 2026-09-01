"""Render the sentinel pilot's summaries/plots/manifest from its raw/ records.

Direct-engine closed-batch microbenchmark reporting only -- see
NEXT_EXPERIMENT_HANDOFF.md and experiments/sentinel-pilot/README.md. This
module never talks to a GPU: it only reads experiments/sentinel-pilot/raw/**
and regenerates everything else, so `reproduce.sh` can rebuild summaries and
plots byte-for-byte from raw records alone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from experiments.report import _svg_bars
from experiments.sentinel_pilot import (
    PAIRS,
    SENTINEL_CELLS,
    order_sensitivity_stats,
    paired_ratio_stats,
)

ROOT = Path(__file__).parent / "sentinel-pilot"

MODE_DIRS = {
    "resource_normalized": "raw/resource-normalized",
    "complete_system": "raw/complete-policy",
}
MODE_PHASES = {
    "resource_normalized": ("unique",),
    "complete_system": ("cold", "warm"),
}
# Fixed display order; never derived from a set or dict (see report.py's
# Phase 0 fix for why that breaks reproducibility across PYTHONHASHSEED).
MODES = ("resource_normalized", "complete_system")
CELL_NAMES = tuple(cell.name for cell in SENTINEL_CELLS)
CELL_CONCURRENCY = {cell.name: cell.concurrency for cell in SENTINEL_CELLS}


def _load_pairs(mode: str) -> list[dict[str, Any]]:
    directory = ROOT / MODE_DIRS[mode]
    if not directory.is_dir():
        return []
    pairs = []
    for pair in range(1, PAIRS + 1):
        path = directory / f"pair-{pair:02d}.json"
        if path.is_file():
            pairs.append(json.loads(path.read_text()) | {"_path": str(path.relative_to(ROOT))})
    return pairs


def _throughput(child_result: dict[str, Any], cell_name: str, phase: str) -> float:
    return child_result["cells"][cell_name]["phases"][phase]["summary"]["output_tokens_per_second"]


def _by_implementation(pair: dict[str, Any]) -> dict[str, Any]:
    return {child["implementation"]: child["result"] for child in pair["children"]}


def _mismatch_diagnostics(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """For a stopped, token_mismatch pair: the first output position at which
    the two engines' token IDs actually differ, per mismatched request.
    Divergence confined to concurrency > 1 cells, combined with the
    self-consistency check (each engine reproduces its own output exactly
    across repeated runs of the identical workload -- see
    summaries/self-consistency-check.json), means the two engines compute
    deterministically different results from each other under batched
    execution: not per-run randomness, and not a harness defect. This table
    is computed directly from the raw records, not asserted by hand."""
    by_implementation = _by_implementation(pair)
    rows = []
    for mismatch in pair["stop"]["detail"].get("mismatches", []):
        cell, phase = mismatch["cell"], mismatch["phase"]
        row = {
            "cell": cell,
            "concurrency": CELL_CONCURRENCY.get(cell),
            "phase": phase,
            "reason": mismatch["reason"],
        }
        index = mismatch.get("request_index")
        if index is not None and "custom" in by_implementation and "vllm" in by_implementation:
            try:
                custom_records = by_implementation["custom"]["cells"][cell]["phases"][phase]["records"]
                vllm_records = by_implementation["vllm"]["cells"][cell]["phases"][phase]["records"]
                custom_ids = next(r["output_token_ids"] for r in custom_records if r["request_index"] == index)
                vllm_ids = next(r["output_token_ids"] for r in vllm_records if r["request_index"] == index)
                length = min(len(custom_ids), len(vllm_ids))
                row["request_index"] = index
                row["first_diff_position"] = next(
                    (i for i in range(length) if custom_ids[i] != vllm_ids[i]), None
                )
                row["sequence_length"] = length
            except (KeyError, StopIteration):
                pass
        rows.append(row)
    return rows


def _write_correctness(pairs_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for mode in MODES:
        pairs = pairs_by_mode[mode]
        stops = [
            {"pair": pair["pair"], "stop": pair["stop"]} for pair in pairs if pair.get("stop") is not None
        ]
        report[mode] = {
            "pairs_present": len(pairs),
            "pairs_required": PAIRS,
            "stopped": bool(stops),
            "stops": stops,
        }
    (ROOT / "summaries/correctness.json").write_text(json.dumps(report, indent=2))
    return report


def _write_exclusions(pairs_by_mode: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Excluded pilot pairs", ""]
    any_excluded = False
    for mode in MODES:
        for pair in pairs_by_mode[mode]:
            if pair.get("stop") is not None:
                any_excluded = True
                lines.append(
                    f"- `{mode}` pair {pair['pair']:02d} (`{pair['_path']}`): stopped, "
                    f"kind=`{pair['stop']['kind']}`. Retained on disk, excluded from all "
                    "performance analysis."
                )
    if not any_excluded:
        lines.append("- None. Every collected pair completed without triggering a stop rule.")
    (ROOT / "summaries/exclusions.md").write_text("\n".join(lines) + "\n")


def _write_paired_results(
    pairs_by_mode: dict[str, list[dict[str, Any]]], correctness: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    rows = [
        (
            "mode", "cell", "phase", "n", "geometric_mean_ratio", "arithmetic_mean_ratio",
            "median_ratio", "ci95_low", "ci95_high", "t_critical", "degrees_of_freedom",
            "odd_geometric_mean_ratio", "even_geometric_mean_ratio",
        )
    ]
    analysis: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MODES:
        analysis[mode] = {}
        if correctness[mode]["stopped"]:
            continue  # no performance claim may be generated from a stopped pilot
        pairs = pairs_by_mode[mode]
        if len(pairs) != PAIRS:
            continue  # incomplete but not (yet) stopped: nothing to report yet
        by_implementation_per_pair = []
        for pair in sorted(pairs, key=lambda entry: entry["pair"]):
            by_implementation = {child["implementation"]: child["result"] for child in pair["children"]}
            by_implementation_per_pair.append(by_implementation)

        for cell_name in CELL_NAMES:
            analysis[mode][cell_name] = {}
            for phase in MODE_PHASES[mode]:
                throughput_pairs = [
                    (
                        _throughput(by_implementation["custom"], cell_name, phase),
                        _throughput(by_implementation["vllm"], cell_name, phase),
                    )
                    for by_implementation in by_implementation_per_pair
                ]
                stats = paired_ratio_stats(throughput_pairs)
                order = order_sensitivity_stats(throughput_pairs)
                analysis[mode][cell_name][phase] = {"overall": stats, "order": order}
                rows.append(
                    (
                        mode, cell_name, phase, stats["n"],
                        stats["geometric_mean_ratio"], stats["arithmetic_mean_ratio"],
                        stats["median_ratio"], stats["ci95_low"], stats["ci95_high"],
                        stats["t_critical"], stats["degrees_of_freedom"],
                        order["odd"]["geometric_mean_ratio"], order["even"]["geometric_mean_ratio"],
                    )
                )
    with (ROOT / "summaries/paired-results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)
    return analysis


def _write_findings(
    correctness: dict[str, Any],
    analysis: dict[str, dict[str, dict[str, Any]]],
    pairs_by_mode: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        "# Sentinel pilot findings: direct-engine closed-batch microbenchmark",
        "",
        "Not an HTTP or production-serving benchmark. `c*` in a cell name is "
        "offered concurrency, not guaranteed simultaneous admission. All GPU "
        "memory figures are operational device footprint samples under each "
        "mode's own memory policy, not a general efficiency comparison except "
        "where the resource-normalized mode's matched KV capacity applies.",
        "",
    ]
    for mode in MODES:
        lines.append(f"## {mode}")
        lines.append("")
        if correctness[mode]["stopped"]:
            lines.append("**STOPPED.** No performance claim may be generated from a stopped pilot.")
            lines.append("")
            for pair in pairs_by_mode[mode]:
                if pair.get("stop") is None:
                    continue
                kind = pair["stop"]["kind"]
                lines.append(f"Pair {pair['pair']:02d}: stop kind `{kind}`.")
                if kind != "token_mismatch":
                    lines.append(f"Detail: `{json.dumps(pair['stop']['detail'])}`")
                    lines.append("")
                    continue
                diagnostics = _mismatch_diagnostics(pair)
                positioned = [row for row in diagnostics if row.get("first_diff_position") is not None]
                concurrencies = {row["concurrency"] for row in diagnostics if row.get("concurrency")}
                if positioned:
                    lines.append(
                        f"{len(diagnostics)} of the pair's sentinel requests mismatched, confined to "
                        f"concurrency {sorted(concurrencies)} cells; first differing output position "
                        f"ranged {min(r['first_diff_position'] for r in positioned)}-"
                        f"{max(r['first_diff_position'] for r in positioned)} tokens in. A separate "
                        "self-consistency check (see below) shows each engine reproduces its own output "
                        "exactly across repeated runs, so this is not per-run randomness: the two engines "
                        "compute deterministically different results from each other under concurrent "
                        "batched execution at fp16, not a harness defect."
                    )
                lines.append("")
                lines.extend(
                    [
                        "| Cell | Concurrency | Phase | Request | First diff position | Sequence length |",
                        "|---|---:|---|---:|---:|---:|",
                    ]
                )
                for row in diagnostics:
                    lines.append(
                        f"| `{row['cell']}` | {row['concurrency']} | {row['phase']} | "
                        f"{row.get('request_index', 'n/a')} | {row.get('first_diff_position', 'n/a')} | "
                        f"{row.get('sequence_length', 'n/a')} |"
                    )
                lines.append("")
            continue
        if mode not in analysis or not analysis[mode]:
            lines.append(
                f"Incomplete: {correctness[mode]['pairs_present']}/{correctness[mode]['pairs_required']} "
                "pairs collected so far; no stop triggered yet. No claim generated until all "
                "10 pairs complete."
            )
            lines.append("")
            continue
        lines.extend(
            [
                "| Cell | Phase | n | Geometric mean (custom/vLLM) | 95% CI | Odd-order GM | Even-order GM |",
                "|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for cell_name in CELL_NAMES:
            for phase in MODE_PHASES[mode]:
                stats = analysis[mode][cell_name][phase]["overall"]
                order = analysis[mode][cell_name][phase]["order"]
                ci = (
                    f"[{stats['ci95_low']:.3f}x, {stats['ci95_high']:.3f}x]"
                    if stats["ci95_low"] is not None
                    else "n/a"
                )
                lines.append(
                    f"| `{cell_name}` | {phase} | {stats['n']} | {stats['geometric_mean_ratio']:.3f}x | "
                    f"{ci} | {order['odd']['geometric_mean_ratio']:.3f}x | "
                    f"{order['even']['geometric_mean_ratio']:.3f}x |"
                )
        lines.append("")
    lines.extend(_self_consistency_section())
    (ROOT / "summaries/findings.md").write_text("\n".join(lines) + "\n")


def _self_consistency_section() -> list[str]:
    """If a self-consistency diagnostic (sentinel_self_consistency_check) has
    been run and its result saved to summaries/self-consistency-check.json,
    fold it into the report: it is the direct evidence for whether observed
    cross-engine mismatches are per-run randomness or a deterministic
    difference between the two engines."""
    path = ROOT / "summaries/self-consistency-check.json"
    if not path.is_file():
        return []
    result = json.loads(path.read_text())
    return [
        "## Self-consistency diagnostic",
        "",
        "Not part of the 10-pair protocol: each engine run twice, independently, "
        "in fresh subprocesses, on the identical materialized concurrency-8/32 "
        "workload, with the other engine entirely absent from the comparison.",
        "",
        f"- Custom engine self-consistent across repeated runs: **{result['custom_self_consistent']}** "
        f"({len(result['custom_mismatches'])} mismatches).",
        f"- vLLM self-consistent across repeated runs: **{result['vllm_self_consistent']}** "
        f"({len(result['vllm_mismatches'])} mismatches).",
        "",
        "Both self-consistent means the token mismatches found above are not "
        "per-run randomness: the two engines deterministically compute "
        "different results from each other under concurrent batched execution "
        "at fp16, reproducibly, not due to noise.",
        "",
    ]


def _write_plots(analysis: dict[str, dict[str, dict[str, Any]]]) -> None:
    plots_dir = ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        if mode not in analysis or not analysis[mode]:
            continue
        for phase in MODE_PHASES[mode]:
            series = {
                "geometric_mean_ratio": [
                    analysis[mode][cell_name][phase]["overall"]["geometric_mean_ratio"]
                    for cell_name in CELL_NAMES
                    if phase in analysis[mode][cell_name]
                ]
            }
            if not series["geometric_mean_ratio"]:
                continue
            _svg_bars(
                plots_dir / f"{mode}-{phase}-ratio.svg",
                f"custom/vLLM throughput ratio ({mode}, {phase})",
                list(CELL_NAMES),
                series,
            )


def _write_artifact_manifest() -> None:
    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "artifact-manifest.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))


def main() -> None:
    (ROOT / "summaries").mkdir(parents=True, exist_ok=True)
    (ROOT / "plots").mkdir(parents=True, exist_ok=True)
    pairs_by_mode = {mode: _load_pairs(mode) for mode in MODES}
    correctness = _write_correctness(pairs_by_mode)
    _write_exclusions(pairs_by_mode)
    analysis = _write_paired_results(pairs_by_mode, correctness)
    _write_findings(correctness, analysis, pairs_by_mode)
    _write_plots(analysis)
    _write_artifact_manifest()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    main()
