"""Render paper-ready controlled-experiment plots and findings from results.csv."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
METRICS = (
    "ttft_ms",
    "itl_ms",
    "total_request_latency_ms",
    "output_tokens_per_second",
    "requests_per_second",
    "peak_gpu_memory_bytes",
    "failures",
    "timeouts",
)


def _raw_results() -> list[dict]:
    paths = sorted((ROOT / "raw/custom-server").glob("*restart-*.json"))
    paths += sorted((ROOT / "raw/vllm").glob("*restart-*.json"))
    return [json.loads(path.read_text()) | {"_path": str(path.relative_to(ROOT))} for path in paths]


def _write_summaries(results: list[dict]) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    failure_lines = ["# Failed and timed-out requests", ""]
    for result in results:
        implementation = result["environment"]["implementation"]
        variant = result["variant"]
        file_failed = False
        for cell in result["cells"]:
            summary = dict(cell["summary"])
            timed_out = sum(
                record.get("timeout", False) or record.get("error") == "timed_out"
                for record in cell["records"]
            )
            summary["failures"] = sum(record.get("error") is not None for record in cell["records"])
            summary["timeouts"] = timed_out
            grouped.setdefault((implementation, variant, cell["cell"]["name"]), []).append(summary)
            if summary["failures"]:
                file_failed = True
                errors = sorted(
                    {record["error"] for record in cell["records"] if record.get("error")}
                )
                failure_lines.append(
                    f"- `{result['_path']}` / `{cell['cell']['name']}`: "
                    f"{summary['failures']} errors, {timed_out} timeouts; {', '.join(errors)}."
                )
        if not file_failed:
            failure_lines.append(f"- `{result['_path']}`: no request failures or timeouts.")

    with (ROOT / "summaries/results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("implementation", "variant", "cell", "restarts", *METRICS))
        for key, summaries in sorted(grouped.items()):
            medians = []
            for metric in METRICS:
                values = [row[metric] for row in summaries if row[metric] is not None]
                medians.append(statistics.median(values) if values else "")
            writer.writerow((*key, len(summaries), *medians))

    with (ROOT / "summaries/variation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "implementation",
                "variant",
                "cell",
                "minimum_tok_s",
                "maximum_tok_s",
                "relative_range",
            )
        )
        for key, summaries in sorted(grouped.items()):
            values = [row["output_tokens_per_second"] for row in summaries]
            writer.writerow(
                (
                    *key,
                    min(values),
                    max(values),
                    (max(values) - min(values)) / statistics.mean(values),
                )
            )
    (ROOT / "summaries/failures.md").write_text("\n".join(failure_lines) + "\n")


def _rows() -> list[dict[str, str]]:
    with (ROOT / "summaries/results.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _svg_bars(path: Path, title: str, labels: list[str], series: dict[str, list[float]]) -> None:
    width, height = 1100, 520
    left, top, bottom = 80, 60, 110
    plot_height = height - top - bottom
    maximum = max(value for values in series.values() for value in values) or 1
    colors = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626")
    group_width = (width - left - 30) / len(labels)
    bar_width = group_width * 0.8 / len(series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - 30}" y2="{height - bottom}" stroke="#333"/>',
    ]
    for series_index, (name, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        for index, value in enumerate(values):
            x = left + index * group_width + group_width * 0.1 + series_index * bar_width
            bar_height = value / maximum * plot_height
            y = height - bottom - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
            )
        legend_x = left + series_index * 190
        parts.extend(
            (
                f'<rect x="{legend_x}" y="{height - 35}" width="14" height="14" fill="{color}"/>',
                f'<text x="{legend_x + 20}" y="{height - 23}" font-family="sans-serif" font-size="13">{name}</text>',
            )
        )
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * group_width
        parts.append(
            f'<text x="{x:.1f}" y="{height - bottom + 18}" transform="rotate(35 {x:.1f} {height - bottom + 18})" font-family="sans-serif" font-size="11">{label}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    raw = _raw_results()
    _write_summaries(raw)
    exclusions_path = ROOT / "summaries/exclusions.md"
    scheduler_exclusion = (
        "- Custom scheduler off: excluded because the scheduler owns the request lifecycle; "
        "`no_continuous_batching` only sets `max_active_sequences` to 1 and is not a scheduler-off run.\n"
    )
    exclusions = exclusions_path.read_text()
    if scheduler_exclusion not in exclusions:
        exclusions_path.write_text(exclusions + scheduler_exclusion)
    rows = _rows()
    cells = sorted({row["cell"] for row in rows})
    complete = {
        implementation: {
            row["cell"]: float(row["output_tokens_per_second"])
            for row in rows
            if row["implementation"] == implementation and row["variant"] == "complete"
        }
        for implementation in {row["implementation"] for row in rows}
    }
    _svg_bars(
        ROOT / "plots/throughput.svg",
        "Output throughput by workload",
        cells,
        {name: [values[cell] for cell in cells] for name, values in complete.items()},
    )

    custom = [row for row in rows if row["implementation"] == "custom-server"]
    effects = {}
    for variant in sorted({row["variant"] for row in custom} - {"complete"}):
        comparison = "no_cuda_graph" if variant == "no_triton" else "complete"
        comparison_values = {
            row["cell"]: float(row["output_tokens_per_second"])
            for row in custom
            if row["variant"] == comparison
        }
        ratios = [
            float(row["output_tokens_per_second"]) / comparison_values[row["cell"]]
            for row in custom
            if row["variant"] == variant and int(float(row["timeouts"])) == 0
        ]
        failures = sum(
            record.get("error") is not None
            for result in raw
            if result["environment"]["implementation"] == "custom-server"
            and result["variant"] == variant
            for cell in result["cells"]
            for record in cell["records"]
        )
        effects[variant] = (statistics.median(ratios), comparison, len(ratios), failures)
    _svg_bars(
        ROOT / "plots/ablation-throughput.svg",
        "Median throughput ratio over failure-free cells",
        ["median across 9 cells"],
        {name: [effect[0]] for name, effect in effects.items()},
    )

    custom_complete = complete["custom-server"]
    vllm_complete = complete["vllm"]
    ratio = statistics.median(custom_complete[cell] / vllm_complete[cell] for cell in cells)
    raw_failures = sum(
        record.get("error") is not None
        for result in raw
        for cell in result["cells"]
        for record in cell["records"]
    )
    raw_timeouts = sum(
        record.get("timeout", False) or record.get("error") == "timed_out"
        for result in raw
        for cell in result["cells"]
        for record in cell["records"]
    )
    successful_hashes: dict[tuple[str, int], set[str]] = {}
    for result in raw:
        for cell in result["cells"]:
            for record in cell["records"]:
                if record.get("error") is None:
                    key = (cell["cell"]["name"], record["request_index"])
                    successful_hashes.setdefault(key, set()).add(record["output_token_ids_sha256"])
    output_mismatches = sum(len(values) != 1 for values in successful_hashes.values())
    with (ROOT / "summaries/variation.csv").open(newline="") as handle:
        variation = list(csv.DictReader(handle))
    maximum_variation = max(variation, key=lambda row: float(row["relative_range"]))
    maximum_range = float(maximum_variation["relative_range"])
    extra_restart_variants = sorted(
        {
            row["variant"]
            for row in rows
            if int(row["restarts"]) > 3
        }
    )
    variation_note = (
        f" Additional restarts were added for: {', '.join(extra_restart_variants)}."
        if extra_restart_variants
        else ""
    )
    lines = [
        "# Controlled experiment findings",
        "",
        "All values are medians across the clean-process restarts in `results.csv`.",
        "",
        f"- Custom/vLLM median throughput ratio across the nine cells: {ratio:.3f}×.",
        f"- Raw request errors: {raw_failures}; all {raw_timeouts} were scheduler timeouts in slow ablations.",
        f"- Successful output-hash mismatches across servers, variants, and restarts: {output_mismatches}.",
        f"- Largest observed restart throughput range: {maximum_range:.1%}.{variation_note}",
        "",
        "## Complete-system comparison",
        "",
        "| Workload | Custom output tok/s | vLLM output tok/s | Custom/vLLM |",
        "|---|---:|---:|---:|",
    ]
    if "no_cuda_graph" in extra_restart_variants:
        first_three: dict[str, list[float]] = {}
        no_cuda_results = sorted(
            (
                result
                for result in raw
                if result["environment"]["implementation"] == "custom-server"
                and result["variant"] == "no_cuda_graph"
            ),
            key=lambda result: result["_path"],
        )[:3]
        for result in no_cuda_results:
            for cell in result["cells"]:
                first_three.setdefault(cell["cell"]["name"], []).append(
                    cell["summary"]["output_tokens_per_second"]
                )
        initial_effect = statistics.median(
            statistics.median(values) / custom_complete[cell]
            for cell, values in first_three.items()
        )
        lines.insert(
            8,
            f"- Maximum variation: `{maximum_variation['variant']}/{maximum_variation['cell']}` "
            f"({maximum_variation['minimum_tok_s']}-{maximum_variation['maximum_tok_s']} output tok/s). "
            f"Two added restarts changed its variant's median effect from {initial_effect:.3f}× "
            f"to {effects['no_cuda_graph'][0]:.3f}×.",
        )
    for cell in cells:
        lines.append(
            f"| `{cell}` | {custom_complete[cell]:.3f} | {vllm_complete[cell]:.3f} | "
            f"{custom_complete[cell] / vllm_complete[cell]:.3f}× |"
        )
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Variant | Comparison | Median throughput ratio | Failure-free cells | Failed requests |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for variant, (effect, comparison, failure_free_cells, failures) in effects.items():
        lines.append(
            f"| `{variant}` | `{comparison}` | {effect:.3f}× | "
            f"{failure_free_cells}/9 | {failures} |"
        )
    (ROOT / "summaries/findings.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
