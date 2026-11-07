"""First-divergence reports from real sentinel captures; no tolerance or pass gate."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from experiments.sentinel_diagnostics import compare_top_k, cross_choice_diagnostics, step_margin


def compare_runs(left: dict, right: dict) -> dict:
    """Compare distributions only through first disagreement on an identical input."""
    if left.get("crashed") or right.get("crashed"):
        return {"status": "unavailable", "reason": "child_crashed"}
    try:
        a, b = left["result"], right["result"]
        ac, bc = a["config"], b["config"]
        if ac["batch_ids"][ac.get("target_index", 0)] != bc["batch_ids"][bc.get("target_index", 0)]:
            return {"status": "unavailable", "reason": "different_input_prefix"}
        ar, br = a["result"], b["result"]
        at, bt = ar["output_tokens"], br["output_tokens"]
        if not at or not bt or ar.get("verified") is not True or br.get("verified") is not True:
            return {"status": "unavailable", "reason": "empty_or_unverified_capture"}
        steps = []
        for position, (token_a, token_b) in enumerate(zip(at, bt, strict=False)):
            ak, bk = ar["top_k"][position], br["top_k"][position]
            for top, token in ((ak, token_a), (bk, token_b)):
                ids = [entry[0] for entry in top]
                values = [entry[1] for entry in top]
                if (
                    len(top) < 2
                    or len(set(ids)) != len(ids)
                    or token not in ids
                    or not all(math.isfinite(value) for value in values)
                    or values != sorted(values, reverse=True)
                    or dict(top)[token] != values[0]
                ):
                    return {
                        "status": "unavailable",
                        "reason": "invalid_top_k",
                        "position": position,
                    }
            steps.append(
                {
                    "position": position,
                    "left_token": token_a,
                    "right_token": token_b,
                    "left_margin": step_margin(ak),
                    "right_margin": step_margin(bk),
                    **compare_top_k(ak, bk),
                    **cross_choice_diagnostics(ak, bk, token_a, token_b),
                }
            )
            if token_a != token_b:
                return {
                    "status": "diverged",
                    "first_difference": position,
                    "compared_steps": steps,
                    "excluded_post_divergence_steps": min(len(at), len(bt)) - position - 1,
                }
        if len(at) != len(bt):
            return {
                "status": "incomplete",
                "reason": "unequal_output_lengths",
                "compared_steps": steps,
            }
        return {"status": "exact_in_observed_window", "compared_steps": steps}
    except (KeyError, IndexError, TypeError, ValueError):
        return {"status": "unavailable", "reason": "malformed_capture"}


def summarize(payload: dict) -> dict:
    comparisons = []
    for pair in payload["comparisons"]:
        comparisons.append(
            {**pair, **compare_runs(payload["runs"][pair["left"]], payload["runs"][pair["right"]])}
        )
    return {
        "schema": "numerical-debugger-v1",
        "environment": payload.get("environment", {}),
        "run_count": len(payload["runs"]),
        "crashed_runs": [name for name, run in payload["runs"].items() if run.get("crashed")],
        "comparisons": comparisons,
        "performance_gate": "unchanged; stopped pilot and V2 holdout remain stopped",
        "limits": [
            "One diagnosed sentinel request, not a new sealed holdout or quality certification.",
            "Top-k intersection metrics are not full-vocabulary error bounds.",
            "Observed batch sensitivity does not identify a causal kernel operation.",
            "Instrumentation and fresh-process diagnostic configuration differ from the full pilot.",
            "Shared-prefix probes re-prefill a modified target input; they do not preserve decode cache state.",
        ],
    }


def write_report(source: Path, destination: Path) -> dict:
    raw = source.read_bytes()
    summary = summarize(json.loads(raw))
    summary["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Numerical divergence study",
        "",
        "Unit-test fixtures are not study results; raw capture provenance is recorded below.",
        "",
        f"Raw SHA-256: `{summary['raw_sha256']}`",
        "",
        f"Runs: {summary['run_count']}; crashes: {len(summary['crashed_runs'])}.",
        "",
        "| Comparison | Status | First differing position | Left / right margin | Intersection max Δ log p |",
        "|---|---|---:|---:|---:|",
    ]
    rows = []
    for comparison in summary["comparisons"]:
        step = comparison.get("compared_steps", [{}])[-1]
        position = comparison.get("first_difference", "—")
        margin = " / ".join(
            f"{step[k]:.7f}" if k in step else "—" for k in ("left_margin", "right_margin")
        )
        drift = step.get("intersection_max_abs_diff")
        drift_text = f"{drift:.7f}" if drift is not None else "—"
        cells = (comparison["label"], comparison["status"], position, margin, drift_text)
        lines.append("| " + " | ".join(map(str, cells)) + " |")
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in cells) + "</tr>")
    lines += [
        "",
        "For exact windows, margin/drift describes the final compared step; otherwise the first disagreement.",
        "",
        "## Environment",
        "```json",
        json.dumps(summary["environment"], indent=2),
        "```",
        "",
        "## Boundaries",
        *(f"- {limit}" for limit in summary["limits"]),
        "",
        summary["performance_gate"],
    ]
    (destination / "findings.md").write_text("\n".join(lines) + "\n")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Numerical divergence study</title><style>body{font:16px system-ui;background:#f5f6fa;color:#121629;margin:40px auto;max-width:1200px;padding:24px}h1{font-size:40px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:16px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}small{color:#555}</style>
<h1>Numerical divergence study</h1><p>First disagreement. Shared prefix. No invented probabilities.</p><div style="overflow:auto"><table><thead><tr><th>Comparison</th><th>Status</th><th>Position</th><th>Margins</th><th>Intersection max Δ log p</th></tr></thead><tbody>"""
    page += "".join(rows) + "</tbody></table></div><h2>Evidence boundaries</h2><ul>"
    page += "".join(f"<li>{html.escape(limit)}</li>" for limit in summary["limits"])
    page += (
        "</ul><p>"
        + html.escape(summary["performance_gate"])
        + "</p><h2>Recorded environment</h2><pre>"
    )
    page += (
        html.escape(json.dumps(summary["environment"], indent=2))
        + "</pre><small>Raw SHA-256: "
        + summary["raw_sha256"]
        + "</small></html>"
    )
    (destination / "index.html").write_text(page)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.capture, args.output)


if __name__ == "__main__":
    main()
