"""Generate comparison report from experiment results.

Reads experiment state and eval logs to produce Markdown comparison tables.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output logs/comparison_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from llm_training.common import ROOT, load_experiment_states, log


def generate_report(output_path: Path | None = None) -> str:
    """Generate full comparison report. Returns Markdown string."""
    output_path = output_path or (ROOT / "logs" / "comparison_report.md")
    states = load_experiment_states()

    if not states:
        log.warning("No experiment states found. Nothing to report.")
        return ""

    # Load eval results
    eval_dir = ROOT / "logs" / "eval"
    eval_results: dict[str, dict] = {}
    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            eval_results[f.stem] = json.loads(f.read_text())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"## Comparison Report — Generated {now}\n"]

    # Group by specialization
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for exp_id, state in states.items():
        parts = exp_id.rsplit("_", 3)
        if len(parts) >= 4:
            spec = parts[-2]
            by_spec[spec].append({
                "id": exp_id,
                "base": "_".join(parts[:-3]),
                "quant": parts[-3],
                "variant": parts[-1],
                "state": state,
                "results": eval_results.get(exp_id, state.get("results", {})),
            })

    for spec, experiments in sorted(by_spec.items()):
        lines.append(f"\n### By Specialization: {spec.title()}\n")

        # Collect all metric names
        all_metrics: set[str] = set()
        for exp in experiments:
            all_metrics.update(exp["results"].keys())

        # Remove non-numeric metrics
        numeric_metrics = sorted(
            m for m in all_metrics
            if any(
                isinstance(exp["results"].get(m), (int, float))
                for exp in experiments
            )
        )

        # Build table
        headers = ["Experiment", "Base", "Quant", "Fine-tuned"]
        headers.extend(numeric_metrics[:8])  # cap columns
        headers.extend(["Status"])

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for exp in sorted(experiments, key=lambda e: (e["base"], e["quant"], e["variant"])):
            row = [
                exp["id"],
                exp["base"],
                exp["quant"],
                "yes" if exp["variant"] == "finetuned" else "no",
            ]
            for m in numeric_metrics[:8]:
                val = exp["results"].get(m)
                if isinstance(val, float):
                    row.append(f"{val:.3f}")
                elif isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append("-")
            row.append(exp["state"].get("status", "unknown"))
            lines.append("| " + " | ".join(row) + " |")

        # Vanilla vs finetuned comparison
        lines.append(f"\n#### Fine-tuning Impact: {spec.title()}\n")
        vanilla_map = {
            e["base"] + "_" + e["quant"]: e
            for e in experiments if e["variant"] == "vanilla"
        }
        finetuned_map = {
            e["base"] + "_" + e["quant"]: e
            for e in experiments if e["variant"] == "finetuned"
        }

        for key in sorted(set(vanilla_map) & set(finetuned_map)):
            v = vanilla_map[key]
            ft = finetuned_map[key]
            lines.append(f"\n**{key}:**\n")
            lines.append("| Metric | Vanilla | Fine-tuned | Delta |")
            lines.append("| --- | --- | --- | --- |")
            for m in numeric_metrics[:8]:
                v_val = v["results"].get(m)
                ft_val = ft["results"].get(m)
                if isinstance(v_val, (int, float)) and isinstance(ft_val, (int, float)):
                    delta = ft_val - v_val
                    pct = (delta / v_val * 100) if v_val != 0 else 0
                    lines.append(
                        f"| {m} | {v_val:.3f} | {ft_val:.3f} | "
                        f"{'+' if delta >= 0 else ''}{delta:.3f} ({pct:+.1f}%) |"
                    )

    # Summary stats
    lines.append("\n### Summary\n")
    total = len(states)
    done = sum(1 for s in states.values() if s.get("status") == "done")
    failed = sum(1 for s in states.values() if s.get("status") == "failed")
    pending = sum(1 for s in states.values() if s.get("status") in ("pending", "running"))

    lines.append(f"- **Total experiments:** {total}")
    lines.append(f"- **Completed:** {done}")
    lines.append(f"- **Failed:** {failed}")
    lines.append(f"- **Pending/Running:** {pending}")

    total_hours = sum(
        s.get("duration_hours", 0) or 0
        for s in states.values()
    )
    lines.append(f"- **Total compute time:** {total_hours:.1f} hours")

    # Failed experiments detail
    failed_exps = [
        (eid, s) for eid, s in states.items() if s.get("status") == "failed"
    ]
    if failed_exps:
        lines.append("\n### Failed Experiments\n")
        for eid, state in failed_exps:
            lines.append(
                f"- **{eid}**: {state.get('error', 'unknown error')} "
                f"(failed at: {state.get('failed_step', 'unknown')})"
            )

    report = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    log.info(f"Report written to {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--output", type=Path, help="Output path")
    args = parser.parse_args()

    generate_report(args.output)


if __name__ == "__main__":
    main()
