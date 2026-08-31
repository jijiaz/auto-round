# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step 6: apply the pre-registered acceptance gates and select the block size.

Reads the JSONL records produced by ``run_reference_sweep.py``, aggregates them
per (method, block size), applies the gates frozen in ``contract.py``, and emits
both a human-readable table and a machine-readable summary.

The gates are never derived from the sweep itself: the absolute floor and the
relative-to-BF16 budget live in ``contract.py`` and must be committed before the
full sweep starts.

Usage::

    python aggregate_results.py records/model_level_records.jsonl --output records/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .contract import ABSOLUTE_GATES, RELATIVE_GATES_VS_BF16
except ImportError:  # executed as a standalone script
    from contract import ABSOLUTE_GATES, RELATIVE_GATES_VS_BF16

METRICS = ("clip", "clip-iqa", "imagereward")


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["run"]].append(record)

    aggregated: dict[str, dict] = {}
    for run, run_records in grouped.items():
        stats: dict[str, dict] = {}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in run_records if metric in r.get("metrics", {})]
            if not values:
                continue
            stats[metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "worst": min(values),
                "runs": len(values),
            }
        first = run_records[0]
        aggregated[run] = {
            "method": first["method"],
            "smooth": first.get("smooth"),
            "terminal": first.get("terminal"),
            "weight_block": first.get("weight_block"),
            "act_group_size": first.get("act_group_size"),
            "metrics": stats,
            "wall_time_s": statistics.fmean([r.get("wall_time_s", 0.0) for r in run_records]),
            "peak_memory_bytes": max((r.get("peak_memory_bytes") or 0) for r in run_records) or None,
        }
    return aggregated


def apply_gates(aggregated: dict[str, dict]) -> dict[str, dict]:
    """Annotate every run with absolute and relative gate outcomes."""
    bf16 = next((value for value in aggregated.values() if value["method"] == "bf16"), None)
    mxfp4 = next((value for value in aggregated.values() if value["method"] == "mxfp4"), None)

    for run, value in aggregated.items():
        outcomes = {}
        for metric in METRICS:
            stats = value["metrics"].get(metric)
            if stats is None:
                outcomes[metric] = {"pass": None, "reason": "missing"}
                continue
            absolute_pass = stats["mean"] >= ABSOLUTE_GATES[metric]
            worst_pass = stats["worst"] >= ABSOLUTE_GATES[metric]
            relative_pass = None
            delta_bf16 = None
            if bf16 is not None and metric in bf16["metrics"]:
                delta_bf16 = stats["mean"] - bf16["metrics"][metric]["mean"]
                relative_pass = delta_bf16 >= RELATIVE_GATES_VS_BF16[metric]
            delta_mxfp4 = None
            if mxfp4 is not None and metric in mxfp4["metrics"]:
                delta_mxfp4 = stats["mean"] - mxfp4["metrics"][metric]["mean"]
            outcomes[metric] = {
                "absolute_pass": absolute_pass,
                "worst_run_pass": worst_pass,
                "relative_pass": relative_pass,
                "delta_vs_bf16": delta_bf16,
                "delta_vs_mxfp4": delta_mxfp4,
                "pass": absolute_pass and worst_pass and (relative_pass is not False),
            }
        value["gates"] = outcomes
        value["passed"] = all(outcome.get("pass") for outcome in outcomes.values())
    return aggregated


def select_block_size(aggregated: dict[str, dict]) -> dict:
    """Largest passing block size, with a monotonicity check on smaller sizes."""
    candidates = [
        value
        for value in aggregated.values()
        if value["method"] == "int4" and value["weight_block"] is not None and value["passed"]
    ]
    if not candidates:
        return {"selected": None, "reason": "no INT4 candidate passed all gates"}

    def block_elements(value) -> int:
        block = value["weight_block"]
        if isinstance(block, list):
            return block[0] * block[1]
        return 1 << 30 if block == -1 else block

    best = max(candidates, key=block_elements)
    failures_below = [
        value["weight_block"]
        for value in aggregated.values()
        if value["method"] == "int4"
        and value["weight_block"] is not None
        and not value["passed"]
        and block_elements(value) < block_elements(best)
    ]
    return {
        "selected": best["weight_block"],
        "elements_per_scale": block_elements(best),
        "run": next(run for run, value in aggregated.items() if value is best),
        "non_monotonic_failures_below": failures_below,
        "reason": (
            "largest block size passing all three model-level gates"
            if not failures_below
            else "largest passing block size, but smaller sizes failed - rerun and explain before accepting"
        ),
    }


def format_table(aggregated: dict[str, dict]) -> str:
    header = f"{'run':<40}{'block':>12}{'CLIP':>10}{'CLIP-IQA':>12}{'ImageReward':>14}{'verdict':>10}"
    lines = [header, "-" * len(header)]
    for run, value in sorted(aggregated.items()):
        block = value["weight_block"]
        block_text = f"{block[0]}x{block[1]}" if isinstance(block, list) else str(block)

        def cell(metric: str) -> str:
            stats = value["metrics"].get(metric)
            return f"{stats['mean']:.4f}" if stats else "-"

        verdict = "PASS" if value.get("passed") else "fail"
        if value["method"] in ("bf16", "mxfp4"):
            verdict = "control"
        lines.append(
            f"{run:<40}{block_text:>12}{cell('clip'):>10}{cell('clip-iqa'):>12}"
            f"{cell('imagereward'):>14}{verdict:>10}"
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    records = load_records(args.records)
    aggregated = apply_gates(aggregate(records))
    selection = select_block_size(aggregated)
    print(format_table(aggregated))
    print("\nselected block size:", json.dumps(selection, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"aggregated": aggregated, "selection": selection}, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
