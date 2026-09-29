#!/usr/bin/env python3
"""Audit complete-boundary alternatives without pooling independent sessions."""

import argparse
import json
from pathlib import Path

import numpy as np


def paired(plans, baseline, candidate):
    a = np.asarray(plans[baseline]["samples_ms"])[:, -1]
    b = np.asarray(plans[candidate]["samples_ms"])[:, -1]
    if a.shape != b.shape:
        raise ValueError("Unpaired sample counts")
    gain = a - b
    return {"baseline": baseline, "candidate": candidate,
            "median_total_reduction_percent": float(100 * (np.median(a) - np.median(b)) / np.median(a)),
            "paired_median_gain_ms": float(np.median(gain)), "wins": int(np.count_nonzero(gain > 0)),
            "repeats": len(gain)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "Same actual cross-attention boundary, no full-flow or closed-loop estimate; sessions not pooled",
              "sessions": {}}
    for name in args.sessions:
        data = json.loads((args.directory / (name + ".json")).read_text())
        load = json.loads((args.directory / (name + "_load.json")).read_text())
        measured = [v for v in load["samples"] if v["phase"] == "measure"]
        plans = data["plans"]["10"]
        pairs = [("ready_joint", "restore_o"), ("ready_joint", "split_parallel_heads"),
                 ("propagated_ready", "split_parallel_heads"), ("split_serial_heads", "split_parallel_heads"),
                 ("split_parallel_heads", "compact_split_parallel"),
                 ("split_parallel_heads", "original_order_compact_parallel"),
                 ("split_parallel_heads", "compact_parallel_cpu_restore"),
                 ("compact_split_parallel", "compact_parallel_cpu_restore"),
                 ("compact_full_cpu_restore", "compact_parallel_cpu_restore"),
                 ("split_parallel_heads", "compact_parallel_cpu_both"),
                 ("compact_parallel_cpu_restore", "compact_parallel_cpu_both"),
                 ("compact_full_cpu_both", "compact_parallel_cpu_both")]
        record = {
            "followup_opt3": data.get("followup_opt3", {"enabled": False}),
            "totals_ms": {count: {n: v["median_ms"][-1] for n, v in variants.items()} for count, variants in data["plans"].items()},
            "ten_consumption_p95_ms": {n: v["p95_ms"][-1] for n, v in plans.items()},
            "ten_consumption_stage_medians_ms": {n: v["median_ms"] for n, v in plans.items()},
            "comparisons": [paired(plans, a, b) for a, b in pairs if b in plans],
            "validation": data["validation"], "stress": data["stress"]["summary"],
            "prefix_effect": data["prefix_effect"], "restore_check": data["restore_check"],
            "clocks": {"samples": len(measured), "npu_hz": sorted({v["npu_hz"] for v in measured}),
                       "dram_hz": sorted({v["dram_hz"] for v in measured})},
            "model_sha256": data["model_sha256"], "library_sha256": data["library_sha256"],
        }
        report["sessions"][name] = record
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({n: {"totals10": v["totals_ms"]["10"], "comparisons": v["comparisons"], "clocks": v["clocks"]}
                      for n, v in report["sessions"].items()}, indent=2))


if __name__ == "__main__":
    main()
