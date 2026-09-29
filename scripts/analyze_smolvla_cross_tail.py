#!/usr/bin/env python3
"""Summarize tail stages and sampled clocks without claiming a causal driver fix."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {"scope": "Original synthetic-prefix C17 boundary, 500 rotated calls per mode/session; instrumented diagnostics, not a full-policy or causal hardware attribution", "sessions": {}}
    for path in args.sessions:
        report = json.loads(path.read_text())
        load = json.loads(path.with_name(path.stem + "_load.json").read_text())
        clock_paths = list(load["samples"][0]["system"])
        clocks = {name: [min(sample["system"][name] for sample in load["samples"]),
                         max(sample["system"][name] for sample in load["samples"])] for name in clock_paths}
        record = {"sampled_system_min_max": clocks, "npu_hz": sorted({s["npu_hz"] for s in load["samples"]}),
                  "dram_hz": sorted({s["dram_hz"] for s in load["samples"]}), "plans": {}}
        for name, plan in report["plans"]["10"].items():
            samples = np.asarray(plan["samples_ms"])
            slow = samples[:, -1] >= np.percentile(samples[:, -1], 95)
            fast = samples[:, -1] < np.percentile(samples[:, -1], 50)
            telemetry = report["call_telemetry"]["10"][name]
            values = {"total_percentiles_ms": dict(zip(("p50", "p95", "p99"), np.percentile(samples[:, -1], [50, 95, 99]).tolist())),
                      "all_stage_median_ms": np.median(samples, axis=0).tolist(),
                      "slowest_5pct_stage_median_ms": np.median(samples[slow], axis=0).tolist(),
                      "slowest_5pct_involuntary_switches": [telemetry[index]["involuntary_switches"] for index in np.flatnonzero(slow)]}
            if report.get("native_trace_columns_ms"):
                traces = np.asarray([call["native_steps_ms"] for call in telemetry])
                values.update(native_trace_columns_ms=report["native_trace_columns_ms"],
                              fast_step_median_ms=np.median(traces[fast], axis=(0, 1)).tolist(),
                              slow_step_median_ms=np.median(traces[slow], axis=(0, 1)).tolist(),
                              maximum_worker_wake_ms=float(np.max(traces[:, :, 5:8])))
            record["plans"][name] = values
        baseline = np.asarray(report["plans"]["10"]["split_parallel_heads"]["samples_ms"])[:, -1]
        candidate = np.asarray(report["plans"]["10"]["compact_parallel_cpu_both"]["samples_ms"])[:, -1]
        record["paired"] = {"wins": int(np.sum(candidate < baseline)), "trials": len(baseline),
                            "median_gain_ms": float(np.median(baseline - candidate)),
                            "median_reduction_percent": float(100 * (1 - np.median(candidate) / np.median(baseline)))}
        output["sessions"][path.name] = record
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {"paired": value["paired"], "percentiles": {plan: result["total_percentiles_ms"] for plan, result in value["plans"].items()}}
                      for name, value in output["sessions"].items()}, indent=2))


if __name__ == "__main__":
    main()
