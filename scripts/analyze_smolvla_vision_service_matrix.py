#!/usr/bin/env python3
"""Keep camera count, timing boundary, input, and session comparisons separate."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


COMPARISONS = (
    ("packed_h4_camera0", "fused_camera0_mask7"),
    ("packed_h4_camera1", "fused_camera1_mask7"),
    ("packed_full_camera0", "fused_camera0_mask7"),
    ("packed_full_camera1", "fused_camera1_mask7"),
    ("packed_h4_dual_serial", "fused_dual_parallel_mask1_mask2"),
    ("packed_h4_dual_parallel", "fused_dual_parallel_mask1_mask2"),
    ("packed_h4_dual_parallel", "packed_h4_dual_serial"),
)


def paired_metrics(candidate, baseline):
    if not candidate or len(candidate) != len(baseline):
        raise ValueError("Matched nonempty iteration samples required")
    if not all(math.isfinite(value) and value > 0 for value in [*candidate, *baseline]):
        raise ValueError("Positive finite latency required")
    differences = [left - right for left, right in zip(baseline, candidate)]
    return {"candidate_median_ms": statistics.median(candidate),
            "baseline_median_ms": statistics.median(baseline),
            "median_reduction_percent": 100 * (1 - statistics.median(candidate) / statistics.median(baseline)),
            "paired_median_saved_ms": statistics.median(differences),
            "paired_mean_saved_ms": statistics.mean(differences),
            "wins": sum(value > 0 for value in differences), "trials": len(differences)}


def summarize(report):
    if report.get("status") != "complete":
        raise ValueError("Incomplete/failed session is not a formal timing result")
    cases = {}
    if set(report["cases"]) != {case["file"] for case in report["fixtures"]["cases"]}:
        raise ValueError("Complete session must contain every manifest case")
    for name, result in report["cases"].items():
        if not result["schedule_swap_restore_and_post_timing_bitwise_checks"]:
            raise ValueError("Missing state validation")
        plans = result["plans"]
        layers = result["head_vs_full_layer_parity"]
        if len(layers) != 2 or any(len(camera) != 12 for camera in layers):
            raise ValueError("Expected parity results for both cameras and all 12 layers")
        if not all(value["bitwise_equal"] for camera in layers for value in camera):
            raise ValueError("Head/full numerical gate failed")
        if any(len(mode["samples_ms"]) != report["repeats"] for plan in plans.values() for mode in plan["timings"].values()):
            raise ValueError("Incomplete timing repetitions")
        comparisons = {}
        for candidate, baseline in COMPARISONS:
            if plans[candidate]["camera_count"] != plans[baseline]["camera_count"]:
                raise ValueError("Cannot compare different camera counts as an acceleration")
            comparisons[f"{candidate}_vs_{baseline}"] = {
                mode: paired_metrics(plans[candidate]["timings"][mode]["samples_ms"],
                                     plans[baseline]["timings"][mode]["samples_ms"])
                for mode in ("resident_compute", "input_compute_output")}
        cases[name] = {"comparisons": comparisons,
                       "latency_medians_ms": {plan: {mode: value["median_ms"] for mode, value in values["timings"].items()}
                                              for plan, values in plans.items()},
                       "head_vs_full_all_tested_layer_bits_equal": all(value["bitwise_equal"] for camera in result["head_vs_full_layer_parity"] for value in camera),
                       "split_vs_fused_camera_errors": [plans[f"packed_h4_camera{camera}"]["errors_vs_fused"][0] for camera in range(2)],
                       "supplemental_stage_diagnostics": result["supplemental_stage_diagnostics"]}
    return {"fixtures_revision": report["fixtures"]["revision"], "repeats": report["repeats"], "cases": cases}


def telemetry_summary(path):
    record = json.loads(path.read_text())
    if record["exit_code"] != 0:
        raise ValueError("Failed native command")
    samples = [sample for sample in record["samples"] if sample["phase"] == "measure"]
    if not samples:
        raise ValueError("No formal-phase telemetry")
    systems = {}
    for sample in samples:
        for name, value in sample.get("system", {}).items():
            systems.setdefault(name, []).append(value)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "formal_phase_samples": len(samples),
            "npu_hz_observed": sorted({sample["npu_hz"] for sample in samples}),
            "dram_hz_observed": sorted({sample["dram_hz"] for sample in samples}),
            "system_observed_ranges": {name: [min(values), max(values)] for name, values in systems.items()},
            "note": "50 ms snapshots only; driver busy is not MAC occupancy or measured DDR bandwidth."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path)
    args = parser.parse_args()
    report = {"note": "No input/session/mode pooling; paired wins are descriptive, not population significance or closed-loop quality.",
              "sessions": {}}
    for path in args.sessions:
        key = f"{path.parent.name}/{path.name}"
        if key in report["sessions"]:
            raise ValueError("Duplicate session key")
        report["sessions"][key] = summarize(json.loads(path.read_text()))
        if args.telemetry_root:
            for frame, case in report["sessions"][key]["cases"].items():
                source = args.telemetry_root / path.stem / f"{Path(frame).stem}.load.json"
                case["telemetry"] = telemetry_summary(source)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for session, values in report["sessions"].items():
        for frame, case in values["cases"].items():
            comparison = case["comparisons"]["packed_h4_dual_serial_vs_fused_dual_parallel_mask1_mask2"]["input_compute_output"]
            print(f"{session} {frame}: dual head gain={comparison['median_reduction_percent']:.2f}%, "
                  f"paired wins={comparison['wins']}/{comparison['trials']}, "
                  f"layer bits={case['head_vs_full_all_tested_layer_bits_equal']}")


if __name__ == "__main__":
    main()
