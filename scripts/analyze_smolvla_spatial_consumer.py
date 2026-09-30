#!/usr/bin/env python3
"""Audit whole-consumer shape experiments without adding profile or stage medians."""

import argparse
import json
from pathlib import Path
import re

from analyze_smolvla_vision_service_matrix import paired_metrics, telemetry_summary
from analyze_smolvla_vision_profiles import category


def profile_groups(path):
    groups = {}
    operators = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        operation, name, microseconds = fields[1], fields[-1], int(fields[7])
        group = category(operation, name)
        groups[group] = groups.get(group, 0) + microseconds
        workloads = [field for field in fields[8:-1] if re.fullmatch(r"[\d.]+%/[\d.]+%/[\d.]+%", field)]
        operators.append({"operation": operation, "name": name, "time_us": microseconds,
                          "sdk_workload": workloads[0] if workloads else None})
    if not operators:
        raise ValueError("Empty operator profile")
    return {"sdk_reported_group_time_us": groups, "operators": operators,
            "note": "SDK profiled times/workload only; not production timing, MAC occupancy or physical DDR."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    if session["status"] != "complete" or set(session["cases"]) != {case["file"] for case in session["fixtures"]["cases"]}:
        raise ValueError("Complete session/manifest required")
    output = {"scope": "Per-input paired whole-consumer comparisons, not whole vision/policy or population inference", "cases": {}}
    for name, result in session["cases"].items():
        if not result["post_timing_and_restored_camera_checks"] or len(result["cameras"]) != 2:
            raise ValueError("Missing camera audit")
        case = {"cameras": [], "telemetry": telemetry_summary(args.telemetry_dir / f"{Path(name).stem}.load.json")}
        for camera, record in enumerate(result["cameras"]):
            plans = record["plans"]
            camera_output = {"plans": {}}
            for plan, value in plans.items():
                if len(value["samples_ms"]) != session["repeats"]:
                    raise ValueError("Incomplete repeats")
                camera_output["plans"][plan] = {
                    "bit_gate_vs_original_native": value["errors_vs_original_native"]["bitwise_equal"],
                    "errors_vs_original_native": value["errors_vs_original_native"],
                    "vs_original_native": paired_metrics(value["samples_ms"], plans["original_native"]["samples_ms"]),
                    "sdk_profile": profile_groups(args.profiles_dir / f"{Path(name).stem}.json.camera{camera}.{plan}.perf.txt"),
                    "sdk_allocation_fields": value["sdk_allocation_fields"],
                    "timing_includes_prefix_and_mlp": value.get("timing_includes_prefix_and_mlp", False),
                    "sdk_profile_and_allocation_exclude_prefix": value.get("sdk_profile_and_allocation_exclude_prefix", False),
                }
                if plan.startswith("split_mlp_"):
                    camera_output["plans"][plan]["errors_vs_split_original"] = value["errors_vs_split_original"]
                    camera_output["plans"][plan]["vs_split_original"] = paired_metrics(
                        value["samples_ms"], plans["split_mlp_original"]["samples_ms"])
            case["cameras"].append(camera_output)
        output["cases"][name] = case
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for name, case in output["cases"].items():
        for camera, record in enumerate(case["cameras"]):
            print(name, f"camera{camera}", {plan: round(value["vs_original_native"]["candidate_median_ms"], 3)
                                           for plan, value in record["plans"].items()})


if __name__ == "__main__":
    main()
