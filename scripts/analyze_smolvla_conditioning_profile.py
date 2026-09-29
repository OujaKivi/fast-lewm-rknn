#!/usr/bin/env python3
"""Audit compiled conditioning shapes; SDK time is not an E2E savings bound."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def classify(operation, input_shape, output_shape, name):
    projection = re.fullmatch(r"Conv:/[kv]_proj_(\d+)/MatMul#2", name)
    if operation == "Conv" and projection and int(projection[1]) % 2:
        if input_shape != "(1,320,1,149),(960,320,1,1)" or output_shape != "(1,960,1,149)":
            raise ValueError(f"Unexpected compiled cross projection: {name}")
        return "cross_projection_reported_width_960"
    if operation == "Reshape" and input_shape == "(1,149,5,64),(4)" and output_shape == "(1,149,1,320)":
        return "cross_prefix_reshape"
    if operation == "Transpose" and input_shape == "(1,149,1,320)" and output_shape == "(1,320,1,149)":
        return "cross_prefix_transpose"
    raw = re.fullmatch(r"Transpose:(key|value)_(\d+)_tp", name)
    if operation == "Transpose" and raw and int(raw[2]) % 2 == 0:
        return "self_static_prefix_transpose"
    if operation == "exSDPAttention":
        return "attention"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--logical-projection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logical = json.loads(args.logical_projection_report.read_text())
    if logical["projection_input_shapes"] != ["[1, 149, 320]"] or logical["projection_weight_shapes"] != ["[320, 320]"]:
        raise ValueError("Unexpected logical projection geometry")
    groups = defaultdict(lambda: {"operators": 0, "time_us": 0, "operator_ids": []})
    rows = []
    for line in args.profile.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].isdigit():
            continue
        category = classify(fields[1], fields[4], fields[5], fields[-1])
        group = groups[category]
        group["operators"] += 1
        group["time_us"] += int(fields[7])
        group["operator_ids"].append(int(fields[0]))
        if category.startswith("cross_projection"):
            rows.append({"id": int(fields[0]), "name": fields[-1],
                         "input_shape": fields[4], "output_shape": fields[5],
                         "sdk_time_us": int(fields[7])})
    for category in ("cross_projection_reported_width_960", "cross_prefix_reshape",
                     "cross_prefix_transpose", "self_static_prefix_transpose", "attention"):
        if groups[category]["operators"] != 32:
            raise ValueError(f"Expected 32 operators in {category}")
    total = sum(group["time_us"] for group in groups.values())
    static = sum(groups[name]["time_us"] for name in (
        "cross_projection_reported_width_960", "cross_prefix_reshape",
        "cross_prefix_transpose", "self_static_prefix_transpose"))
    record = {
        "scope": "Existing SDK per-operator profile plus prior logical CPU projection audit; no new timing run",
        "warning": "Category sums are reported SDK time, not E2E savings, hardware MAC counts, actual peak memory, or measured DRAM traffic",
        "profile": str(args.profile), "logical_projection_report": str(args.logical_projection_report),
        "groups": dict(groups), "sdk_operator_total_us": total,
        "static_candidate_sdk_time_us": static,
        "logical_projection_width": 320, "reported_compiled_projection_width": 960,
        "reported_width_ratio": 3,
        "root_cause_status": "GQA expansion folded into projection is a hypothesis; compiled weight values and hardware work not independently verified",
        "nominal_dense_projection_macs_per_step_from_shapes": {
            "logical": 32 * 149 * 320 * 320, "reported_compiled": 32 * 149 * 320 * 960},
        "logical_cross_conditioning_fp16_bytes": 32 * 149 * 320 * 2,
        "expanded_cross_conditioning_fp16_bytes_if_materialized": 32 * 149 * 960 * 2,
        "projection_rows": rows,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items()
                      if key not in ("groups", "projection_rows")}, indent=2))


if __name__ == "__main__":
    main()
