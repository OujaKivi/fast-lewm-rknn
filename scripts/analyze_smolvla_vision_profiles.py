#!/usr/bin/env python3
"""Group SDK operator times without interpreting RW as real DRAM traffic."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def category(operation, name):
    if "patch_embedding" in name:
        return "patch_embedding"
    if operation == "exSDPAttention":
        return "attention"
    if "/self_attn/" in name and any(f"/{projection}_proj/" in name for projection in "qkv"):
        return "qkv_projection"
    if "/self_attn/out_proj/" in name:
        return "attention_output_projection"
    if "/mlp/fc1/" in name:
        return "mlp_fc1_activation"
    if "/mlp/fc2/" in name:
        return "mlp_fc2"
    if "layer_norm" in name:
        return "normalization"
    if "connector" in name:
        return "connector"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "SDK reported operator times, not end-to-end latency or measured DRAM bandwidth", "profiles": {}}
    for path in args.profiles:
        groups = defaultdict(lambda: {"operators": 0, "time_us": 0})
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8 or not fields[0].isdigit():
                continue
            group = groups[category(fields[1], fields[-1])]
            group["operators"] += 1
            group["time_us"] += int(fields[7])
        total = sum(group["time_us"] for group in groups.values())
        for group in groups.values():
            group["share_percent"] = group["time_us"] / total * 100
        report["profiles"][path.name] = {"total_us": total, "groups": dict(groups)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
