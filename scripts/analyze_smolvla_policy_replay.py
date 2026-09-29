#!/usr/bin/env python3
"""Keep per-input/session policy deltas separate; never sum stage medians."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {"note": "No session/input pooling. Paired means/gains describe the recorded calls, not an input-population or success-rate estimate.", "sessions": {}}
    for path in args.sessions:
        report = json.loads(path.read_text())
        cases = {}
        for name, plans in report["cases"].items():
            totals = {plan: np.asarray([sample["total_ms"] for sample in value["samples"]]) for plan, value in plans.items()}
            deltas = {}
            for candidate, baseline in (("original_resident", "deployed_lite"), ("hoisted_consumer_ready", "original_resident")):
                delta = totals[baseline] - totals[candidate]
                deltas[f"{candidate}_vs_{baseline}"] = {"paired_median_ms": float(np.median(delta)),
                    "paired_mean_ms": float(np.mean(delta)), "wins": int(np.sum(delta > 0)), "trials": len(delta),
                    "median_reduction_percent": float(100 * (1 - np.median(totals[candidate]) / np.median(totals[baseline])))}
            cases[name] = {"comparisons": deltas, "plans": {}}
            for plan, value in plans.items():
                means = {key: float(np.mean([sample[key] for sample in value["samples"]]))
                         for key in ("vision_ms", "prefill_ms", "denoise_ms", "other_ms", "total_ms")}
                cases[name]["plans"][plan] = {"median_total_ms": value["median_ms"]["total_ms"],
                    "mean_stage_ms": means,
                    "fraction_of_mean_total_percent": {key: 100 * mean / means["total_ms"] for key, mean in means.items() if key != "total_ms"}}
        output["sessions"][path.name] = {"vision_control": report["vision_control"], "cases": cases}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
