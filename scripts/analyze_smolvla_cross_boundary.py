#!/usr/bin/env python3
"""Within-session paired audit; independent sessions are never pooled."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = "compact_grouped_propagated_ready"
    baseline = "ready_joint"
    report = {"scope": "Single actual cross-attention block, independent hidden activations; no full-model/closed-loop estimate",
              "bootstrap": "20000 paired resamples of iterations within each session, seed 929; descriptive only, not task/device/thermal inference",
              "sessions": {}}
    for session in args.sessions:
        data = json.loads((args.directory / (session + ".json")).read_text())
        plans = data["plans"]["10"]
        anchor = np.asarray(plans[baseline]["samples_ms"])[:, -1]
        changed = np.asarray(plans[candidate]["samples_ms"])[:, -1]
        gain = anchor - changed
        rng = np.random.default_rng(929)
        samples = rng.integers(0, len(gain), size=(20000, len(gain)))
        bootstrap = np.median(gain[samples], axis=1)
        graphs = data["graphs"]
        load = json.loads((args.directory / (session + "_load.json")).read_text())
        measured = [value for value in load["samples"] if value["phase"] == "measure"]
        record = {
            "compiler_controls_opt3": data.get("compiler_controls_opt3", False),
            "total_medians_ms": {count: {name: value["median_ms"][-1] for name, value in variants.items()}
                                 for count, variants in data["plans"].items()},
            "ten_consumption_p95_ms": {name: value["p95_ms"][-1] for name, value in plans.items()},
            "candidate_vs_ready": {"median_total_reduction_percent": float(100 * (np.median(anchor) - np.median(changed)) / np.median(anchor)),
                                   "paired_median_gain_ms": float(np.median(gain)),
                                   "paired_bootstrap_interval_ms": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                                   "wins": int(np.count_nonzero(gain > 0)), "repeats": len(gain)},
            "native_conditioning_bytes": {"expanded_ready": sum(value["bytes"] for value in graphs["prep_ready"]["native_outputs"]),
                                          "compact_ready": sum(value["bytes"] for value in graphs["group_prepare"]["native_outputs"])},
            "validation": {case: value["plans_vs_original"][candidate] for case, value in data["validation"].items()},
            "synthetic_numerical_stress": data.get("synthetic_numerical_stress", {}).get("summary"),
            "clocks": {"measurement_samples": len(measured),
                       "npu_hz": sorted({value["npu_hz"] for value in measured}),
                       "dram_hz": sorted({value["dram_hz"] for value in measured})},
            "model_sha256": data["model_sha256"],
        }
        report["sessions"][session] = record
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({session: {"gain": value["candidate_vs_ready"], "totals": value["total_medians_ms"]["10"],
                               "clocks": value["clocks"]} for session, value in report["sessions"].items()}, indent=2))


if __name__ == "__main__":
    main()
