#!/usr/bin/env python3
"""Summarize paired conditioning controls without pooling distinct sessions."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "Per-session paired descriptive audit, not inference over independent deployments or robot tasks",
              "bootstrap": "20,000 paired resamples of within-session iteration gains; seed 812; session effects not modeled",
              "full_sessions": {}, "fragment": {}, "clock_samples": {}}
    for filename in ("full_flow.json", "full_flow_repeat.json"):
        record = json.loads((args.directory / filename).read_text())
        plans = record["plans"]
        cached = np.asarray(plans["hoisted_consumer_ready"]["samples_ms"])
        summaries = {}
        for name in ("original_resident", "original_native_passthrough"):
            baseline = np.asarray(plans[name]["samples_ms"])
            if baseline.shape != cached.shape:
                raise ValueError("Unpaired measurements")
            gains = baseline - cached
            generator = np.random.default_rng(812)
            medians = np.median(gains[generator.integers(0, len(gains), (20000, len(gains)))], axis=1)
            summaries[name] = {"baseline_median_ms": float(np.median(baseline)),
                               "cached_median_ms": float(np.median(cached)),
                               "median_reduction_percent": float((1 - np.median(cached) / np.median(baseline)) * 100),
                               "paired_median_gain_ms": float(np.median(gains)),
                               "paired_median_bootstrap95_ms": np.percentile(medians, [2.5, 97.5]).tolist(),
                               "faster_iterations": int((gains > 0).sum()), "iterations": len(gains)}
        result["full_sessions"][filename] = summaries
    fragment = json.loads((args.directory / "fragment_ready_core7.json").read_text())
    result["fragment"] = {steps: {name: {"total_median_ms": values["median_ms"][-1],
                                        "preparation_median_ms": values["median_ms"][0],
                                        "consumer_run_median_ms": values["median_ms"][2]}
                                   for name, values in plans.items()} for steps, plans in fragment["plans"].items()}
    for filename in ("full_flow_load.json", "full_flow_repeat_load.json", "fragment_ready_core7_load.json"):
        record = json.loads((args.directory / filename).read_text())
        samples = [value for value in record["samples"] if value["phase"] == "measure"]
        if not samples or record["exit_code"] != 0:
            raise ValueError("Missing successful measurement phase")
        result["clock_samples"][filename] = {"measure_samples": len(samples),
                                             "npu_hz": sorted({value["npu_hz"] for value in samples}),
                                             "dram_hz": sorted({value["dram_hz"] for value in samples})}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["full_sessions"], indent=2))


if __name__ == "__main__":
    main()
