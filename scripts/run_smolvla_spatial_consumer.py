#!/usr/bin/env python3
"""Run the bounded spatial-representation test against verified camera service inputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_smolvla_vision_service_matrix import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--toolkit-image-id", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Positive repeats required")
    if args.stage_dir and not args.split_dir:
        parser.error("Stage comparison requires the matched split controls")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "session.json"
    if output.exists():
        parser.error("Refusing to overwrite an existing session")
    os.sched_setaffinity(0, {4, 5, 6, 7})
    service = json.loads((args.inputs_dir / "session.json").read_text())
    if service["status"] != "complete":
        raise ValueError("Complete camera service session required")
    if any(sha256(Path(path)) != value for path, value in service["artifact_sha256"].items()):
        raise ValueError("Vision graph changed since camera service validation")
    candidates = sorted(args.candidate_dir.glob("*.rknn"))
    if len(candidates) != 5:
        raise ValueError("Expected recompilation control plus four spatial variants")
    report = {"scope": "Bounded native-layout spatial factoring, actual last-layer consumer only; not complete policy or fused vision bit parity",
              "status": "running", "affinity": sorted(os.sched_getaffinity(0)), "repeats": args.repeats,
              "inputs_session_sha256": sha256(args.inputs_dir / "session.json"),
              "fixtures": service["fixtures"], "export": json.loads((args.candidate_dir / "manifest.json").read_text()),
              "binary_sha256": sha256(args.binary), "script_sha256": sha256(Path(__file__)),
              "toolkit_image_id": args.toolkit_image_id,
              "candidate_rknn_sha256": {path.name: sha256(path) for path in candidates}, "cases": {}}
    if args.split_dir:
        split_graphs = sorted(args.split_dir.glob("*.rknn"))
        if len(split_graphs) != 6:
            raise ValueError("Expected flat normalization prefix plus five MLP variants")
        report["split_export"] = json.loads((args.split_dir / "manifest.json").read_text())
        report["split_candidate_rknn_sha256"] = {path.name: sha256(path) for path in split_graphs}
        report["split_timing_scope"] = "Combined prefix + MLP calls, including the extra graph invocation; SDK profile/allocation describes only MLP"
    if args.stage_dir:
        stage_graphs = sorted(args.stage_dir.glob("*.rknn"))
        if len(stage_graphs) != 4:
            raise ValueError("Expected four in-graph stage spatial variants")
        report["stage_export"] = json.loads((args.stage_dir / "manifest.json").read_text())
        report["stage_candidate_rknn_sha256"] = {path.name: sha256(path) for path in stage_graphs}
    try:
        for name, case in service["cases"].items():
            images = [args.inputs_dir / f"{Path(name).stem}_camera{camera}_fp32.bin" for camera in range(2)]
            for path in images:
                if sha256(path) != case["prepared_camera_sha256"][path.name]:
                    raise ValueError("Camera input changed")
            raw = args.output_dir / f"{Path(name).stem}.json"
            command = [sys.executable, str(args.sampler), "--output", str(raw.with_suffix(".load.json")),
                       "--stdout-log", str(raw.with_suffix(".log")), "--system-telemetry", "--quiet", "--",
                       str(args.binary), str(args.vision_dir), str(args.candidate_dir), *map(str, images), str(args.repeats), str(raw)]
            if args.split_dir:
                command.append(str(args.split_dir))
            if args.stage_dir:
                command.append(str(args.stage_dir))
            print(f"Spatial representation test: {name}", flush=True)
            subprocess.run(command, check=True)
            result = json.loads(raw.read_text())
            if not result["post_timing_and_restored_camera_checks"]:
                raise ValueError("Missing state audit")
            report["cases"][name] = result
            output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Spatial representation session complete: {output}", flush=True)


if __name__ == "__main__":
    main()
