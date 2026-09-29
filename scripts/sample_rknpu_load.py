#!/usr/bin/env python3
"""Sample driver busy counters while running a marked graph microbenchmark."""

import argparse
import json
import re
import subprocess
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-telemetry", action="store_true")
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    phase = ["setup"]
    plan = [None]
    lines = []
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def consume():
        log = args.stdout_log.open("w") if args.stdout_log else None
        try:
            for line in process.stdout:
                lines.append(line.rstrip())
                if log:
                    log.write(line)
                    log.flush()
                if line.startswith("PROBE_MEASURE_START"):
                    phase[0] = "measure"
                elif line.startswith("PROBE_MEASURE_END"):
                    phase[0] = "done"
                elif line.startswith("PROBE_PLAN_START "):
                    plan[0] = line.strip().split(" ", 1)[1]
                elif line.startswith("PROBE_PLAN_END"):
                    plan[0] = None
        finally:
            if log:
                log.close()

    reader = threading.Thread(target=consume)
    reader.start()
    samples = []
    while process.poll() is None:
        load = Path("/sys/kernel/debug/rknpu/load").read_text().strip()
        samples.append({
            "monotonic_ns": time.monotonic_ns(),
            "seconds": time.monotonic() - start,
            "phase": phase[0],
            "plan": plan[0],
            "core_busy_percent": [int(value) for value in re.findall(r"Core\d:\s*(\d+)%", load)],
            "npu_hz": int(Path("/sys/class/devfreq/fdab0000.npu/cur_freq").read_text()),
            "dram_hz": int(Path("/sys/class/devfreq/dmc/cur_freq").read_text()),
        })
        if args.system_telemetry:
            paths = list(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"))
            paths += list(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
            samples[-1]["system"] = {str(path): int(path.read_text()) for path in paths}
            samples[-1]["loadavg"] = Path("/proc/loadavg").read_text().strip()
        time.sleep(0.05)
    reader.join()
    record = {"command": command, "exit_code": process.returncode, "stdout": lines,
              "samples": samples,
              "system_telemetry": args.system_telemetry,
              "note": "50 ms snapshots; no frequency lock. Driver busy counters have an unknown averaging window; not MAC occupancy or measured DRAM bandwidth. Short per-call throttling/scheduler events cannot be excluded by these snapshots."}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"exit_code": process.returncode, "samples": len(samples),
                      **({} if args.quiet else {"stdout": lines})}))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
