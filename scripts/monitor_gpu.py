from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dft.monitor import GPU_FIELDS, parse_nvidia_smi_line

_QUERY = ("timestamp,index,utilization.gpu,memory.used,"
          "memory.total,temperature.gpu,power.draw")


def _sample() -> list[dict]:
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={_QUERY}",
         "--format=csv,noheader,nounits"]).decode()
    return [parse_nvidia_smi_line(l) for l in out.strip().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gpu_log.csv")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0=直到 Ctrl-C")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t_end = time.time() + args.duration if args.duration > 0 else None
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GPU_FIELDS)
        w.writeheader()
        try:
            while t_end is None or time.time() < t_end:
                for row in _sample():
                    w.writerow(row)
                f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
