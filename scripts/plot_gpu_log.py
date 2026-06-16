from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/gpu_log.csv")
    ap.add_argument("--out", default="results/figures/gpu_util_mem.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["t"] = range(len(df))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for idx, g in df.groupby("index"):
        ax[0].plot(g["t"], g["utilization_gpu"], label=f"GPU{idx} util%")
        ax[1].plot(g["t"], g["memory_used"], label=f"GPU{idx} mem MB")
    ax[0].set_ylabel("Util %"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_ylabel("Mem MB"); ax[1].set_xlabel("sample"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
