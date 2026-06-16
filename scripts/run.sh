#!/usr/bin/env bash
set -euo pipefail
# 双卡启动封装：NPROC 默认 2。所有参数透传给 train/sft.py
NPROC="${NPROC:-2}"
cd "$(dirname "$0")/.."
torchrun --standalone --nproc_per_node="${NPROC}" train/sft.py "$@"
