#!/usr/bin/env bash
set -euo pipefail
# 断点续传演示：先跑到中途被 kill，再 --resume_from_checkpoint 续训。
# 用法: bash scripts/resume_demo.sh <checkpoint_dir>
cd "$(dirname "$0")/.."
CKPT="${1:?用法: resume_demo.sh outputs/e7-resume/checkpoint-XX}"
NPROC="${NPROC:-2}" torchrun --standalone --nproc_per_node="${NPROC:-2}" \
  train/sft.py --config configs/base.yaml --run_name e7-resume \
  --output_dir outputs/e7-resume --resume_from_checkpoint "${CKPT}"
