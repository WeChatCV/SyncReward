#!/usr/bin/env bash
# Reproduce the paper's SyncReward-Bench result with a released checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (dir with syncreward_bench.jsonl)}"
VIDEO_ROOT="${VIDEO_ROOT:?set VIDEO_ROOT (root of the relative video paths)}"
CKPT="${CKPT:-checkpoints/syncreward-stage2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" --master_port="${MASTER_PORT:-29610}" \
  evaluate.py --config configs/eval.yaml \
  --checkpoint "${CKPT}" \
  --data "${DATA_ROOT}/syncreward_bench.jsonl" --video_root "${VIDEO_ROOT}" \
  --output "${OUTPUT_DIR}/syncreward_bench.predictions.jsonl" \
  --metrics_output "${OUTPUT_DIR}/syncreward_bench.metrics.json" \
  "$@"
