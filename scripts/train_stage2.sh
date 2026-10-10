#!/usr/bin/env bash
# Stage 2 human alignment. The paper uses 2 nodes x 8 GPUs (NNODES=2, NODE_RANK=0/1).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (dir with stage2_train.jsonl / stage2_val.jsonl)}"
VIDEO_ROOT="${VIDEO_ROOT:?set VIDEO_ROOT (root of the relative video paths)}"
STAGE1_CKPT="${STAGE1_CKPT:?set STAGE1_CKPT (released syncreward-stage1 dir or a Stage-1 checkpoint dir)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage2}"

torchrun \
  --nnodes="${NNODES:-2}" --nproc_per_node="${NPROC_PER_NODE:-8}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  train_stage2.py --config configs/stage2.yaml \
  --train_jsonl "${DATA_ROOT}/stage2_train.jsonl" \
  --val_jsonl "${DATA_ROOT}/stage2_val.jsonl" \
  --video_root "${VIDEO_ROOT}" \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
