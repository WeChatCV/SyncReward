#!/usr/bin/env bash
# Stage 1 contrastive training. Run once per node; set NODE_RANK / MASTER_ADDR for multi-node.
# The paper runs two phases: a first run on AudioSet, then a warm-started run on filtered VGGSound
# (pass --init_from <phase-1 checkpoint dir> for the second phase).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stage1}"

torchrun \
  --nnodes="${NNODES:-1}" --nproc_per_node="${NPROC_PER_NODE:-8}" --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
  train_stage1.py --config configs/stage1.yaml \
  --train_list "${DATA_ROOT}/stage1_train.txt" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
