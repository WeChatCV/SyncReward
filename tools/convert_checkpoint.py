#!/usr/bin/env python3
"""Convert a training checkpoint (`training_state.pt`) into the released format.

  python tools/convert_checkpoint.py --stage 2 --input outputs/stage2/best --output checkpoints/syncreward-stage2
  python tools/convert_checkpoint.py --stage 1 --input outputs/stage1/checkpoint-step-5500 \
      --output checkpoints/syncreward-stage1

The output directory holds `model.safetensors` (fp32, weights only) and `config.json`. Stage-2 output
contains the full reward model (frozen encoder included) and is what `evaluate.py` / `score.py` load;
Stage-1 output contains the PE-AV towers + projection heads and can be passed as `--stage1_checkpoint`.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syncreward.model import (DEFAULT_ENCODER, PeAVEncoder, SyncRewardModel, remap_legacy_keys,  # noqa: E402
                              save_pretrained)

STAGE2_ARCH = dict(num_segments=18, sync_dim=768, num_layers=3, num_heads=8, dropout=0.1, max_reward=2.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], required=True)
    p.add_argument("--input", required=True, help="Checkpoint dir or training_state.pt.")
    p.add_argument("--output", required=True)
    p.add_argument("--encoder_path", default=DEFAULT_ENCODER, help="Used to verify the state dict loads strictly.")
    p.add_argument("--no_verify", action="store_true")
    args = p.parse_args()

    path = Path(args.input)
    path = path / "training_state.pt" if path.is_dir() else path
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model_state_dict"]
    meta = {"source_global_step": int(state.get("global_step", -1)), "source_epoch": int(state.get("epoch", -1))}

    if args.stage == 2:
        weights = remap_legacy_keys(weights)
        config = {"model_type": "syncreward", "stage": 2, "encoder": "facebook/pe-av-large", **STAGE2_ARCH,
                  "segment_duration": 0.5, "segment_stride": 0.25, "crop": "center", **meta}
    else:
        log_temp = state.get("loss_fn_state_dict", {}).get("log_temp")
        config = {"model_type": "syncreward-stage1-encoder", "stage": 1, "encoder": "facebook/pe-av-large",
                  "num_segments": 18, "segment_duration": 0.5, "embedding": "mean(last_hidden_state) -> head -> L2",
                  "temperature": math.exp(float(log_temp)) if log_temp is not None else None, **meta}
    weights = {k: v.float() for k, v in weights.items()}

    if not args.no_verify:
        encoder = PeAVEncoder.from_pretrained(args.encoder_path)
        module = SyncRewardModel(encoder, **STAGE2_ARCH) if args.stage == 2 else encoder
        module.load_state_dict(weights, strict=True)
        print(f"verified: {len(weights)} tensors load strictly into the stage-{args.stage} module")

    save_pretrained(weights, args.output, config)
    size = os.path.getsize(Path(args.output) / "model.safetensors") / 1e9
    print(f"wrote {args.output} ({size:.2f} GB, {sum(v.numel() for v in weights.values()) / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
