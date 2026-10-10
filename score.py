#!/usr/bin/env python3
"""Score the audio-visual synchronization of video files with SyncReward (0 = unsynced, 2 = well synced)."""

import argparse
import json

import torch

from syncreward.data import MIN_AUDIO_SAMPLES, MIN_VIDEO_FRAMES, load_av
from syncreward.model import DEFAULT_ENCODER, build_reward_model, load_processor, prepare_inputs, processor_sample_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="+", help="Video files with an audio track.")
    p.add_argument("--checkpoint", required=True, help="Stage-2 checkpoint dir (or training_state.pt).")
    p.add_argument("--encoder_path", default=DEFAULT_ENCODER)
    p.add_argument("--num_segments", type=int, default=18)
    p.add_argument("--segment_duration", type=float, default=0.5)
    p.add_argument("--segment_stride", type=float, default=0.25)
    p.add_argument("--crop", default="center", choices=["center", "random", "random_independent"])
    p.add_argument("--encode_batch_size", type=int, default=18)
    p.add_argument("--output", default=None, help="Optional jsonl output.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    processor = load_processor(args.encoder_path)
    sr = processor_sample_rate(processor)
    model = build_reward_model(args.encoder_path, checkpoint=args.checkpoint, num_segments=args.num_segments)
    model.to(device).eval()

    results = []
    for path in args.videos:
        video, fps, audio = load_av(path, sr)
        if video.shape[0] < MIN_VIDEO_FRAMES or audio.shape[0] < MIN_AUDIO_SAMPLES:
            print(f"{path}\tskipped (too short)")
            continue
        inputs = prepare_inputs(processor, {"video": [video], "audio": [audio], "fps": [fps]}, args, args.crop, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reward = float(model(inputs, encode_batch_size=args.encode_batch_size).float()[0])
        results.append({"video_path": path, "reward": reward})
        print(f"{path}\t{reward:.4f}", flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in results)


if __name__ == "__main__":
    main()
