#!/usr/bin/env python3
"""Evaluate agreement with human sync ratings on SyncReward-Bench.

Two modes:
  1. --checkpoint CKPT --data bench.jsonl        score every clip with SyncReward, then compute metrics.
  2. --predictions preds.jsonl --data bench.jsonl
                                                 evaluate an existing predictions file (any model). Rows are
                                                 joined to --data on `id`, else `video_path`, else (uid, model).

Reports Spearman / Top-1 (N=4) / Pair Acc (the paper's headline metrics) with bootstrap 95% CIs, plus
Pearson, Kendall, MAE and the strict pairwise accuracy.
"""

import argparse
import json
from pathlib import Path

from syncreward.data import read_jsonl
from syncreward.metrics import SEED, bootstrap_ci, compute_metrics
from syncreward.utils import parse_args_with_config


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="Clips jsonl with `video_path` and `target` (see README).")
    p.add_argument("--video_root", default=None, help="Prefix for relative `video_path`s in --data.")
    p.add_argument("--predictions", help="Existing predictions jsonl to evaluate (skips scoring).")
    p.add_argument("--checkpoint", help="Released checkpoint dir (model.safetensors) or a training checkpoint.")
    p.add_argument("--encoder_path", default="facebook/pe-av-large")
    p.add_argument("--output", default=None, help="Where to write predictions jsonl (scoring mode).")
    p.add_argument("--metrics_output", default=None, help="Optional json file for the metrics.")
    p.add_argument("--target_key", default="target")
    p.add_argument("--pred_key", default="reward")
    p.add_argument("--num_segments", type=int, default=18)
    p.add_argument("--segment_duration", type=float, default=0.5)
    p.add_argument("--segment_stride", type=float, default=0.25)
    p.add_argument("--crop", default="center", choices=["center", "random", "random_independent"],
                   help="`center` (default, used in the paper) is deterministic.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encode_batch_size", type=int, default=18)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--top1_n", type=int, default=4, help="Group size for Top-1.")
    p.add_argument("--top1_draws", type=int, default=1000, help="Random partitions averaged for Top-1.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for 95%% CIs (0 = off).")
    p.add_argument("--seed", type=int, default=SEED)
    return p


def score_clips(args):
    import torch
    from torch.utils.data import DataLoader

    from syncreward.data import ClipDataset, ShardSampler, collate
    from syncreward.model import build_reward_model, load_processor, predict, processor_sample_rate
    from syncreward.utils import cleanup_distributed, is_main, setup_distributed

    rank, world, device = setup_distributed(backend="gloo")
    processor = load_processor(args.encoder_path)
    model = build_reward_model(args.encoder_path, checkpoint=args.checkpoint, num_segments=args.num_segments)
    model.to(device).eval()
    dataset = ClipDataset.from_file(args.data, processor_sample_rate(processor), video_root=args.video_root)
    loader = DataLoader(dataset, sampler=ShardSampler(dataset, rank, world), batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate)
    torch.manual_seed(args.seed)
    preds = predict(model, processor, loader, device, args, crop=args.crop)
    main_rank = is_main()
    rows = []
    if main_rank:
        for i, row in enumerate(dataset.rows):
            if i in preds:
                rows.append({**row, args.pred_key: preds[i]})
            else:
                print(f"no prediction for {row['video_path']} (missing or undecodable)", flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            print(f"wrote {len(rows)} predictions to {args.output}", flush=True)
    cleanup_distributed()
    return rows if main_rank else None


def row_key(row, field):
    return row.get(field) if field != "uid_model" else (row.get("uid"), row.get("model"))


def join_targets(pred_rows, data_rows, args):
    """Pair predictions with targets in --data order. Returns (targets, preds)."""
    pred_keys = {f: {row_key(r, f) for r in pred_rows} for f in ("id", "video_path", "uid_model")}
    field = max(pred_keys, key=lambda f: sum(row_key(r, f) in pred_keys[f] for r in data_rows
                                             if row_key(r, f) not in (None, (None, None))))
    preds = {row_key(r, field): float(r[args.pred_key]) for r in pred_rows if r.get(args.pred_key) is not None}
    rows = [r for r in data_rows if r.get(args.target_key) is not None]
    missing = sum(row_key(r, field) not in preds for r in rows)
    if missing:
        print(f"warning: {missing} of {len(rows)} target rows have no prediction; they are excluded")
    rows = [r for r in rows if row_key(r, field) in preds]
    return [float(r[args.target_key]) for r in rows], [preds[row_key(r, field)] for r in rows]


def main():
    args = parse_args_with_config(build_parser())
    if not args.data:
        raise SystemExit("--data is required")
    if args.predictions:
        pred_rows = read_jsonl(args.predictions)
    elif args.checkpoint:
        pred_rows = score_clips(args)
        if pred_rows is None:
            return
    else:
        raise SystemExit("give --predictions or --checkpoint")
    target, pred = join_targets(pred_rows, read_jsonl(args.data), args)
    if not target:
        print(f"no rows with both `{args.target_key}` and `{args.pred_key}`; nothing to evaluate")
        return

    metrics = compute_metrics(target, pred, args.top1_n, args.top1_draws, args.seed)
    if args.bootstrap > 0:
        metrics["ci95"] = bootstrap_ci(target, pred, args.bootstrap, args.top1_n, args.seed)
    n = args.top1_n
    print(f"n={metrics['n']}  pairs={metrics['pair_n']}  strict pairs={metrics['pair_n_gap0.75']}")
    for name in ("spearman", f"top1_n{n}", "pair_acc", "pearson", "kendall", "pair_acc_gap0.75", "mae"):
        if name not in metrics:
            continue
        ci = metrics.get("ci95", {}).get(name)
        extra = f"  [95% CI {ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"{name:>24s}: {metrics[name]:.4f}{extra}")
    if f"top1_n{n}_chance" in metrics:
        print(f"{'top1 chance':>24s}: {metrics[f'top1_n{n}_chance']:.4f}")
    if args.metrics_output:
        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_output).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
