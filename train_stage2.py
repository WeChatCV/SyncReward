#!/usr/bin/env python3
"""Stage 2: human alignment of the sync reward model.

The PE-AV encoder is initialised from Stage 1 and frozen; the projections, learnable tokens,
cross-modal Transformer and reward head are trained with SmoothL1(R, y) + lambda * in-batch ranking loss.
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from syncreward.data import ClipDataset, ShardSampler, collate
from syncreward.losses import stage2_loss
from syncreward.metrics import compute_metrics
from syncreward.model import (DEFAULT_ENCODER, build_reward_model, load_processor, predict, prepare_inputs,
                              processor_sample_rate)
from syncreward.utils import (any_rank_empty, cleanup_distributed, cosine_with_warmup, is_main,
                              parse_args_with_config, print0, setup_distributed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_jsonl", help="Rows with `video_path` and `target` (see README).")
    p.add_argument("--val_jsonl", default=None, help="Optional; selects the `best` checkpoint by Spearman.")
    p.add_argument("--target_key", default="target")
    p.add_argument("--video_root", default=None, help="Prefix for relative `video_path`s.")
    p.add_argument("--stage1_checkpoint", help="Stage-1 checkpoint dir (or training_state.pt).")
    p.add_argument("--encoder_path", default=DEFAULT_ENCODER, help="HF id or local dir of PE-AV weights.")
    p.add_argument("--output_dir", default="outputs/stage2")
    p.add_argument("--num_segments", type=int, default=18)
    p.add_argument("--segment_duration", type=float, default=0.5)
    p.add_argument("--segment_stride", type=float, default=0.25)
    p.add_argument("--train_crop", default="random", choices=["random", "center", "random_independent"],
                   help="Window crop for training clips; `random` shares one random start for audio and video.")
    p.add_argument("--sync_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8, help="Clips per GPU.")
    p.add_argument("--encode_batch_size", type=int, default=18)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no limit.")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ranking_weight", type=float, default=0.1)
    p.add_argument("--ranking_margin", type=float, default=0.25)
    p.add_argument("--ranking_min_gap", type=float, default=0.75)
    p.add_argument("--smooth_l1_beta", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every_steps", type=int, default=200)
    p.add_argument("--log_every_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260721)
    return p


def save(model, optimizer, scheduler, path, epoch, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    core = model.module if isinstance(model, DDP) else model
    torch.save({"epoch": epoch, "global_step": step, "model_state_dict": core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict()},
               path / "training_state.pt")


def evaluate(model, processor, loader, device, args, step):
    core = model.module if isinstance(model, DDP) else model
    preds = predict(core, processor, loader, device, args, crop="center")
    if not is_main() or not preds:
        return None
    rows = loader.dataset.rows
    idx = sorted(preds)
    metrics = compute_metrics([rows[i][args.target_key] for i in idx], [preds[i] for i in idx], draws=0)
    print0(f"val @ step {step}: {metrics}")
    logging.info("val @ step %d: %s", step, metrics)
    return metrics


def main():
    args = parse_args_with_config(build_parser())
    if not args.train_jsonl or not args.stage1_checkpoint:
        raise SystemExit("--train_jsonl and --stage1_checkpoint are required")
    rank, world, device = setup_distributed()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.output_dir)
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=out / "training.log", level=logging.INFO,
                            format="%(asctime)s - %(message)s", force=True)
        logging.info("args: %s", vars(args))

    processor = load_processor(args.encoder_path)
    sample_rate = processor_sample_rate(processor)
    model = build_reward_model(args.encoder_path, stage1_checkpoint=args.stage1_checkpoint,
                               num_segments=args.num_segments, sync_dim=args.sync_dim, num_layers=args.num_layers,
                               num_heads=args.num_heads, dropout=args.dropout).to(device)
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, gradient_as_bucket_view=True)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate,
                     pin_memory=True, persistent_workers=args.num_workers > 0,
                     prefetch_factor=1 if args.num_workers > 0 else None)
    train_set = ClipDataset.from_file(args.train_jsonl, sample_rate, args.target_key, args.video_root)
    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, drop_last=True, **loader_kw)
    val_loader = None
    if args.val_jsonl:
        val_set = ClipDataset.from_file(args.val_jsonl, sample_rate, args.target_key, args.video_root)
        val_loader = DataLoader(val_set, sampler=ShardSampler(val_set, rank, world), **loader_kw)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.max_steps or len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup(args.warmup_steps, total_steps))
    print0(f"world={world} train={len(train_set)} steps/epoch={len(train_loader)} "
           f"trainable={sum(p.numel() for p in params):,}")

    step, best = 0, -1e9
    if val_loader is not None:
        evaluate(model, processor, val_loader, device, args, step)
    epoch = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for batch in tqdm(train_loader, disable=not is_main(), desc=f"epoch {epoch + 1}/{args.num_epochs}"):
            if any_rank_empty(batch, device):
                continue
            inputs = prepare_inputs(processor, batch, args, args.train_crop, device)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(inputs, encode_batch_size=args.encode_batch_size).float()
                loss, reg, rank_loss = stage2_loss(pred, target, args.ranking_weight, args.smooth_l1_beta,
                                                   args.ranking_min_gap, args.ranking_margin)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every_steps == 0:
                stats = torch.tensor([loss.detach(), reg.detach(), rank_loss.detach(), (pred - target).abs().mean()],
                                     device=device, dtype=torch.float64)
                if dist.is_initialized():
                    dist.all_reduce(stats)
                    stats /= world
                if is_main():
                    msg = (f"step {step} loss={stats[0]:.4f} reg={stats[1]:.4f} rank={stats[2]:.4f} "
                           f"mae={stats[3]:.4f} gnorm={float(grad_norm):.3f} lr={scheduler.get_last_lr()[0]:.2e}")
                    logging.info(msg)
            step += 1
            if args.save_every_steps > 0 and step % args.save_every_steps == 0 and is_main():
                save(model, optimizer, scheduler, out / f"checkpoint-step-{step}", epoch, step)
            if args.max_steps and step >= args.max_steps:
                break
        if val_loader is not None:
            metrics = evaluate(model, processor, val_loader, device, args, step)
            if is_main() and metrics and metrics["spearman"] > best:
                best = metrics["spearman"]
                save(model, optimizer, scheduler, out / "best", epoch, step)
        if args.max_steps and step >= args.max_steps:
            break
    if is_main():
        save(model, optimizer, scheduler, out / f"checkpoint-step-{step}", epoch, step)
    cleanup_distributed()


if __name__ == "__main__":
    main()
