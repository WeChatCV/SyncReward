#!/usr/bin/env python3
"""Stage 1: semantic-temporal mixed contrastive learning of PE-AV segment encoders on real clips.

Each clip is cut into T segments; the audio and video segment at the same position form a positive
pair, all other segments in the batch are negatives. The full-batch loss is computed exactly with a
GradCache-style two-pass scheme so that only `encode_batch_size` segments are in memory at a time.
"""

import argparse
import itertools
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from syncreward.data import ClipDataset, collate, spread_segments
from syncreward.losses import SegmentInfoNCE
from syncreward.model import DEFAULT_ENCODER, PeAVEncoder, load_processor, processor_sample_rate
from syncreward.utils import (any_rank_empty, cleanup_distributed, cosine_with_warmup, get_world_size, is_main,
                              parse_args_with_config, print0, setup_distributed)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_list", help="Real videos: .txt (one path per line) or .jsonl with `video_path`.")
    p.add_argument("--output_dir", default="outputs/stage1")
    p.add_argument("--encoder_path", default=DEFAULT_ENCODER, help="HF id or local dir of PE-AV weights.")
    p.add_argument("--num_segments", type=int, default=18)
    p.add_argument("--segment_duration", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=2, help="Clips per GPU.")
    p.add_argument("--encode_batch_size", type=int, default=6, help="Segments per encoder forward.")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no limit.")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--init_temperature", type=float, default=0.07)
    p.add_argument("--learnable_temperature", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--log_every_steps", type=int, default=10)
    p.add_argument("--init_from", default=None, help="Warm-start encoder + temperature; fresh optimizer/schedule.")
    p.add_argument("--resume", default=None, help="Resume model, optimizer, schedule and step.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


class OffsetSampler(DistributedSampler):
    """DistributedSampler that skips the first `offset` indices (for resuming mid-epoch)."""

    offset = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.offset, None)


def gradcache_backward(model, loss_fn, inputs, chunk, autocast):
    """Accumulate exact gradients of the full-batch contrastive loss, encoding `chunk` segments at a time."""
    n = inputs["input_values"].shape[0]
    # Pass 1: embed all segments without a graph, then backprop the loss to the embeddings only.
    with torch.no_grad(), torch.autocast(**autocast):
        embeds = [model.encode_range(inputs, s, min(s + chunk, n)) for s in range(0, n, chunk)]
    audio_cache = torch.cat([e[0] for e in embeds]).detach().requires_grad_(True)
    video_cache = torch.cat([e[1] for e in embeds]).detach().requires_grad_(True)
    del embeds
    with torch.autocast(**autocast):
        loss, logits_a2v, _ = loss_fn(audio_cache, video_cache)
    loss.backward()
    # Pass 2: re-encode chunk by chunk and inject the cached embedding gradients.
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        with torch.autocast(**autocast):
            a, v = model.encode_range(inputs, s, e)
            surrogate = (a * audio_cache.grad[s:e]).sum() + (v * video_cache.grad[s:e]).sum()
        surrogate.backward()
    return loss.detach(), logits_a2v.detach()


def all_reduce_grads(params):
    if not dist.is_initialized():
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(get_world_size())


def save(path, model, loss_fn, optimizer, scheduler, epoch, step):
    if is_main():
        os.makedirs(path, exist_ok=True)
        torch.save({"epoch": epoch, "global_step": step, "model_state_dict": model.state_dict(),
                    "loss_fn_state_dict": loss_fn.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()}, os.path.join(path, "training_state.pt"))
        print0(f"saved {path}")
    if dist.is_initialized():
        dist.barrier()


def main():
    args = parse_args_with_config(build_parser())
    if not args.train_list:
        raise SystemExit("--train_list is required")
    if args.init_from and args.resume:
        raise SystemExit("use only one of --init_from / --resume")
    rank, world, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(args.output_dir, "training.log"), level=logging.INFO,
                            format="%(asctime)s - %(message)s", force=True)
        logging.info("args: %s", vars(args))

    processor = load_processor(args.encoder_path)
    sample_rate = processor_sample_rate(processor)
    model = PeAVEncoder.from_pretrained(args.encoder_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device).train()
    loss_fn = SegmentInfoNCE(args.init_temperature, args.learnable_temperature).to(device)

    state, start_epoch, global_step = None, 0, 0
    ckpt = args.resume or args.init_from
    if ckpt:
        ckpt = ckpt if os.path.isfile(ckpt) else os.path.join(ckpt, "training_state.pt")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        if "loss_fn_state_dict" in state:
            loss_fn.load_state_dict(state["loss_fn_state_dict"])
        if args.resume:
            start_epoch, global_step = int(state["epoch"]), int(state["global_step"])
        else:
            state = None
    if dist.is_initialized():
        for t in itertools.chain(model.parameters(), loss_fn.parameters()):
            dist.broadcast(t.data, src=0)

    params = [p for p in itertools.chain(model.parameters(), loss_fn.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.0)

    dataset = ClipDataset.from_file(args.train_list, sample_rate)
    sampler = OffsetSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
                        prefetch_factor=1 if args.num_workers > 0 else None)
    steps_per_epoch = len(loader)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.num_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup(args.warmup_steps, total_steps))
    if state is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
    del state
    print0(f"clips={len(dataset)} world={world} steps/epoch={steps_per_epoch} total_steps={total_steps}")

    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    epoch = start_epoch
    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        sampler.offset = max(0, global_step - epoch * steps_per_epoch) * args.batch_size if epoch == start_epoch else 0
        for batch in tqdm(loader, disable=not is_main(), desc=f"epoch {epoch + 1}/{args.num_epochs}"):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            if any_rank_empty(batch, device):
                continue
            clips_v, clips_a = [], []
            for video, audio, fps in zip(batch["video"], batch["audio"], batch["fps"]):
                v, a = spread_segments(video, audio, fps, sample_rate, args.num_segments, args.segment_duration)
                clips_v.extend(v)
                clips_a.extend(a)
            inputs = processor(videos=clips_v, audio=clips_a, return_tensors="pt", padding=True,
                               sampling_rate=sample_rate)
            inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}
            n = inputs["input_values"].shape[0]
            loss, logits_a2v = gradcache_backward(model, loss_fn, inputs, args.encode_batch_size, autocast)
            all_reduce_grads(loss_fn.parameters())
            all_reduce_grads([p for p in model.parameters() if p.requires_grad])
            grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                         if args.max_grad_norm > 0 else torch.zeros(()))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main() and global_step % args.log_every_steps == 0:
                acc = (logits_a2v.argmax(dim=1) == torch.arange(n, device=device)).float().mean().item()
                msg = (f"epoch {epoch} step {global_step} loss={loss.item():.4f} acc_a2v={acc:.3f} "
                       f"temp={loss_fn.temperature.item():.4f} gnorm={float(grad_norm):.3f} "
                       f"lr={scheduler.get_last_lr()[0]:.2e}")
                print(msg, flush=True)
                logging.info(msg)
            global_step += 1
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save(os.path.join(args.output_dir, f"checkpoint-step-{global_step}"),
                     model, loss_fn, optimizer, scheduler, epoch, global_step)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    save(os.path.join(args.output_dir, f"checkpoint-step-{global_step}"),
         model, loss_fn, optimizer, scheduler, epoch, global_step)
    cleanup_distributed()


if __name__ == "__main__":
    main()
