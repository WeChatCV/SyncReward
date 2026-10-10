"""Config parsing and torch.distributed helpers."""

import argparse
import os

import torch
import torch.distributed as dist
import yaml


def parse_args_with_config(parser: argparse.ArgumentParser, argv=None):
    """Parse CLI args; values from `--config file.yaml` become defaults that the CLI overrides."""
    parser.add_argument("--config", default=None, help="YAML file with default values for any argument.")
    known, _ = parser.parse_known_args(argv)
    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)


def setup_distributed(backend=None):
    """Initialise torch.distributed when launched by torchrun with WORLD_SIZE > 1."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend or ("nccl" if torch.cuda.is_available() else "gloo"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return get_rank(), get_world_size(), device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    return get_rank() == 0


def print0(*args):
    if is_main():
        print(*args, flush=True)


def any_rank_empty(batch, device):
    """True if the batch is empty on any rank (keeps collectives aligned)."""
    flag = torch.tensor([1 if batch is None else 0], device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def all_gather_list(local):
    if not dist.is_initialized():
        return list(local)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(local))
    return [item for shard in gathered for item in shard]


def cosine_with_warmup(warmup_steps, total_steps):
    import math

    def fn(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return fn
