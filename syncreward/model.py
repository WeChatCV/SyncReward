"""PE-AV segment encoder, cross-modal reward transformer, checkpoint I/O and batched inference."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import window_segments
from .utils import all_gather_list

DEFAULT_ENCODER = "facebook/pe-av-large"


def load_processor(encoder_path=DEFAULT_ENCODER):
    from transformers import PeAudioVideoProcessor

    return PeAudioVideoProcessor.from_pretrained(encoder_path)


def processor_sample_rate(processor):
    return int(getattr(processor.feature_extractor, "sampling_rate", 48000))


class PeAVEncoder(nn.Module):
    """Audio and video towers of PE-AV plus their contrastive heads.

    A segment embedding is L2-normalised head(mean over tokens of the last hidden state).
    Attribute names match the state-dict layout of the Stage-1 checkpoints.
    """

    TOWERS = ("audio_encoder", "video_encoder", "audio_head", "video_head")

    def __init__(self, audio_encoder, video_encoder, audio_head, video_head):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.video_encoder = video_encoder
        self.audio_head = audio_head
        self.video_head = video_head

    @classmethod
    def from_pretrained(cls, encoder_path=DEFAULT_ENCODER):
        """Load from the HF PE-AV checkpoint, or from a local dir whose `model.safetensors` stores only the
        towers + heads in this module's flat layout (`audio_encoder.*`, ...), which transformers cannot map."""
        from transformers import PeAudioVideoModel

        full = PeAudioVideoModel.from_pretrained(encoder_path)
        encoder = cls(full.audio_model.audio_encoder, full.video_model.video_encoder,
                      full.audio_model.audio_head, full.video_model.video_head)
        flat = Path(encoder_path) / "model.safetensors"
        if flat.is_file():
            from safetensors import safe_open
            from safetensors.torch import load_file

            with safe_open(str(flat), "pt") as f:
                is_flat = all(k.split(".", 1)[0] in cls.TOWERS for k in f.keys())
            if is_flat:
                encoder.load_state_dict(load_file(str(flat)), strict=True)
        return encoder

    @property
    def embed_dim(self):
        return self.audio_head.proj.out_features

    def gradient_checkpointing_enable(self):
        self.audio_encoder.gradient_checkpointing_enable()
        self.video_encoder.gradient_checkpointing_enable()

    def encode_range(self, inputs, start, end):
        def sl(key):
            value = inputs.get(key)
            return value[start:end] if value is not None else None

        a = self.audio_encoder(input_values=sl("input_values"), padding_mask=sl("padding_mask"))
        v = self.video_encoder(pixel_values_videos=sl("pixel_values_videos"),
                               padding_mask_videos=sl("padding_mask_videos"))
        a = F.normalize(self.audio_head(a.last_hidden_state.mean(dim=1)), dim=-1)
        v = F.normalize(self.video_head(v.last_hidden_state.mean(dim=1)), dim=-1)
        return a, v

    def forward(self, inputs, num_segments, chunk_size):
        """Encode B*T flat segments in chunks; returns audio, video embeddings [B, T, D]."""
        total = inputs["input_values"].shape[0]
        chunks = [self.encode_range(inputs, s, min(s + chunk_size, total)) for s in range(0, total, chunk_size)]
        a = torch.cat([c[0] for c in chunks], dim=0)
        v = torch.cat([c[1] for c in chunks], dim=0)
        return a.reshape(total // num_segments, num_segments, -1), v.reshape(total // num_segments, num_segments, -1)


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SyncRewardModel(nn.Module):
    """Frozen PE-AV segment encoder -> [REW; V_1..V_T; MOD; A_1..A_T] -> Transformer -> R in [0, max_reward]."""

    def __init__(self, encoder, num_segments=18, sync_dim=768, num_layers=3, num_heads=8,
                 dropout=0.1, max_reward=2.0):
        super().__init__()
        self.encoder = encoder
        self.num_segments = int(num_segments)
        self.max_reward = float(max_reward)
        dim = encoder.embed_dim
        self.audio_proj = nn.Linear(dim, sync_dim)
        self.video_proj = nn.Linear(dim, sync_dim)
        self.audio_ln = nn.LayerNorm(sync_dim)
        self.video_ln = nn.LayerNorm(sync_dim)
        self.reward_token = nn.Parameter(torch.randn(1, 1, sync_dim) * 0.02)
        self.mod_token = nn.Parameter(torch.randn(1, 1, sync_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 2 + 2 * self.num_segments, sync_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=sync_dim, nhead=num_heads, dim_feedforward=4 * sync_dim,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(sync_dim)
        self.reward_head = nn.Sequential(nn.Linear(sync_dim, sync_dim), nn.GELU(), nn.Linear(sync_dim, 1))
        for module in (self.audio_proj, self.video_proj, self.audio_ln, self.video_ln,
                       self.interaction, self.final_ln, self.reward_head):
            module.apply(_init_weights)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def score_embeddings(self, audio_embeds, video_embeds):
        a = self.audio_ln(self.audio_proj(audio_embeds))
        v = self.video_ln(self.video_proj(video_embeds))
        b = a.shape[0]
        x = torch.cat([self.reward_token.expand(b, -1, -1), v, self.mod_token.expand(b, -1, -1), a], dim=1)
        x = self.final_ln(self.interaction(x + self.pos_embed[:, : x.shape[1], :]))
        return self.max_reward * torch.sigmoid(self.reward_head(x[:, 0, :]).squeeze(-1))

    def forward(self, inputs, encode_batch_size=18):
        with torch.no_grad():
            audio_embeds, video_embeds = self.encoder(inputs, self.num_segments, encode_batch_size)
        return self.score_embeddings(audio_embeds, video_embeds)


WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
ARCH_KEYS = ("num_segments", "sync_dim", "num_layers", "num_heads", "dropout", "max_reward")


def load_state_dict(path):
    """Weights from a released dir (`model.safetensors`), a `.safetensors` file, or a training checkpoint."""
    path = Path(path)
    if path.is_dir():
        path = path / WEIGHTS_NAME if (path / WEIGHTS_NAME).is_file() else path / "training_state.pt"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state.get("model_state_dict", state)


def load_config(path):
    """`config.json` next to released weights, or {} for training checkpoints."""
    path = Path(path)
    cfg = (path if path.is_dir() else path.parent) / CONFIG_NAME
    return json.loads(cfg.read_text()) if cfg.is_file() else {}


def remap_legacy_keys(state_dict):
    """Map the development-code layout (`encoder.base_model.*`) onto this module layout."""
    return {k.replace("encoder.base_model.", "encoder.", 1): v for k, v in state_dict.items()}


def build_reward_model(encoder_path=DEFAULT_ENCODER, checkpoint=None, stage1_checkpoint=None, **arch):
    """Build the reward model; optionally load Stage-1 encoder weights or a full Stage-2 checkpoint.

    Architecture values stored in a released checkpoint's `config.json` override `arch`.
    """
    encoder = PeAVEncoder.from_pretrained(encoder_path)
    if stage1_checkpoint:
        missing, _ = encoder.load_state_dict(load_state_dict(stage1_checkpoint), strict=False)
        if len(missing) == len(encoder.state_dict()):
            raise RuntimeError(f"no encoder tensors loaded from {stage1_checkpoint}")
    if checkpoint:
        arch.update({k: v for k, v in load_config(checkpoint).items() if k in ARCH_KEYS})
    model = SyncRewardModel(encoder, **arch)
    if checkpoint:
        model.load_state_dict(remap_legacy_keys(load_state_dict(checkpoint)), strict=True)
    return model


def save_pretrained(state_dict, out_dir, config):
    """Write `model.safetensors` + `config.json` (the released checkpoint format)."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out_dir / WEIGHTS_NAME),
              metadata={"format": "pt"})
    (out_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")


def prepare_inputs(processor, batch, args, crop, device):
    """Cut T segments per clip and run the PE-AV processor. Returns model inputs on `device`."""
    sr = processor_sample_rate(processor)
    clips_v, clips_a = [], []
    for video, audio, fps in zip(batch["video"], batch["audio"], batch["fps"]):
        v, a = window_segments(video, audio, fps, sr, args.num_segments, args.segment_duration,
                               args.segment_stride, crop)
        clips_v.extend(v)
        clips_a.extend(a)
    inputs = processor(videos=clips_v, audio=clips_a, return_tensors="pt", padding=True, sampling_rate=sr)
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}


@torch.no_grad()
def predict(model, processor, loader, device, args, crop="center"):
    """Score every clip in `loader` (sharded across ranks). Returns {dataset index: reward} on all ranks."""
    was_training = model.training
    model.eval()
    local = []
    for batch in loader:
        if batch is None:
            continue
        inputs = prepare_inputs(processor, batch, args, crop, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reward = model(inputs, encode_batch_size=args.encode_batch_size)
        local.extend(zip(batch["index"], reward.float().cpu().tolist()))
    model.train(was_training)
    return dict(all_gather_list(local))
