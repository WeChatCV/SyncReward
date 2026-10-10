"""Audio/video decoding, temporal segmentation and jsonl datasets."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

MIN_VIDEO_FRAMES = 2
MIN_AUDIO_SAMPLES = 4800
_RESAMPLERS = {}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_video_list(path):
    """Plain text (one path per line) or jsonl with a `video_path` field."""
    paths = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(json.loads(line)["video_path"] if line.startswith("{") else line)
    return paths


def load_av(path, sample_rate):
    """Decode all frames (uint8 [T, H, W, C]) and a mono waveform resampled to `sample_rate`."""
    import torchaudio
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path))
    meta = decoder.metadata
    fps = max(float(getattr(meta, "average_fps", 25.0) or 25.0), 1e-3)
    num_frames = int(getattr(meta, "num_frames", 0) or 0)
    if num_frames <= 0:
        raise ValueError(f"no frames in {path}")
    frames = decoder.get_frames_at(indices=np.arange(num_frames, dtype=np.int64)).data
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = np.transpose(frames, (0, 2, 3, 1))

    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        key = (int(sr), int(sample_rate))
        if key not in _RESAMPLERS:
            _RESAMPLERS[key] = torchaudio.transforms.Resample(*key)
        wav = _RESAMPLERS[key](wav)
    audio = wav[0].cpu().numpy().astype(np.float32, copy=False)
    return frames.astype(np.uint8, copy=False), fps, audio


def _take_video(video, start, length):
    seg = video[start:start + length]
    if seg.shape[0] == 0:
        return np.repeat(video[-1:], length, axis=0)
    if seg.shape[0] < length:
        seg = np.concatenate([seg, np.repeat(seg[-1:], length - seg.shape[0], axis=0)], axis=0)
    return seg


def _take_audio(audio, start, length):
    seg = audio[start:start + length]
    if seg.shape[0] < length:
        seg = np.pad(seg, (0, length - seg.shape[0]))
    return seg


def spread_segments(video, audio, fps, sr, num_segments, duration):
    """Stage 1: `num_segments` windows at uniformly spaced offsets over the whole clip.

    The audio window of each segment starts at the same timestamp as its video window.
    """
    seg_frames = max(1, int(round(duration * fps)))
    seg_samples = max(1, int(round(duration * sr)))
    available = max(video.shape[0] - seg_frames, 0)
    if num_segments == 1 or available == 0:
        offsets = np.zeros(num_segments, dtype=np.int64)
    else:
        offsets = np.rint(np.linspace(0, available, num_segments)).astype(np.int64)
    frame_to_audio = sr / fps
    out_v, out_a = [], []
    for sf in offsets:
        sf = int(sf)
        out_v.append(_take_video(video, sf, seg_frames))
        out_a.append(_take_audio(audio, int(round(sf * frame_to_audio)), seg_samples))
    return out_v, out_a


def window_segments(video, audio, fps, sr, num_segments, duration, stride, crop="center"):
    """Stage 2 / inference: `num_segments` overlapping windows from one continuous span.

    crop="center":  deterministic window centred in the clip, shared by audio and video.
    crop="random":  random start shared by audio and video (Stage 2 training augmentation).
    crop="random_independent": audio and video starts drawn independently, reproducing the
        `_cut_segments` behaviour of the self-supervised corruption stage (not used in the paper
        configuration; it can misalign audio and video by up to the clip slack).
    """
    span = duration + (num_segments - 1) * stride
    video_sec, audio_sec = len(video) / fps, len(audio) / sr
    if crop == "random_independent":
        slack_v, slack_a = max(0.0, video_sec - span), max(0.0, audio_sec - span)
        start_v = random.random() * slack_v if slack_v > 0 else 0.0
        start_a = random.random() * slack_a if slack_a > 0 else 0.0
    else:
        slack = max(0.0, min(video_sec, audio_sec) - span)
        if crop == "random":
            start_v = start_a = random.random() * slack if slack > 0 else slack / 2.0
        elif crop == "center":
            start_v = start_a = slack / 2.0
        else:
            raise ValueError(f"unknown crop mode: {crop}")
    seg_frames = max(1, int(round(duration * fps)))
    seg_samples = max(1, int(round(duration * sr)))
    out_v, out_a = [], []
    for i in range(num_segments):
        out_v.append(_take_video(video, int(round((start_v + i * stride) * fps)), seg_frames))
        out_a.append(_take_audio(audio, int(round((start_a + i * stride) * sr)), seg_samples))
    return out_v, out_a


class ClipDataset(Dataset):
    """Rows with a `video_path` and optionally a numeric target. Undecodable clips yield None.

    Relative `video_path`s are resolved against `video_root` (the released manifests are relative).
    """

    def __init__(self, rows, sample_rate, target_key=None, video_root=None):
        self.rows = rows
        self.sample_rate = int(sample_rate)
        self.target_key = target_key
        self.video_root = Path(video_root) if video_root else None
        if target_key is not None:
            self.rows = [r for r in rows if r.get(target_key) is not None]

    @classmethod
    def from_file(cls, path, sample_rate, target_key=None, video_root=None):
        if str(path).endswith(".jsonl"):
            return cls(read_jsonl(path), sample_rate, target_key, video_root)
        return cls([{"video_path": p} for p in read_video_list(path)], sample_rate, target_key, video_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["video_path"])
        if self.video_root is not None and not path.is_absolute():
            path = self.video_root / path
        if not path.is_file():
            return None
        try:
            video, fps, audio = load_av(path, self.sample_rate)
        except Exception as exc:
            print(f"skipping undecodable video {path}: {type(exc).__name__}: {exc}", flush=True)
            return None
        if video.shape[0] < MIN_VIDEO_FRAMES or audio.shape[0] < MIN_AUDIO_SAMPLES:
            return None
        item = {"index": index, "video": video, "audio": audio, "fps": float(fps)}
        if self.target_key is not None:
            item["target"] = float(row[self.target_key])
        return item


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {key: [b[key] for b in batch] for key in ("index", "video", "audio", "fps")}
    if "target" in batch[0]:
        out["target"] = torch.tensor([b["target"] for b in batch], dtype=torch.float32)
    return out


class ShardSampler(Sampler):
    """Shard a dataset across ranks without padding or duplicating samples."""

    def __init__(self, dataset, rank=0, world=1):
        self.n, self.rank, self.world = len(dataset), rank, world

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world))

    def __len__(self):
        return max(0, (self.n - self.rank + self.world - 1) // self.world)
