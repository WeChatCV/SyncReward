"""Stage 1 contrastive loss and Stage 2 regression + ranking losses."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentInfoNCE(nn.Module):
    """Symmetric InfoNCE over all B*T segments of a batch.

    Row i of `audio` and `video` ([N, D], L2-normalised) is a positive pair. Every other row is a
    negative: other positions of the same clip (temporal negatives) and segments of other clips
    (semantic negatives). The temperature is learnable, initialised to `init_temperature`.
    """

    def __init__(self, init_temperature=0.07, learnable=True):
        super().__init__()
        log_t = torch.tensor(math.log(init_temperature), dtype=torch.float64)
        if learnable:
            self.log_temp = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temp", log_t)

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=1e-4, max=1.0)

    def forward(self, audio, video):
        labels = torch.arange(audio.shape[0], device=audio.device)
        logits_a2v = audio @ video.T / self.temperature
        logits_v2a = video @ audio.T / self.temperature
        loss = (F.cross_entropy(logits_a2v.float(), labels) + F.cross_entropy(logits_v2a.float(), labels)) / 2.0
        return loss, logits_a2v, logits_v2a


def regression_loss(pred, target, beta=0.1):
    return F.smooth_l1_loss(pred, target, beta=beta)


def ranking_loss(pred, target, min_gap=0.75, margin=0.25):
    """Mean over ordered in-batch pairs with |y_i - y_j| >= min_gap of softplus(margin - sign(y_i - y_j)(R_i - R_j)).

    softplus(margin - d) = -log sigmoid(d - margin). Returns 0 (with a graph) if no pair qualifies.
    """
    diff_t = target[:, None] - target[None, :]
    mask = diff_t.abs() >= min_gap
    if not mask.any():
        return pred.sum() * 0.0
    diff_p = pred[:, None] - pred[None, :]
    return F.softplus(margin - diff_t.sign()[mask] * diff_p[mask]).mean()


def stage2_loss(pred, target, ranking_weight=0.1, beta=0.1, min_gap=0.75, margin=0.25):
    reg = regression_loss(pred, target, beta)
    rank = ranking_loss(pred, target, min_gap, margin)
    return reg + ranking_weight * rank, reg, rank
