"""CPU smoke test with a stub encoder (no pretrained weights, no video decoding).

Run from the repository root:  python -m tests.test_smoke
"""

import types

import numpy as np
import torch
import torch.nn as nn

from syncreward.data import spread_segments, window_segments
from syncreward.losses import SegmentInfoNCE, stage2_loss
from syncreward.metrics import bootstrap_ci, compute_metrics, kendall_tau_b
from syncreward.model import PeAVEncoder, SyncRewardModel, load_config, load_state_dict, save_pretrained
from train_stage1 import gradcache_backward


class StubTower(nn.Module):
    def __init__(self, in_dim, hidden, tokens, key):
        super().__init__()
        self.proj, self.tokens, self.key = nn.Linear(in_dim, hidden * tokens), tokens, key

    def forward(self, **kw):
        x = kw[self.key].flatten(1)
        return types.SimpleNamespace(last_hidden_state=self.proj(x).view(x.shape[0], self.tokens, -1))


class StubHead(nn.Module):
    def __init__(self, hidden, out):
        super().__init__()
        self.layer_norm, self.proj = nn.LayerNorm(hidden), nn.Linear(hidden, out, bias=False)

    def forward(self, x):
        return self.proj(self.layer_norm(x))


def stub_encoder(audio_len=64, video_numel=3 * 2 * 4 * 4, hidden=32, dim=16):
    return PeAVEncoder(StubTower(audio_len, hidden, 5, "input_values"),
                       StubTower(video_numel, hidden, 3, "pixel_values_videos"),
                       StubHead(hidden, dim), StubHead(hidden, dim))


def fake_inputs(n):
    return {"input_values": torch.randn(n, 64), "pixel_values_videos": torch.randn(n, 2, 3, 4, 4)}


def test_segments():
    video = np.random.randint(0, 255, (250, 8, 8, 3), dtype=np.uint8)
    audio = np.random.randn(48000 * 10).astype(np.float32)
    v, a = spread_segments(video, audio, 25.0, 48000, 18, 0.5)
    assert len(v) == len(a) == 18 and v[0].shape == (12, 8, 8, 3) and a[0].shape == (24000,)
    v1, a1 = window_segments(video, audio, 25.0, 48000, 18, 0.5, 0.25, "center")
    v2, a2 = window_segments(video, audio, 25.0, 48000, 18, 0.5, 0.25, "center")
    assert all((x == y).all() for x, y in zip(v1 + a1, v2 + a2)), "center crop must be deterministic"
    for crop in ("random", "random_independent"):
        v3, a3 = window_segments(video, audio, 25.0, 48000, 18, 0.5, 0.25, crop)
        assert len(v3) == 18 and v3[0].shape == (12, 8, 8, 3) and a3[0].shape == (24000,)


def test_stage1():
    torch.manual_seed(0)
    model, loss_fn = stub_encoder(), SegmentInfoNCE()
    inputs = fake_inputs(2 * 18)
    autocast = dict(device_type="cpu", enabled=False)
    loss, _ = gradcache_backward(model, loss_fn, inputs, 6, autocast)
    grads = [p.grad.clone() for p in model.parameters()]
    temp_grad = loss_fn.log_temp.grad.clone()
    model.zero_grad(), loss_fn.zero_grad()
    a, v = model(inputs, num_segments=18, chunk_size=36)
    ref, _, _ = loss_fn(a.reshape(36, -1), v.reshape(36, -1))
    ref.backward()
    assert torch.allclose(loss, ref.detach(), atol=1e-6)
    assert all(torch.allclose(g, p.grad, atol=1e-5) for g, p in zip(grads, model.parameters())), "GradCache mismatch"
    assert torch.allclose(temp_grad, loss_fn.log_temp.grad, atol=1e-6)
    print(f"stage1 ok: loss={loss.item():.4f} (GradCache gradients match full-graph gradients)")


def test_stage2():
    torch.manual_seed(0)
    model = SyncRewardModel(stub_encoder(), num_segments=18, sync_dim=32, num_layers=2, num_heads=4).train()
    assert not model.encoder.training
    pred = model(fake_inputs(8 * 18), encode_batch_size=18)
    assert pred.shape == (8,) and (pred >= 0).all() and (pred <= 2).all()
    target = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0])
    loss, reg, rank = stage2_loss(pred, target)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters()), "encoder must stay frozen"
    assert all(p.grad is not None for n, p in model.named_parameters() if not n.startswith("encoder."))
    _, _, zero = stage2_loss(pred.detach(), torch.ones(8))
    assert zero.item() == 0.0
    print(f"stage2 ok: loss={loss.item():.4f} reg={reg.item():.4f} rank={rank.item():.4f}")


def test_metrics():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 7, 300) / 3.0
    pred = target + rng.normal(0, 0.5, 300)
    m = compute_metrics(target, pred, draws=50)
    ci = bootstrap_ci(target, pred, num_samples=50)
    assert 0 < m["spearman"] <= 1 and 0.5 < m["pair_acc"] <= 1 and ci["spearman"][0] <= m["spearman"]
    assert m["top1_n4"] > m["top1_n4_chance"]
    perfect = compute_metrics(target, target, draws=20)
    assert perfect["pair_acc"] == 1.0 and perfect["top1_n4"] == 1.0
    try:
        from scipy.stats import kendalltau
        assert abs(kendall_tau_b(pred, target) - kendalltau(pred, target).statistic) < 1e-10
    except ImportError:
        pass
    print("metrics ok:", {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


def test_checkpoint_roundtrip():
    import tempfile

    torch.manual_seed(0)
    model = SyncRewardModel(stub_encoder(), num_segments=18, sync_dim=32, num_layers=2, num_heads=4).eval()
    with tempfile.TemporaryDirectory() as tmp:
        save_pretrained(model.state_dict(), tmp, {"num_segments": 18, "sync_dim": 32, "num_layers": 2, "num_heads": 4})
        assert load_config(tmp)["sync_dim"] == 32
        clone = SyncRewardModel(stub_encoder(), num_segments=18, sync_dim=32, num_layers=2, num_heads=4).eval()
        clone.load_state_dict(load_state_dict(tmp), strict=True)
    inputs = fake_inputs(2 * 18)
    assert torch.equal(model(inputs), clone(inputs))
    print("checkpoint ok: safetensors round-trip is exact")


if __name__ == "__main__":
    test_segments()
    test_stage1()
    test_stage2()
    test_metrics()
    test_checkpoint_roundtrip()
    print("all smoke tests passed")
