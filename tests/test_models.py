"""Tests for the simulated quantizers in probedir.models (CPU)."""
import pytest

torch = pytest.importorskip("torch")

from probedir import models  # noqa: E402


def test_rtn_matches_reference_formula():
    torch.manual_seed(0)
    w = torch.randn(64, 300, dtype=torch.float32)
    q = w.clone()
    models.rtn_quantize_(q, 4)
    g = torch.nn.functional.pad(w, (0, 84)).reshape(64, -1, 128)
    s = (g.abs().amax(-1, keepdim=True) / 7).clamp_min(1e-10)
    ref = ((g / s).round().clamp(-7, 7) * s).reshape(64, -1)[:, :300]
    assert torch.equal(q, ref)
    levels = (g / s).round().clamp(-7, 7)
    assert levels.abs().max() <= 7


def test_native_fp16_variant_keeps_dtype_and_grid():
    torch.manual_seed(1)
    w = torch.randn(8, 256).half()
    q = models.rtn_quantize_native(w, 8)
    assert q.dtype == torch.float16 and q.shape == w.shape
    assert (q.float() - w.float()).abs().max() < 0.05


def test_activation_quantizer_per_token_int8():
    block = torch.nn.Linear(16, 16)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([block])

    m = Wrapper()
    x = torch.randn(2, 5, 16)
    raw = block(x).detach()
    hook = models.ActivationQuantizer(m)
    quantized = block(x).detach()
    scale = raw.abs().amax(-1, keepdim=True) / 127
    assert torch.allclose(quantized, (raw / scale).round() * scale)
    hook.remove()
    assert torch.equal(block(x).detach(), raw)
