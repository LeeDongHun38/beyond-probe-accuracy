import numpy as np
import pytest

from probedir.transfer import transfer_metrics


def _data(n=200, d=16, seed=0):
    rng = np.random.default_rng(seed)
    hm = rng.normal(size=(n, d))
    hp = hm + 0.0
    hp[:, 0] += 1.0
    v = np.zeros(d)
    v[0] = 1.0
    return hp, hm, v


def test_identical_quantization_changes_nothing():
    hp, hm, v = _data()
    m = transfer_metrics(hp, hm, hp, hm, v)
    assert m["auroc_drop"] == 0
    assert m["margin_flip_n"] == 0
    assert m["relative_shrinkage"] == 0
    assert m["positive_shrinkage_rate"] == 0


def test_margin_definitions():
    hp, hm, v = _data()
    qp = hp.copy()
    qp[:, 0] -= 0.5                     # every margin shrinks from 1.0 to 0.5
    m = transfer_metrics(hp, hm, qp, hm, v)
    assert m["mean_margin_fp16"] == pytest.approx(1.0)
    assert m["mean_margin_quantized"] == pytest.approx(0.5)
    assert m["relative_shrinkage"] == pytest.approx(0.5)
    assert m["positive_shrinkage_rate"] == 1.0
    assert m["margin_flip_n"] == 0


def test_sign_zero_counts_as_flip_only_against_nonzero():
    hp, hm, v = _data(n=4)
    qp = hp.copy()
    qp[0, 0] -= 1.0                     # margin 1 -> 0: sign(1) != sign(0) -> flip
    m = transfer_metrics(hp, hm, qp, hm, v)
    assert m["margin_flip_n"] == 1
    z = hm.copy()                       # FP16 margin 0 and quantized margin 0 -> no flip
    assert transfer_metrics(z, hm, z, hm, v)["margin_flip_n"] == 0


def test_auroc_uses_independent_candidates_and_drop_sign():
    hp, hm, v = _data()
    m = transfer_metrics(hp, hm, hm, hp, v)   # quantization swaps the classes
    assert m["fp16_auroc"] > 0.5 > m["quantized_auroc"]
    assert m["auroc_drop"] == pytest.approx(m["fp16_auroc"] - m["quantized_auroc"])
