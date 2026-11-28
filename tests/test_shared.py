"""Tests for the shared modules: config, io, stats."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from probedir import config, io, stats


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(0.5, 1, 300), 1)
    neg = np.round(rng.normal(0.0, 1, 250), 1)
    y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
    assert stats.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]), abs=1e-12)


def test_auroc_is_not_symmetrized():
    assert stats.auroc([0.0, 0.1], [1.0, 2.0]) == 0.0


def test_empirical_p_plus_one():
    exceed, p = stats.empirical_p(0.5, np.array([0.1, 0.6, 0.5, 0.2]))
    assert exceed == 2 and p == pytest.approx(3 / 5)
    assert stats.empirical_p(9.0, np.zeros(127))[1] == pytest.approx(1 / 128)


def test_bh_matches_scipy():
    from scipy.stats import false_discovery_control
    p = np.random.default_rng(1).uniform(size=375) ** 3
    np.testing.assert_allclose(stats.bh_qvalues(p), false_discovery_control(p, method="bh"), atol=1e-12)


def test_cluster_bootstrap_ci_brackets_mean():
    x = np.arange(100, dtype=float)
    lo, hi = stats.cluster_bootstrap_ci(lambda idx: x[idx].mean(), np.arange(100) // 2, n_boot=200)
    assert lo < x.mean() < hi


def test_config_cells_and_layers():
    cells = config.all_cells()
    assert len(cells) == 15
    assert config.Cell("llama-3.1-8b", "triviaqa").layer == 16
    assert config.Cell("ministral-3-14b", "triviaqa").layer == 36
    assert config.pairs_path("triviaqa", "train").is_file()


def test_hidden_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    hp = rng.normal(size=(2, 4, 3)).astype(np.float16)
    hm = rng.normal(size=(2, 4, 3)).astype(np.float16)
    c = io.HiddenCache(hp, hm, ["a", "b", "c", "d"], ["train", "test", "train", "test"], [5, 7], {"model": "x"})
    path = tmp_path / "c.safetensors"
    c.save(path)
    d = io.HiddenCache.load(path)
    assert d.layers == [5, 7] and d.meta == {"model": "x"}
    p, m = d.layer(7, split="test")
    np.testing.assert_array_equal(p, hp[1, [1, 3]].astype(np.float64))
    assert m.shape == (2, 3)
