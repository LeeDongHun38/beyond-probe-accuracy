"""Unit tests for probedir.probes on small synthetic data (CPU, < 10 s)."""
import numpy as np
import pytest

from probedir import probes
from probedir.stats import cosine


def synthetic_pairs(n=400, d=32, shift=1.0, seed=0):
    """Pairs whose difference has a fixed mean along axis 0 plus isotropic noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    signal = np.zeros(d)
    signal[0] = shift
    h_plus = base + signal / 2 + 0.5 * rng.normal(size=(n, d))
    h_minus = base - signal / 2 + 0.5 * rng.normal(size=(n, d))
    return h_plus.astype(np.float32), h_minus.astype(np.float32)


def test_md_direction_is_unit_and_recovers_signal():
    hp, hm = synthetic_pairs()
    v = probes.md_direction(hp, hm)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v[0] > 0.9


def test_heldout_auroc_orientation_and_range():
    hp, hm = synthetic_pairs()
    v = probes.md_direction(hp[:200], hm[:200])
    a = probes.heldout_auroc(v, hp[200:], hm[200:])
    assert 0.5 < a <= 1.0
    assert np.isclose(probes.heldout_auroc(-v, hp[200:], hm[200:]), 1 - a)


def test_lr_back_transform_matches_manual_division():
    hp, hm = synthetic_pairs(d=8)
    hp[:, 3] *= 50.0
    hm[:, 3] *= 50.0
    w, info = probes.fit_lr(hp, hm, C=1.0, standardize=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x = np.concatenate((hp, hm)).astype(np.float32)
    y = np.r_[np.ones(len(hp)), np.zeros(len(hm))]
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=1.0, **probes.LR_SETTINGS).fit(sc.transform(x), y)
    assert np.allclose(w, clf.coef_[0] / sc.scale_)
    assert info["converged"]


def test_lr_direction_orientation():
    hp, hm = synthetic_pairs()
    md = probes.md_direction(hp, hm)
    v = probes.lr_direction(hm, hp, C=1e-2, orient=md)   # labels swapped, then re-oriented
    assert np.dot(v, md) > 0


def test_split_halves_partition_training_rows():
    for a, b in probes.split_halves(11, seeds=(42, 7)):
        assert len(a) == 5 and len(b) == 6
        assert sorted(np.concatenate([a, b]).tolist()) == list(range(11))


def test_split_half_stability_md_high_for_clear_signal():
    hp, hm = synthetic_pairs(shift=3.0)
    s = probes.split_half_stability(hp, hm)
    assert len(s["cosines"]) == 5 and s["mean"] > 0.9


def test_participation_ratio_bounds():
    assert probes.participation_ratio(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
    assert probes.participation_ratio(np.ones(16)) == pytest.approx(16.0)
    assert probes.participation_ratio(3 * np.ones(16)) == pytest.approx(16.0)


class FakeCache:
    """Signal present only in layer 2."""

    def __init__(self):
        rng = np.random.default_rng(1)
        n, d = 300, 16
        self.layers = [0, 1, 2, 3]
        self.split = np.array(["train"] * 150 + ["test"] * 150)
        self.data = {}
        for layer in self.layers:
            hp, hm = rng.normal(size=(n, d)), rng.normal(size=(n, d))
            if layer == 2:
                hp[:, 0] += 2.0
            self.data[layer] = (hp, hm)

    def layer(self, layer, split=None):
        hp, hm = self.data[layer]
        if split is None:
            return hp, hm
        m = self.split == split
        return hp[m], hm[m]


def test_select_layer_uses_training_auroc():
    assert probes.select_layer(FakeCache()) == 2


def test_select_layer_tie_breaks_to_lower_layer():
    rows = [{"layer": 5, "train_auroc": 0.9}, {"layer": 3, "train_auroc": 0.9}, {"layer": 4, "train_auroc": 0.8}]
    assert probes.select_layer_from_profile(rows) == 3


def test_estimator_sweep_reports_alignment():
    hp, hm = synthetic_pairs(n=600, d=16, shift=2.0)
    res = probes.estimator_sweep(hp[:300], hm[:300], hp[300:], hm[300:], ("D0", "D1", "D3"))
    assert res["D0"]["cos_with_md"] == pytest.approx(1.0)
    assert res["D1"]["cos_with_md"] > 0.99          # strong regularization approaches MD
    for r in res.values():
        assert 0.5 < r["heldout_auroc"] <= 1.0
        assert len(r["split_half"]["cosines"]) == 5


def test_length_control_median_split():
    pos = np.array([3.0, 2.0, 1.0, 0.0])
    neg = np.array([-1.0, -2.0, 2.5, 0.5])
    res = probes.length_control(pos, neg, [10, 10, 15, 20], [10, 11, 10, 10])
    assert (res["n_small"], res["n_large"]) == (2, 2)          # |diff| = 0,1 | 5,10; median 3
    assert res["auroc_small"] == pytest.approx(1.0)
    assert res["g_len"] == pytest.approx(abs(res["auroc_small"] - res["auroc_large"]))


def test_template_control_range_and_assigned_groups():
    hp, hm = synthetic_pairs(n=90, d=8, shift=2.0)
    v = probes.md_direction(hp, hm)
    res = probes.template_length_control_assigned(v, hp, hm, [0, 1, 2] * 30, [5] * 90, [5, 8] * 45)
    assert len(res["template_aurocs"]) == 3
    assert res["g_temp"] == pytest.approx(max(res["template_aurocs"]) - min(res["template_aurocs"]))
    rendered = probes.template_length_control_rendered(v, [(hp, hm)] * 5, [5] * 90, [5, 9] * 45)
    assert rendered["g_temp"] == 0.0 and len(rendered["template_aurocs"]) == 5


def test_template_control_subset_is_fixed():
    pairs = [{"id": i} for i in range(1000)]
    a = probes.template_control_subset(pairs)
    assert len(a) == 200 and a == probes.template_control_subset(pairs)
    assert cosine(np.ones(3), np.ones(3)) == pytest.approx(1.0)
