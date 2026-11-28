import numpy as np
import pytest

from probedir import geometry as G


def _pairs(n=300, d=64, seed=0):
    rng = np.random.default_rng(seed)
    hm = rng.normal(size=(n, d)).astype(np.float32)
    shift = np.zeros(d, np.float32)
    shift[0] = 2.0
    hp = hm + shift + 0.5 * rng.normal(size=(n, d)).astype(np.float32)
    return hp, hm


def _md(hp, hm):
    v = hp.mean(0, dtype=np.float64) - hm.mean(0, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_paper_seed_labels():
    # Seeds of two runs reported in the paper (7-8B TriviaQA null; 14B / Ministral TriviaQA PCA).
    assert G.paper_seed_scheme("llama-3.1-8b", "triviaqa").null_seed("rtn") == 4292478624
    assert G.derive_seed(20260731, "triviaqa", 36, "pca", sep="|") == G.paper_seed_scheme(
        "ministral-3-14b", "triviaqa").pca_seed()
    assert G.paper_seed_scheme("qwen2.5-14b", "arc_easy").master == 20260802


def test_orthonormal_basis_keeps_v_first_and_nested_spans():
    rng = np.random.default_rng(1)
    v = rng.normal(size=40)
    c = rng.normal(size=(40, 6))
    b = G.orthonormal_basis(v, c, 6)
    assert np.allclose(b.T @ b, np.eye(7), atol=1e-12)
    assert np.allclose(b[:, 0], v / np.linalg.norm(v))
    # span(b[:, 1:3]) contains the v-removed first two candidates
    c2 = c[:, :2] - np.outer(b[:, 0], b[:, 0] @ c[:, :2])
    resid = c2 - b[:, 1:3] @ (b[:, 1:3].T @ c2)
    assert np.abs(resid).max() < 1e-10


def test_contrastive_subspace_is_orthonormal_and_pool_excludes_v():
    hp, hm = _pairs()
    v = _md(hp, hm)
    sub = G.contrastive_subspace(hp, hm, v, seed=3, k_max=10)
    assert sub.basis.shape == (64, 10)
    assert np.allclose(sub.basis.T @ sub.basis, np.eye(10), atol=1e-10)
    assert np.abs(sub.pool.astype(np.float64) @ v).max() < 1e-4


def test_error_statistics_bounds_and_definitions():
    hp, hm = _pairs()
    v = _md(hp, hm)
    sub = G.contrastive_subspace(hp, hm, v, seed=3, k_max=10)
    rng = np.random.default_rng(2)
    err = rng.normal(size=(50, 64))
    ks = (1, 2, 5, 10)
    s = G.error_statistics(err, sub.basis, ks)
    assert np.all(np.diff(s["R"]) >= -1e-12) and 0 <= s["R"][0] and s["R"][-1] <= 1
    assert s["R_inc"][0] == 0
    assert np.isclose(s["R"][0], np.square(err @ v).sum() / np.square(err).sum())
    mean = err.mean(0)
    assert np.isclose(s["M"][0], (mean @ v) ** 2)
    assert np.allclose(s["A"], s["M"] / (mean @ mean))
    # error entirely inside U_k -> R_k = 1
    inside = rng.normal(size=(20, 5)) @ sub.basis[:, :5].T
    assert np.isclose(G.error_statistics(inside, sub.basis, ks)["R"][2], 1.0)


def test_null_is_deterministic_and_nonnegative():
    hp, hm = _pairs()
    v = _md(hp, hm)
    sub = G.contrastive_subspace(hp, hm, v, seed=3, k_max=10)
    err = np.random.default_rng(4).normal(size=(40, 64))
    a = G.activation_matched_null(err, sub, seed=11, n_null=15, ks=(1, 2, 5, 10))
    b = G.activation_matched_null(err, sub, seed=11, n_null=15, ks=(1, 2, 5, 10))
    assert np.array_equal(a["R"], b["R"])
    assert a["R"].shape == (15, 4)
    assert (a["R_inc"] >= -1e-12).all()
    assert np.allclose(a["R"][:, 0], a["R"][0, 0])  # R_1 uses only v


def test_tests_and_bh():
    obs = {"R": np.array([.1, .5]), "R_inc": np.array([0, .4]), "M": np.array([1., 2.]), "A": np.array([.1, .2])}
    null = {k: np.zeros((99, 2)) for k in obs}
    rows = G.geometry_tests(obs, null, ks=(1, 2))
    assert {r["measure"] for r in rows} == {"R_inc", "M", "A"}
    assert all(r["exceedance"] == 0 and r["p"] == pytest.approx(0.01) for r in rows)
    G.add_bh(rows)
    assert all(r["significant"] for r in rows)


def test_label_permutation_panel_reuses_test_errors_and_refits_basis(tmp_path, monkeypatch):
    from probedir.io import HiddenCache, cache_path

    monkeypatch.setenv("PROBEDIR_RESULTS", str(tmp_path))
    model, dataset, layer = "qwen2.5-14b", "arc_easy", 30
    hp, hm = _pairs(n=90, d=52)
    rng = np.random.default_rng(42)
    ids = [str(i) for i in range(len(hp))]
    split = ["train"] * 60 + ["test"] * 30
    for precision in ("fp16", *G.QUANTIZATIONS):
        pos = hp if precision == "fp16" else hp + rng.normal(0, 0.1, hp.shape)
        HiddenCache(pos[None], hm[None], ids, split, [layer]).save(cache_path(model, dataset, precision))
    G.run_permutation(model, dataset, n_permutations=2)
    with np.load(tmp_path / "label_permutation" / model / dataset / "panel.npz") as panel:
        assert panel["null"].shape == (2, 5, 6)
        assert panel["observed"].shape == (5, 6)
        assert np.all(np.diff(panel["observed"], axis=-1) >= 0)
        assert not np.allclose(panel["null"][0], panel["null"][1])
