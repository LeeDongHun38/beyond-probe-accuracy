"""Unit tests for probedir.controls (synthetic data, CPU)."""
import numpy as np
import pytest

from probedir import controls as C


def test_normalize_and_parse():
    assert C.normalize_answer("The  Beatles!") == "beatles"
    assert C.parse_short_answer("The answer is Paris\nbecause...") == "Paris"
    assert C.parse_short_answer("Answer:Paris") == "Answer:Paris"           # own-generation: needs whitespace
    assert C.parse_short_answer("Answer:Paris", direct=True) == "Paris"     # direct generation: does not


def test_label_answer_exact_span_and_negation():
    aliases = ["Greece"]
    assert C.label_answer("greece.", aliases)["label"] == "correct"
    assert C.label_answer("Athens, Greece", aliases)["label"] == "incorrect"
    assert C.label_answer("Athens, Greece", aliases, allow_span_match=True)["match_type"] == "span"
    # a negation near any occurrence excludes the answer
    neg = C.label_answer("Greece (part of Greece, not a country)", aliases, allow_span_match=True)
    assert neg["label"] == "excluded" and neg["match_type"] == "span_negation_context"
    assert C.label_answer("", aliases)["label"] == "excluded"


def test_p_correct_ignores_excluded():
    s = [{"label": "correct"}, {"label": "incorrect"}, {"label": "excluded"}, {"label": "correct"}]
    assert C.p_correct(s) == pytest.approx(2 / 3)
    assert C.p_correct([{"label": "excluded"}]) is None


def test_paired_accuracy_mcnemar():
    base = [True, False, False, True, None]
    cond = [True, True, True, False, True]
    r = C.paired_accuracy(base, cond)
    assert r["n_scored"] == 4 and r["improve_count"] == 2 and r["worsen_count"] == 1
    assert r["accuracy_delta"] == pytest.approx(0.25)
    assert 0 < r["mcnemar_exact_two_sided_p"] <= 1


def test_direct_generation_selection_is_disjoint_and_sorted():
    pairs = [{"id": f"q{i:04d}"} for i in range(1000)]
    a = C.select_direct_generation_questions(pairs, 200, 1)
    b = C.select_direct_generation_questions(pairs, 400, 2, exclude_ids=[p["id"] for p in a])
    ids_a, ids_b = [p["id"] for p in a], [p["id"] for p in b]
    assert ids_a == sorted(ids_a) and not set(ids_a) & set(ids_b) and len(ids_b) == 400


def test_arc_selection_shards():
    pairs = [{"id": f"t{i}", "split": "train"} for i in range(600)] + [{"id": f"h{i}", "split": "test"} for i in range(600)]
    s0, s1 = C.select_arc_owngen_questions(pairs, [f"t{i}" for i in range(50)], seed=7)
    assert len(s0) == len(s1) == 500
    assert sum(p["split"] == "train" for p in s0) == 250
    assert not {p["id"] for p in s0 + s1} & {f"t{i}" for i in range(50)}


def test_readouts_detect_signal():
    rng = np.random.default_rng(0)
    d, n = 16, 400
    v = np.zeros(d); v[0] = 1.0
    labels = rng.integers(0, 2, n)
    h = rng.normal(size=(n, d)) + 1.5 * labels[:, None] * v
    ids = np.repeat(np.arange(n // 4), 4).astype(str)
    split = np.where(np.arange(n) < n // 2, "train", "test")
    r = C.self_readout(h, labels, ids, split, v, n_boot=50)
    assert r["n"] == n // 2 and r["auroc"] > 0.75 and r["auroc_ci"][0] < r["auroc"] < r["auroc_ci"][1]
    y = list(labels.astype(float))
    p = C.pre_answer_readout(h, y, ids, split, v, n_boot=50)
    assert p["spearman"] > 0.3


def test_gold_donor_control_uses_fixed_direction():
    rng = np.random.default_rng(1)
    v = np.ones(8) / np.sqrt(8)
    pos, neg = rng.normal(size=(50, 8)) + v, rng.normal(size=(50, 8)) - v
    out = C.gold_donor_control(v, (pos, neg), (neg, pos))
    assert out["original_auroc"] > 0.8 and out["gold_donor_transfer_auroc"] < 0.2


def test_style_classifier_on_separable_strings():
    pairs = [{"split": s, "true_answer": f"alpha{i}", "false_answer": f"zzzz{i}"}
             for i in range(60) for s in ("train", "test")]
    assert C.style_classifier_auroc(pairs)["test_auroc"] > 0.9


def test_random_direction_streams():
    pytest.importorskip("torch")
    a = C.gaussian_directions_sequential(32, 5, seed=3)
    b = C.gaussian_directions_batched(32, 40, seed=3)
    assert a.shape == (5, 32) and b.shape == (40, 32)
    assert np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)
    assert np.allclose(a, C.gaussian_directions_sequential(32, 5, seed=3))


def test_span_readout_signal():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 300)
    v = np.eye(12)[0]
    h = rng.normal(size=(300, 12)) + 2 * labels[:, None] * v
    r = C.span_readout(h, labels, np.repeat(np.arange(100), 3).astype(str), v, n_boot=50, n_random=50)
    assert r["auroc"] > 0.8 and r["random_p"] == pytest.approx(1 / 51)
