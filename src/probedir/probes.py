"""Probe directions: estimation, layer selection, and stability.

Implements the paper's probe-side quantities:

* mean-difference (MD) direction, Eq. 2 (Sec. 3.2)
* L2-regularized logistic-regression (LR) directions on raw or feature-wise
  standardized representations, mapped back to activation space, Eq. 10 (App. B.2)
* training-only layer selection by MD AUROC, Eq. 11 (App. B.2, Table 9)
* held-out AUROC of the projection score s(h; v) = h^T v (Sec. 3.3)
* split-half estimation stability, Eq. 3 (Sec. 3.3, App. C.1)
* cosine alignment with MD and participation ratio (App. C.2, Tables 1, 13, 14)
* the prompt-template and input-length controls G_temp and G_len (App. D.1, Table 21)

All functions take NumPy arrays ``h_plus`` / ``h_minus`` of shape ``[n, d]``
(row i of both arrays is pair i). Directions are oriented with correct answers
as the positive class and are never sign-adjusted with held-out data.
"""
from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .stats import auroc, cosine, unit

# Split-half partitions (Eq. 3): one permutation of the training pairs per seed, K = 5.
SPLIT_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 999)

# Logistic-regression settings shared by every LR direction (App. B.2).
LR_SETTINGS: dict[str, Any] = dict(solver="lbfgs", penalty="l2", max_iter=1000, tol=1e-4,
                                   fit_intercept=True, class_weight=None, random_state=42)


@dataclass(frozen=True)
class Estimator:
    """A direction estimator (Table 8)."""

    name: str
    kind: str                    # "md" or "lr"
    C: float | None = None       # inverse L2 strength for LR
    standardize: bool = False    # fit LR on feature-wise standardized representations

    @property
    def label(self) -> str:
        if self.kind == "md":
            return "Mean Difference"
        return f"{'Standardized' if self.standardize else 'Raw'} LR (C={self.C:g})"


MD = Estimator("D0", "md")
ESTIMATORS: dict[str, Estimator] = {e.name: e for e in (
    MD,
    Estimator("D1", "lr", 1e-4),
    Estimator("D2", "lr", 1e-2),
    Estimator("D3", "lr", 1.0),
    Estimator("D4", "lr", 1e-4, standardize=True),
    Estimator("D5", "lr", 1.0, standardize=True),
)}
# The 7-8B sweep uses D0-D5 (Table 13); the 14B extension uses the raw grid D0-D3 (Table 14).
SWEEP_7B = ("D0", "D1", "D2", "D3", "D4", "D5")
SWEEP_14B = ("D0", "D1", "D2", "D3")


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def md_vector(h_plus: np.ndarray, h_minus: np.ndarray) -> np.ndarray:
    """Un-normalized MD vector: mean_i (h+_i - h-_i), computed in float64 (Eq. 2)."""
    return (np.asarray(h_plus, np.float64).mean(0) - np.asarray(h_minus, np.float64).mean(0))


def md_direction(h_plus: np.ndarray, h_minus: np.ndarray) -> np.ndarray:
    """Unit MD direction (Eq. 2)."""
    return unit(md_vector(h_plus, h_minus))


def fit_lr(h_plus: np.ndarray, h_minus: np.ndarray, C: float,
           standardize: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit L2 logistic regression (correct = 1) and return the original-space weight.

    Features are passed to scikit-learn as float32, which represents FP16 hidden
    states exactly. For standardized probes the scaler is fit on the given
    (training) rows only, and the coefficient is mapped back with
    v_LR = w_z / sigma (Eq. 10).
    """
    x = np.concatenate((h_plus, h_minus)).astype(np.float32, copy=False)
    y = np.r_[np.ones(len(h_plus), dtype=np.int8), np.zeros(len(h_minus), dtype=np.int8)]
    scaler = StandardScaler() if standardize else None
    if scaler is not None:
        x = scaler.fit_transform(x)
    clf = LogisticRegression(C=C, **LR_SETTINGS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    weight = np.asarray(clf.coef_[0], dtype=np.float64)
    if scaler is not None:
        weight = weight / np.asarray(scaler.scale_, dtype=np.float64)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    n_iter = int(clf.n_iter_[0])
    return weight, {"n_iter": n_iter, "converged": (not warned) and n_iter < LR_SETTINGS["max_iter"]}


def lr_direction(h_plus: np.ndarray, h_minus: np.ndarray, C: float, standardize: bool = False,
                 orient: np.ndarray | None = None) -> np.ndarray:
    """Unit LR direction in activation space.

    If ``orient`` (e.g. the training MD direction) is given, the sign is chosen so
    that the direction has a non-negative dot product with it.
    """
    w = unit(fit_lr(h_plus, h_minus, C, standardize)[0])
    if orient is not None and float(np.dot(w, orient)) < 0:
        w = -w
    return w


def estimate(estimator: Estimator | str, h_plus: np.ndarray, h_minus: np.ndarray) -> np.ndarray:
    """Un-normalized weight vector of ``estimator`` fit on the given rows."""
    est = ESTIMATORS[estimator] if isinstance(estimator, str) else estimator
    if est.kind == "md":
        return md_vector(h_plus, h_minus)
    return fit_lr(h_plus, h_minus, est.C, est.standardize)[0]


def heldout_auroc(direction: np.ndarray, h_plus: np.ndarray, h_minus: np.ndarray) -> float:
    """AUROC of the projection score h^T v, correct answers as positives (Sec. 3.3)."""
    v = np.asarray(direction, np.float64)
    return auroc(np.asarray(h_plus, np.float64) @ v, np.asarray(h_minus, np.float64) @ v)


def participation_ratio(weight: np.ndarray) -> float:
    """(sum_i v_i^2)^2 / sum_i v_i^4 (App. C.2)."""
    sq = np.square(np.asarray(weight, np.float64))
    return float(sq.sum() ** 2 / np.square(sq).sum())


# ---------------------------------------------------------------------------
# Stability and layer selection
# ---------------------------------------------------------------------------

def split_halves(n: int, seeds: Sequence[int] = SPLIT_SEEDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random halves (A_k, B_k) of n training pairs: permute, cut at floor(n/2)."""
    out = []
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(n)
        out.append((perm[: n // 2], perm[n // 2:]))
    return out


def split_half_stability(h_plus: np.ndarray, h_minus: np.ndarray, estimator: Estimator | str = MD,
                         seeds: Sequence[int] = SPLIT_SEEDS, question_keys=None) -> dict[str, Any]:
    """Mean signed cosine between directions fit independently on random halves (Eq. 3)."""
    cosines = []
    from .data import question_half

    if question_keys is None:
        halves = split_halves(len(h_plus), seeds)
    else:
        if len(question_keys) != len(h_plus):
            raise ValueError("training questions and hidden-state rows must align")
        masks = [question_half(question_keys, seed) for seed in seeds]
        halves = [(mask, ~mask) for mask in masks]
    for a, b in halves:
        cosines.append(cosine(estimate(estimator, h_plus[a], h_minus[a]),
                              estimate(estimator, h_plus[b], h_minus[b])))
    return {"seeds": list(seeds), "cosines": cosines, "mean": float(np.mean(cosines)),
            "min": float(np.min(cosines)), "max": float(np.max(cosines))}


def layer_profile(cache, layers: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """Train and held-out MD AUROC for every stored layer of a ``HiddenCache``.

    The MD direction of each layer is fit on the training split only.
    """
    rows = []
    for layer in layers if layers is not None else cache.layers:
        tp, tm = cache.layer(layer, "train")
        ep, em = cache.layer(layer, "test")
        raw = md_vector(tp, tm)
        v = unit(raw)
        rows.append({"layer": int(layer), "train_auroc": heldout_auroc(v, tp, tm),
                     "test_auroc": heldout_auroc(v, ep, em), "md_norm": float(np.linalg.norm(raw))})
    return rows


def select_layer_from_profile(rows: Sequence[dict[str, Any]]) -> int:
    """Layer with the highest training MD AUROC; ties go to the lower layer (Eq. 11)."""
    return int(max(rows, key=lambda r: (r["train_auroc"], -r["layer"]))["layer"])


def select_layer(cache) -> int:
    """Training-only probing-layer selection (Eq. 11, Table 9)."""
    return select_layer_from_profile(layer_profile(cache))


# ---------------------------------------------------------------------------
# Estimator sweep (Tables 1, 13, 14)
# ---------------------------------------------------------------------------

def estimator_sweep(train_plus: np.ndarray, train_minus: np.ndarray,
                    test_plus: np.ndarray, test_minus: np.ndarray,
                    estimators: Sequence[str] = SWEEP_7B,
                    seeds: Sequence[int] = SPLIT_SEEDS, question_keys=None) -> dict[str, dict[str, Any]]:
    """Fit each estimator on the training pairs at a fixed layer and evaluate it.

    Returns, per estimator: unit ``direction``, held-out AUROC, cosine alignment
    with the training MD direction, split-half stability, participation ratio and
    LR convergence information. LR directions are oriented to the training MD.
    """
    md = md_direction(train_plus, train_minus)
    out: dict[str, dict[str, Any]] = {}
    for name in estimators:
        est = ESTIMATORS[name]
        if est.kind == "md":
            weight, info = md_vector(train_plus, train_minus), {"n_iter": None, "converged": True}
        else:
            weight, info = fit_lr(train_plus, train_minus, est.C, est.standardize)
            if float(np.dot(weight, md)) < 0:
                weight = -weight
        v = unit(weight)
        out[name] = {
            "estimator": est.label, "C": est.C, "standardized": est.standardize, "direction": v,
            "heldout_auroc": heldout_auroc(v, test_plus, test_minus),
            "cos_with_md": cosine(v, md),
            "participation_ratio": participation_ratio(weight),
            "split_half": split_half_stability(train_plus, train_minus, est, seeds, question_keys),
            **info,
        }
    return out


# ---------------------------------------------------------------------------
# App. D.1: prompt-template and input-length controls (Table 21)
# ---------------------------------------------------------------------------

# System messages of the template control. Templates 0-2 are the three
# templates assigned across pairs for the 7-8B models (field ``template_id``);
# the 14B control renders the same held-out subset under all five.
TEMPLATE_SYSTEM_MESSAGES: tuple[str, ...] = (
    "You are a helpful assistant.",
    "You are a helpful assistant. Answer the user's question directly.",
    "You are a knowledgeable AI assistant that helps users with various tasks.",
    "Provide concise and factual answers.",
    "Answer carefully using your knowledge; do not add unnecessary explanation.",
)
TEMPLATE_CONTROL_SEED = 20260713    # 14B fixed held-out subset
TEMPLATE_CONTROL_PAIRS = 200
CONTROL_TOLERANCE = 0.05            # predefined diagnostic tolerance for G_temp and G_len


def orientation_free_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """max(AUROC, 1 - AUROC), as used for the subgroup diagnostics of App. D.1."""
    a = auroc(pos, neg)
    return max(a, 1.0 - a)


def template_control_subset(heldout_pairs: Sequence[dict], n: int = TEMPLATE_CONTROL_PAIRS,
                            seed: int = TEMPLATE_CONTROL_SEED) -> list[dict]:
    """Fixed held-out subset rendered under every template in the 14B control."""
    return random.Random(seed).sample(list(heldout_pairs), min(n, len(heldout_pairs)))


def length_control(pos: np.ndarray, neg: np.ndarray, len_plus: np.ndarray,
                   len_minus: np.ndarray) -> dict[str, Any]:
    """G_len: AUROC gap between pairs at or below vs. above the median absolute
    difference in tokenized input length."""
    diff = np.abs(np.asarray(len_plus) - np.asarray(len_minus))
    median = float(np.median(diff))
    small = diff <= median
    a_small = orientation_free_auroc(pos[small], neg[small])
    a_large = orientation_free_auroc(pos[~small], neg[~small])
    return {"median_abs_input_length_difference": median, "n_small": int(small.sum()),
            "n_large": int((~small).sum()), "auroc_small": a_small, "auroc_large": a_large,
            "g_len": abs(a_small - a_large)}


def template_control(scores_by_template: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """G_temp: range of per-template AUROCs."""
    aurocs = [orientation_free_auroc(p, n) for p, n in scores_by_template]
    return {"template_aurocs": aurocs, "g_temp": max(aurocs) - min(aurocs)}


def template_length_control_assigned(direction: np.ndarray, h_plus: np.ndarray, h_minus: np.ndarray,
                                     template_ids: Sequence[int], len_plus: Sequence[int],
                                     len_minus: Sequence[int]) -> dict[str, Any]:
    """D.1 for the 7-8B models: templates are assigned across held-out pairs."""
    v = np.asarray(direction, np.float64)
    pos, neg = np.asarray(h_plus, np.float64) @ v, np.asarray(h_minus, np.float64) @ v
    tids = np.asarray(template_ids)
    groups = [(pos[tids == t], neg[tids == t]) for t in sorted(set(tids.tolist()))]
    return {"n_pairs": len(pos), **template_control(groups),
            **length_control(pos, neg, np.asarray(len_plus), np.asarray(len_minus))}


def template_length_control_rendered(direction: np.ndarray,
                                     per_template: Sequence[tuple[np.ndarray, np.ndarray]],
                                     len_plus: Sequence[int], len_minus: Sequence[int]) -> dict[str, Any]:
    """D.1 for the 14B models: the same subset rendered under each system message.

    ``per_template[t]`` holds (h_plus, h_minus) under template t; input lengths
    are those of template 0, on which G_len is computed.
    """
    v = np.asarray(direction, np.float64)
    scores = [(np.asarray(hp, np.float64) @ v, np.asarray(hm, np.float64) @ v) for hp, hm in per_template]
    return {"n_pairs": len(scores[0][0]), **template_control(scores),
            **length_control(*scores[0], np.asarray(len_plus), np.asarray(len_minus))}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def training_questions(cache, dataset):
    if dataset != "triviaqa":
        return None
    from .data import load_pairs, question_key

    by_id = {r["id"]: r for r in load_pairs(dataset)}
    keys = []
    for row_id, split in zip(cache.ids, cache.split):
        pair = by_id[row_id]
        if pair["split"] != split:
            raise ValueError("TriviaQA cache split differs from the camera-ready pair manifest")
        if split == "train":
            keys.append(question_key(pair["question"]))
    return keys


def run_probes(model: str, dataset: str) -> dict[str, Any]:
    """Layer profile and selection, MD at the selected layer, and split-half stability."""
    from safetensors.numpy import save_file

    from .config import results_dir
    from .io import HiddenCache, cache_path, write_json

    cache = HiddenCache.load(cache_path(model, dataset, "fp16"))
    profile = layer_profile(cache)
    layer = select_layer_from_profile(profile)
    tp, tm = cache.layer(layer, "train")
    ep, em = cache.layer(layer, "test")
    v = md_direction(tp, tm)
    result = {"model": model, "dataset": dataset, "selected_layer": layer,
              "train_n": len(tp), "heldout_n": len(ep), "heldout_auroc": heldout_auroc(v, ep, em),
              "split_half": split_half_stability(tp, tm, question_keys=training_questions(cache, dataset)), "layer_profile": profile}
    out = results_dir() / "probes" / model / dataset
    write_json(out / "md.json", result)   # creates ``out``
    save_file({"md": v.astype(np.float32)}, str(out / "md_direction.safetensors"))
    return result


def run_estimator_sweep(model: str, dataset: str, layer: int | None = None) -> dict[str, Any]:
    from safetensors.numpy import save_file

    from .config import Cell, results_dir
    from .io import HiddenCache, cache_path, write_json

    cell = Cell(model, dataset)
    layer = cell.layer if layer is None else layer
    cache = HiddenCache.load(cache_path(model, dataset, "fp16"))
    tp, tm = cache.layer(layer, "train")
    ep, em = cache.layer(layer, "test")
    sweep = estimator_sweep(tp, tm, ep, em, SWEEP_14B if cell.is_14b else SWEEP_7B,
                            question_keys=training_questions(cache, dataset))
    out = results_dir() / "estimator_sweep" / model / dataset
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: r["direction"].astype(np.float32) for k, r in sweep.items()}, str(out / "directions.safetensors"))
    summary = {k: {kk: vv for kk, vv in r.items() if kk != "direction"} for k, r in sweep.items()}
    write_json(out / "sweep.json", {"model": model, "dataset": dataset, "layer": layer, "estimators": summary})
    return summary


def run_template_control(model: str, dataset: str, layer: int | None = None) -> dict[str, Any]:
    """D.1 for caches whose rows carry ``template_id`` and input lengths (7-8B design)."""
    import json

    from .config import Cell, pairs_path, results_dir
    from .io import HiddenCache, cache_path, write_json

    cell = Cell(model, dataset)
    layer = cell.layer if layer is None else layer
    cache = HiddenCache.load(cache_path(model, dataset, "fp16"))
    tid = {p["id"]: p["template_id"] for p in json.loads(pairs_path(dataset).read_text())}
    test = np.asarray(cache.split) == "test"
    ids = [i for i, keep in zip(cache.ids, test) if keep]
    lp = np.asarray(cache.meta["input_len_plus"])[test]
    lm = np.asarray(cache.meta["input_len_minus"])[test]
    v = md_direction(*cache.layer(layer, "train"))
    ep, em = cache.layer(layer, "test")
    result = {"model": model, "dataset": dataset, "layer": layer,
              **template_length_control_assigned(v, ep, em, [tid[i] for i in ids], lp, lm)}
    write_json(results_dir() / "template_control" / model / dataset / "d1.json", result)
    return result


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("probes", help="layer selection, MD direction and split-half stability (Sec. 3.2-3.3)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.set_defaults(func=lambda a: run_probes(a.model, a.dataset))

    p = subparsers.add_parser("estimator-sweep", help="MD vs. LR estimator sweep at the selected layer (Tables 1, 13, 14)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="triviaqa")
    p.add_argument("--layer", type=int)
    p.set_defaults(func=lambda a: run_estimator_sweep(a.model, a.dataset, a.layer))

    p = subparsers.add_parser("template-control", help="prompt-template and input-length controls (App. D.1)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--layer", type=int)
    p.set_defaults(func=lambda a: run_template_control(a.model, a.dataset, a.layer))
