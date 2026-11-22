"""Shared statistics: AUROC, empirical p-values, Benjamini-Hochberg, bootstrap."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import rankdata


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC of positives (correct answers, class 1) vs negatives, ties counted 1/2.

    No sign flipping and no max(AUC, 1-AUC): directions are oriented on training data.
    """
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    ranks = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def empirical_p(observed: float, null: np.ndarray, greater: bool = True) -> tuple[int, float]:
    """One-sided Monte-Carlo p-value with the +1 correction (Phipson & Smyth, 2010).

    Returns (exceedance count, p). Exceedance counts null draws >= observed
    (or <= observed when ``greater`` is False).
    """
    null = np.asarray(null, float)
    exceed = int(np.sum(null >= observed) if greater else np.sum(null <= observed))
    return exceed, (1 + exceed) / (len(null) + 1)


def bh_qvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) over one family."""
    p = np.asarray(pvalues, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def cluster_bootstrap_ci(stat: Callable[[np.ndarray], float], clusters: np.ndarray,
                         n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI from resampling cluster ids (e.g. questions) with replacement.

    ``stat`` receives an index array of resampled rows.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(stat(np.concatenate([rows_of[k] for k in pick])))
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))
