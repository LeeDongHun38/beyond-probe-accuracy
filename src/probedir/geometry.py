"""Differential quantization-error geometry (Sec. 3.5, Eq. 8; App. E.2-E.3).

For one model-dataset cell the FP16-selected layer and the FP16 training
mean-difference (MD) direction ``v`` are fixed. Quantized models are never
refit.

Task-defined contrastive subspace
    Training pair differences ``d_i = h+_i - h-_i`` are centred, the ``v``
    component is removed, and PCA on this residual variation gives orthonormal
    axes ``p_1, p_2, ...``. ``U_k = [v, p_1, ..., p_{k-1}]`` for
    ``k`` in {1, 2, 5, 10, 20, 50}.

Differential pair error
    For held-out pair ``i`` and quantization ``q``:
    ``e_i,q = (h+_q,i - h-_q,i) - (h+_FP16,i - h-_FP16,i)``. ``E_q`` stacks the
    rows. Errors common to both answers cancel.

Statistics, per k
    ``R_k = ||E U_k||_F^2 / ||E||_F^2`` is the share of differential error
    energy inside ``U_k``. ``R_inc,k = R_k - R_1`` is the share captured by the
    contrastive axes beyond ``v`` (the incremental statistic).
    ``M_k = ||mean(E) U_k||^2`` and ``A_k = M_k / ||mean(E)||^2`` are the
    auxiliary mean-error statistics (App. E.3).

Camera-ready label-permutation null (B = 10,000)
    Each training pair independently swaps labels. The MD and 49 residual PCs are
    refitted per permutation. Table 26 tests R1 and R50 jointly and R50-R1 separately.

Supplementary activation-matched null (B = 1,000)
    Each draw samples ``k_max - 1`` rows, without replacement, from the
    FP16-training residual pool (centred pair differences with ``v`` removed).
    It builds ``[v, q_1, ...]`` by float64 two-pass QR in the ambient hidden
    space and evaluates the same statistics. The p-value is one-sided with the
    +1 correction. BH-FDR is applied over the 375 R_inc tests, and separately
    over the 375 M and the 375 A tests.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .stats import bh_qvalues, empirical_p

KS: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
N_NULL = 1000
PCA_OVERSAMPLES = 10
POOL_MIN_NORM = 1e-6
QUANTIZATIONS = ("awq", "gptq", "rtn", "w8a8", "w4a8")


# --------------------------------------------------------------------------- seeds
def derive_seed(master: int, *parts: object, sep: str = ":") -> int:
    """32-bit seed: first four bytes (little endian) of sha256(``master<sep>part...``)."""
    raw = sep.join(map(str, (master, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


@dataclass(frozen=True)
class SeedScheme:
    """Seed labels for the PCA (randomized SVD) and for the null draws of one cell."""

    master: int
    pca_parts: tuple
    null_prefix: tuple
    null_suffix: str
    sep: str = ":"

    def pca_seed(self) -> int:
        return derive_seed(self.master, *self.pca_parts, sep=self.sep)

    def null_seed(self, quantization: str) -> int:
        return derive_seed(self.master, *self.null_prefix, quantization, self.null_suffix, sep=self.sep)


# Seed labels used for the runs reported in the paper. They reproduce the
# reported PCA axes and null draws exactly. New analyses may use any scheme.
_TAG_7B = {"llama-3.1-8b": "llama", "qwen2.5-7b": "qwen", "mistral-7b": "mistral"}
_TAG_14B = {"qwen2.5-14b": "qwen14b", "ministral-3-14b": "ministral14b"}


def paper_seed_scheme(model: str, dataset: str, layer: int | None = None) -> SeedScheme:
    if model in _TAG_7B:
        tag = _TAG_7B[model]
        if dataset == "triviaqa":
            return SeedScheme(20260727, (tag, "observed_pca"), (tag,), "activation_matched_residual")
        return SeedScheme(20260727, (tag, dataset, "observed_pca"), (tag, dataset), "activation_matched_residual")
    if model == "ministral-3-14b" and dataset == "triviaqa":
        layer = 36 if layer is None else layer
        return SeedScheme(20260731, ("triviaqa", layer, "pca"), ("triviaqa", layer), "activation_matched", sep="|")
    tag = _TAG_14B[model]
    return SeedScheme(20260802, (tag, dataset, "observed_pca"), (tag, dataset), "activation_matched_residual")


# ------------------------------------------------------------------- subspaces
def orthonormal_basis(v: np.ndarray, candidates: np.ndarray, extra: int) -> np.ndarray:
    """Return ``[v, q_1, ..., q_extra]`` (d x (extra+1)), with q orthonormal and orthogonal to v.

    ``candidates`` is d x m (m >= extra). Uses float64 two-pass QR after removing ``v``.
    QR keeps nested spans, so ``span(q_1..q_j) = span(c_1..c_j)`` minus ``v``.
    """
    v = np.asarray(v, np.float64)
    v = v / np.linalg.norm(v)
    c = np.array(candidates, dtype=np.float64)
    c -= np.outer(v, v @ c)
    q, _ = np.linalg.qr(c, mode="reduced")
    q -= np.outer(v, v @ q)
    q, _ = np.linalg.qr(q, mode="reduced")
    basis = np.column_stack([v, q[:, :extra]])
    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(extra + 1)))
    if gram_error > 3e-10:
        raise ArithmeticError(f"basis not orthonormal (Frobenius error {gram_error:g})")
    return basis


@dataclass
class ContrastiveSubspace:
    basis: np.ndarray                # d x k_max, column 0 = v
    pool: np.ndarray                 # float32 [n_pool, d] training residual activations (v removed)
    explained_variance_ratio: np.ndarray


def contrastive_subspace(h_plus_train: np.ndarray, h_minus_train: np.ndarray, v: np.ndarray,
                         seed: int, k_max: int = max(KS)) -> ContrastiveSubspace:
    """Build ``U_{k_max}`` and the null residual pool from FP16 training pairs (Sec. 3.5)."""
    v = np.asarray(v, np.float64)
    v = v / np.linalg.norm(v)
    d32 = np.asarray(h_plus_train, np.float32) - np.asarray(h_minus_train, np.float32)
    # PCA input: float32 centred differences with the v component removed.
    residual32 = d32 - d32.mean(axis=0, dtype=np.float64).astype(np.float32)
    v32 = v.astype(np.float32)
    residual32 -= np.outer(residual32 @ v32, v32)
    pca = PCA(n_components=k_max - 1, svd_solver="randomized", random_state=seed,
              n_oversamples=PCA_OVERSAMPLES)
    pca.fit(residual32)
    basis = orthonormal_basis(v, pca.components_.astype(np.float64).T, k_max - 1)
    # Null pool: the same residual activations, with v removed once more in float64.
    # A pair with identical representations (h+ = h-) leaves only a roundoff-size
    # residual. It is kept when its norm exceeds POOL_MIN_NORM, as in the paper's
    # runs, so that pool rows (and hence the sampled null draws) match them.
    pool64 = residual32.astype(np.float64)
    pool64 -= np.outer(pool64 @ v, v)
    pool = pool64[np.linalg.norm(pool64, axis=1) > POOL_MIN_NORM].astype(np.float32)
    if len(pool) < k_max - 1:
        raise ValueError("not enough non-zero training residual activations for the null")
    return ContrastiveSubspace(basis, pool, pca.explained_variance_ratio_.astype(np.float64))


# ------------------------------------------------------------------ statistics
def differential_error(fp_plus, fp_minus, q_plus, q_minus) -> np.ndarray:
    """``E_q``: rows ``(h+_q - h-_q) - (h+_FP16 - h-_FP16)`` over held-out pairs (float64)."""
    e = (np.asarray(q_plus, np.float32) - np.asarray(fp_plus, np.float32)
         - np.asarray(q_minus, np.float32) + np.asarray(fp_minus, np.float32))
    if not np.isfinite(e).all():
        raise ValueError("non-finite differential error")
    return e.astype(np.float64)


def error_statistics(error: np.ndarray, basis: np.ndarray, ks=KS) -> dict[str, np.ndarray]:
    """R_k, R_inc,k, M_k, A_k for all k from one projection onto ``basis`` (prefix sums)."""
    projected = error @ basis
    energy = np.cumsum(np.square(projected).sum(axis=0))
    total = float(np.square(error).sum())
    mean_error = error.mean(axis=0)
    mean_energy = np.cumsum(np.square(mean_error @ basis))
    idx = np.asarray(ks) - 1
    r = energy[idx] / total
    m = mean_energy[idx]
    return {"R": r, "R_inc": r - r[0], "M": m, "A": m / float(mean_error @ mean_error)}


def activation_matched_null(error: np.ndarray, subspace: ContrastiveSubspace, seed: int,
                            n_null: int = N_NULL, ks=KS) -> dict[str, np.ndarray]:
    """Null draws ``[n_null, len(ks)]`` of R, R_inc, M, A for random activation-matched subspaces."""
    v = subspace.basis[:, 0]
    pool = subspace.pool
    extra = max(ks) - 1
    rng = np.random.default_rng(seed)
    out = {key: np.empty((n_null, len(ks))) for key in ("R", "R_inc", "M", "A")}
    for b in range(n_null):
        rows = rng.choice(len(pool), size=extra, replace=False)
        basis = orthonormal_basis(v, pool[rows].T, extra)
        stats = error_statistics(error, basis, ks)
        for key in out:
            out[key][b] = stats[key]
    return out


def geometry_tests(observed: dict[str, np.ndarray], null: dict[str, np.ndarray], ks=KS) -> list[dict]:
    """One-sided empirical tests for R_inc, M and A at each k > 1 (+1-corrected p)."""
    rows = []
    for j, k in enumerate(ks):
        if k == 1:
            continue
        for measure in ("R_inc", "M", "A"):
            draws = null[measure][:, j]
            exceed, p = empirical_p(float(observed[measure][j]), draws)
            rows.append({"k": k, "measure": measure, "observed": float(observed[measure][j]),
                         "null_mean": float(draws.mean()), "null_median": float(np.median(draws)),
                         "null_sd": float(draws.std(ddof=1)), "n_null": int(len(draws)),
                         "exceedance": exceed, "p": p})
    return rows


def add_bh(rows: list[dict]) -> list[dict]:
    """BH q-values within each measure across all supplied cells (the paper's 375-test families)."""
    for measure in {r["measure"] for r in rows}:
        family = [r for r in rows if r["measure"] == measure]
        for r, q in zip(family, bh_qvalues([r["p"] for r in family])):
            r["q_bh"] = float(q)
            r["significant"] = bool(q < 0.05)
    return rows


def analyze_condition(h_plus_train, h_minus_train, fp_plus, fp_minus, q_plus, q_minus,
                      v: np.ndarray, pca_seed: int, null_seed: int, n_null: int = N_NULL,
                      subspace: ContrastiveSubspace | None = None) -> dict:
    """Observed statistics, null draws and tests for one quantization condition."""
    sub = subspace or contrastive_subspace(h_plus_train, h_minus_train, v, pca_seed)
    error = differential_error(fp_plus, fp_minus, q_plus, q_minus)
    observed = error_statistics(error, sub.basis)
    null = activation_matched_null(error, sub, null_seed, n_null)
    return {"observed": observed, "null": null, "tests": geometry_tests(observed, null)}


# ------------------------------------------------------------------------- CLI
def run(model: str, dataset: str, n_null: int = N_NULL, direction_file: str | None = None) -> None:
    """Run all quantization conditions of one cell.

    The direction is the FP16 training MD direction (float64), unless
    ``direction_file`` (.npy) supplies a fixed vector, e.g. the exact vectors of
    the paper's runs (7-8B ARC-Easy/CommonsenseQA used an MD vector stored at
    float16 precision).
    """
    from .config import Cell, results_dir
    from .io import HiddenCache, cache_path, write_json
    from .probes import md_direction

    cell = Cell(model, dataset)
    fp = HiddenCache.load(cache_path(model, dataset, "fp16"))
    hp_tr, hm_tr = fp.layer(cell.layer, "train")
    hp_te, hm_te = fp.layer(cell.layer, "test")
    v = np.load(direction_file) if direction_file else md_direction(hp_tr, hm_tr)
    seeds = paper_seed_scheme(model, dataset, cell.layer)
    sub = contrastive_subspace(hp_tr, hm_tr, v, seeds.pca_seed())
    out_dir = results_dir() / "geometry" / model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_test_ids = [i for i, s in zip(fp.ids, fp.split) if s == "test"]
    summary = []
    for qn in QUANTIZATIONS:
        qc = HiddenCache.load(cache_path(model, dataset, qn))
        if [i for i, s in zip(qc.ids, qc.split) if s == "test"] != fp_test_ids:
            raise ValueError(f"{model}/{dataset}/{qn}: held-out ids differ from FP16")
        qp, qm = qc.layer(cell.layer, "test")
        res = analyze_condition(hp_tr, hm_tr, hp_te, hm_te, qp, qm, v,
                                pca_seed=seeds.pca_seed(), null_seed=seeds.null_seed(qn),
                                n_null=n_null, subspace=sub)
        np.savez_compressed(out_dir / f"{qn}_null.npz", ks=np.asarray(KS),
                            **{f"observed_{k}": a for k, a in res["observed"].items()},
                            **{f"null_{k}": a for k, a in res["null"].items()})
        summary += [{"model": model, "dataset": dataset, "layer": cell.layer, "quantization": qn, **t}
                    for t in res["tests"]]
    write_json(out_dir / "tests.json", summary)


def collect_families() -> list[dict]:
    """Gather every cell's ``tests.json`` and add the 375-test BH q-values per measure."""
    import json

    from .config import results_dir

    rows = []
    for path in sorted((results_dir() / "geometry").glob("*/*/tests.json")):
        rows += json.loads(path.read_text())
    add_bh(rows)
    return rows


def permutation_axes(contrasts, seed):
    mean = contrasts.mean(axis=0)
    md = mean / np.linalg.norm(mean)
    residual = contrasts - mean
    residual -= np.outer(residual @ md, md)
    pca = PCA(n_components=49, svd_solver="randomized", random_state=seed,
              n_oversamples=20, iterated_power=7)
    v = pca.fit(residual).components_.T
    v -= np.outer(md, md @ v)
    v = np.linalg.qr(v, mode="reduced")[0]
    if np.max(np.abs(md @ v)) > 1e-10:
        raise RuntimeError("PCA basis leaks into fitted MD")
    return md, v


def permutation_energies(errors, denominators, md, v):
    output = np.empty((len(errors), len(KS)), dtype=np.float64)
    for j, error in enumerate(errors):
        r1 = np.square(error @ md).sum() / denominators[j]
        pc = np.square(error @ v).sum(axis=0)
        output[j, 0] = r1
        output[j, 1:] = r1 + np.cumsum(pc)[np.asarray(KS[1:]) - 2] / denominators[j]
    return output


def run_permutation(model, dataset, n_permutations=10000):
    """Refit MD and 49 residual PCs for every within-pair label swap (Table 26)."""
    from .config import Cell, results_dir
    from .io import HiddenCache, cache_path
    from .probes import training_questions

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    panel = [("qwen2.5-14b", "triviaqa"), ("qwen2.5-14b", "arc_easy"), ("qwen2.5-14b", "commonsenseqa"),
             ("ministral-3-14b", "arc_easy"), ("ministral-3-14b", "commonsenseqa")]
    if (model, dataset) not in panel:
        raise ValueError("The published label-permutation panel covers five 14B model–dataset cells")
    cell = Cell(model, dataset)
    fp = HiddenCache.load(cache_path(model, dataset))
    training_questions(fp, dataset)
    tp, tm = fp.layer(cell.layer, "train")
    contrasts = (tp.astype(np.float32) - tm.astype(np.float32)).astype(np.float64)
    ep, em = fp.layer(cell.layer, "test")
    ids = [rid for rid, split in zip(fp.ids, fp.split) if split == "test"]
    errors = []
    for precision in QUANTIZATIONS:
        quant = HiddenCache.load(cache_path(model, dataset, precision))
        if ids != [rid for rid, split in zip(quant.ids, quant.split) if split == "test"]:
            raise ValueError(f"Held-out IDs differ at {precision}")
        qp, qm = quant.layer(cell.layer, "test")
        errors.append(differential_error(ep, em, qp, qm))
    denominators = [np.square(error).sum() for error in errors]
    if min(denominators) <= 0:
        raise ValueError("Zero held-out error energy")
    seed = 2026091801
    rng = np.random.default_rng(seed + panel.index((model, dataset)))
    observed = permutation_energies(errors, denominators, *permutation_axes(contrasts, seed))
    null = np.empty((n_permutations, *observed.shape))
    for j in range(n_permutations):
        signs = rng.choice([-1., 1.], size=len(contrasts))
        axes = permutation_axes(contrasts * signs[:, None], seed)
        null[j] = permutation_energies(errors, denominators, *axes)
    dest = results_dir() / "label_permutation" / model / dataset
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "panel.npz", observed=observed, null=null, ks=KS,
                        layer=cell.layer, quantizations=QUANTIZATIONS)


def add_cli(subparsers: argparse._SubParsersAction) -> None:
    panel = subparsers.add_parser("geometry-permutation", help="refit the label-permuted PCA panel (Table 26)")
    panel.add_argument("--model", required=True)
    panel.add_argument("--dataset", required=True)
    panel.add_argument("--n-permutations", type=int, default=10000)
    panel.set_defaults(func=lambda a: run_permutation(a.model, a.dataset, a.n_permutations))

    p = subparsers.add_parser("geometry", help="quantization-error geometry and activation-matched null (Sec. 3.5)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-null", type=int, default=N_NULL)
    p.add_argument("--direction", help="optional .npy direction (default: FP16 training MD)")
    p.set_defaults(func=lambda a: run(a.model, a.dataset, a.n_null, a.direction))

    f = subparsers.add_parser("geometry-bh", help="BH-FDR over all geometry tests (375 per measure)")
    f.set_defaults(func=lambda a: _write_families())


def _write_families() -> None:
    from .config import results_dir
    from .io import write_json

    write_json(results_dir() / "geometry" / "bh_families.json", collect_families())
