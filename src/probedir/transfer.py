"""Fixed-direction cross-precision transfer (Sec. 3.5, 6.1; App. E.1, E.3; Tables 2, 25, 28).

The probing layer and the unit MD direction ``v`` are estimated once, on the FP16
training split. They are applied unchanged to the held-out representations of
each quantized model: no probe refit, no layer reselection, no sign adjustment.

Predictive transfer (Table 2, Table 25)
    Held-out AUROC of the projection score ``h^T v``. Correct and incorrect
    candidates are scored as independent samples. The drop is
    ``AUROC_FP16 - AUROC_q``.

Pair margins (App. E.3, Table 28)
    ``g_i = v^T (h+_i - h-_i)`` per held-out pair; positive means the correct
    answer scores higher. Reported quantities:
    - relative shrinkage ``(mean g^FP16 - mean g^q) / mean g^FP16``, a ratio of means;
    - the share of pairs whose margin shrinks;
    - sign flips ``sign(g^FP16) != sign(g^q)``, with ``sign(0) = 0``, so a zero
      margin counts as a flip only when the other margin is non-zero.
"""
from __future__ import annotations

import argparse

import numpy as np

from .stats import auroc

QUANTIZATIONS = ("awq", "gptq", "rtn", "w8a8", "w4a8")


def transfer_metrics(fp_plus, fp_minus, q_plus, q_minus, v: np.ndarray) -> dict:
    """AUROC transfer and pair-margin diagnostics for one quantization condition."""
    v = np.asarray(v, np.float64)
    v = v / np.linalg.norm(v)
    fp_p, fp_m = np.asarray(fp_plus, np.float64), np.asarray(fp_minus, np.float64)
    q_p, q_m = np.asarray(q_plus, np.float64), np.asarray(q_minus, np.float64)
    s_fp_p, s_fp_m, s_q_p, s_q_m = fp_p @ v, fp_m @ v, q_p @ v, q_m @ v
    auc_fp, auc_q = auroc(s_fp_p, s_fp_m), auroc(s_q_p, s_q_m)
    g_fp, g_q = s_fp_p - s_fp_m, s_q_p - s_q_m
    mean_fp, mean_q = float(g_fp.mean()), float(g_q.mean())
    flips = np.sign(g_fp) != np.sign(g_q)
    return {
        "n_heldout_pairs": int(len(g_fp)),
        "fp16_auroc": auc_fp,
        "quantized_auroc": auc_q,
        "auroc_drop": auc_fp - auc_q,
        "mean_margin_fp16": mean_fp,
        "mean_margin_quantized": mean_q,
        "mean_margin_change": mean_q - mean_fp,
        "relative_shrinkage": (mean_fp - mean_q) / mean_fp if abs(mean_fp) > 1e-12 else None,
        "positive_shrinkage_rate": float(np.mean((g_fp - g_q) > 0)),
        "margin_flip_n": int(flips.sum()),
        "margin_flip_rate": float(flips.mean()),
    }


def run(model: str, dataset: str, direction_file: str | None = None) -> list[dict]:
    """Transfer metrics for all quantization conditions of one cell (direction as in geometry.run)."""
    from .config import Cell, results_dir
    from .io import HiddenCache, cache_path, write_json
    from .probes import md_direction

    cell = Cell(model, dataset)
    fp = HiddenCache.load(cache_path(model, dataset, "fp16"))
    v = np.load(direction_file) if direction_file else md_direction(*fp.layer(cell.layer, "train"))
    fp_p, fp_m = fp.layer(cell.layer, "test")
    fp_test_ids = [i for i, s in zip(fp.ids, fp.split) if s == "test"]
    rows = []
    for qn in QUANTIZATIONS:
        qc = HiddenCache.load(cache_path(model, dataset, qn))
        if [i for i, s in zip(qc.ids, qc.split) if s == "test"] != fp_test_ids:
            raise ValueError(f"{model}/{dataset}/{qn}: held-out ids differ from FP16")
        rows.append({"model": model, "dataset": dataset, "layer": cell.layer, "quantization": qn,
                     **transfer_metrics(fp_p, fp_m, *qc.layer(cell.layer, "test"), v)})
    write_json(results_dir() / "transfer" / model / dataset / "transfer.json", rows)
    return rows


def add_cli(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("transfer", help="fixed FP16 direction on quantized models (Table 2, App. E.1/E.3)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--direction", help="optional .npy direction (default: FP16 training MD)")
    p.set_defaults(func=lambda a: run(a.model, a.dataset, a.direction))
