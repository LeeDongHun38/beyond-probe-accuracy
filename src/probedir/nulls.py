"""Norm-matched isotropic random-direction nulls for steering specificity (Sec. 3.3; App. C.4).

Each random direction is a standard-normal draw in R^d, unit-normalized, and applied
with the same layer, token positions and magnitude alpha as the probe direction. For
side c in {false, true} the statistic is the absolute seed-mean effect (Eq. 6)

    T_c(u) = | mean_seeds( mean_i  Delta P_i(u) ) |,

and the Monte-Carlo p-value is p = (1 + #{j : T_c(r_j) >= T_c(v)}) / (127 + 1).
Benjamini-Hochberg is applied over the 30 model x dataset x side tests at fixed alpha.

The generator seeds below reproduce the exact random directions used in the paper.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .config import results_dir
from .stats import bh_qvalues, empirical_p

N_RANDOM = 127

# torch.Generator seeds of the 127-direction sets used for each model-dataset cell
# (fixed alpha = 4).
RANDOM_DIRECTION_SEEDS: dict[tuple[str, str], int] = {
    ("llama-3.1-8b", "triviaqa"): 2026073400,
    ("qwen2.5-7b", "triviaqa"): 2026073401,
    ("mistral-7b", "triviaqa"): 2026073402,
    ("llama-3.1-8b", "arc_easy"): 2026073200,
    ("qwen2.5-7b", "arc_easy"): 2026073201,
    ("mistral-7b", "arc_easy"): 2026073202,
    ("llama-3.1-8b", "commonsenseqa"): 2026073300,
    ("qwen2.5-7b", "commonsenseqa"): 2026073301,
    ("mistral-7b", "commonsenseqa"): 2026073302,
    ("qwen2.5-14b", "triviaqa"): 2026073403,
    ("qwen2.5-14b", "arc_easy"): 2026080102,
    ("qwen2.5-14b", "commonsenseqa"): 2026080103,
    ("ministral-3-14b", "triviaqa"): 2026073404,
    ("ministral-3-14b", "arc_easy"): 2026072902,
    ("ministral-3-14b", "commonsenseqa"): 2026072903,
}

# TriviaQA r = .05 isotropic controls (Table 19).
SENSITIVITY_SEEDS: dict[tuple[str, str], int] = {
    ("llama-3.1-8b", "triviaqa"): 2026073001,
    ("qwen2.5-7b", "triviaqa"): 2026073002,
    ("mistral-7b", "triviaqa"): 2026073003,
    ("qwen2.5-14b", "triviaqa"): 2026080101,
}


def random_directions(d: int, n: int = N_RANDOM, seed: int = 0) -> np.ndarray:
    """``n`` isotropic unit directions in R^d (float32), reproducing the paper's draws.

    Uses ``torch.randn`` on a CPU ``torch.Generator`` seeded with ``seed`` followed by
    row-wise L2 normalization in float32, exactly as the original runs did.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(n, d, generator=generator, dtype=torch.float32)
    return (raw / raw.norm(dim=1, keepdim=True)).contiguous().numpy()


def cell_random_directions(model: str, dataset: str, d: int, sensitivity: bool = False) -> np.ndarray:
    seeds = SENSITIVITY_SEEDS if sensitivity else RANDOM_DIRECTION_SEEDS
    return random_directions(d, N_RANDOM, seeds[(model, dataset)])


def side_statistic(seed_mean_effects: Sequence[float]) -> float:
    """T_c (Eq. 6): absolute mean over seeds of the seed-mean Delta P on one side."""
    return float(abs(np.mean(np.asarray(seed_mean_effects, dtype=float))))


def specificity_test(probe_seed_effects: Sequence[float],
                     random_seed_effects: np.ndarray) -> dict[str, float]:
    """Compare the probe direction with the random null on one side.

    probe_seed_effects:  [n_seeds] seed-mean Delta P of the probe direction
    random_seed_effects: [n_random, n_seeds] seed-mean Delta P of each random direction
    """
    random_seed_effects = np.asarray(random_seed_effects, dtype=float)
    observed_signed = float(np.mean(probe_seed_effects))
    t_probe = abs(observed_signed)
    t_null = np.abs(random_seed_effects.mean(axis=1))
    exceed, p = empirical_p(t_probe, t_null)
    return {
        "probe_effect": observed_signed,
        "T_probe": t_probe,
        "n_random": int(len(t_null)),
        "null_mean": float(t_null.mean()),
        "null_median": float(np.median(t_null)),
        "null_sd": float(t_null.std(ddof=1)) if len(t_null) > 1 else 0.0,
        "null_q05": float(np.quantile(t_null, 0.05)),
        "null_q95": float(np.quantile(t_null, 0.95)),
        "null_max": float(t_null.max()),
        "exceedance": exceed,
        "p_mc": p,
    }


def family_bh(tests: Sequence[Mapping]) -> list[dict]:
    """Add BH q-values over one family of tests (e.g. the 30 fixed-alpha tests)."""
    q = bh_qvalues([t["p_mc"] for t in tests])
    return [{**t, "q_bh": float(qi)} for t, qi in zip(tests, q)]


def null_from_effects(rows: Sequence[Mapping], probe_key: str = "probe") -> list[dict]:
    """Build the fixed-alpha test family from long-format effects.

    ``rows`` have keys model, dataset, side, direction ('probe' or 'R000'..'R126'),
    seed, delta_p (seed-mean). Returns one test per (model, dataset, side) with BH q.
    """
    cells: dict[tuple, dict[str, dict[int, float]]] = {}
    for r in rows:
        key = (r["model"], r["dataset"], r["side"])
        cells.setdefault(key, {}).setdefault(r["direction"], {})[int(r["seed"])] = float(r["delta_p"])
    tests = []
    for (model, dataset, side), by_dir in cells.items():
        seeds = sorted(by_dir[probe_key])
        probe = [by_dir[probe_key][s] for s in seeds]
        randoms = np.array([[by_dir[d][s] for s in seeds] for d in sorted(by_dir) if d != probe_key])
        tests.append({"model": model, "dataset": dataset, "side": side,
                      **specificity_test(probe, randoms)})
    return family_bh(tests)


def _cmd_family(args: argparse.Namespace) -> None:
    """Assemble the fixed-alpha specificity family from steering outputs."""
    root = Path(args.root) if args.root else results_dir() / "steering"
    rows = []
    for path in sorted(root.glob("*/*/alpha4/summary.json")):
        payload = json.loads(path.read_text())
        rows.extend(payload["seed_effects"])
    tests = null_from_effects(rows)
    out = results_dir() / "steering" / "specificity_fixed_alpha.json"
    out.write_text(json.dumps({"family": "fixed alpha=4, 30 tests", "tests": tests}, indent=2))
    print(f"wrote {out} ({len(tests)} tests)")


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("null-family", help="random-direction specificity tests + BH (App. C.4)")
    p.add_argument("--root", default=None, help="steering results root (default: results/steering)")
    p.set_defaults(func=_cmd_family)
