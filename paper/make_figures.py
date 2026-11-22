#!/usr/bin/env python3
"""Regenerate Figures 2-4 of the EACL 2026 paper from released artifacts.

Run from the repository root:

    python paper/make_figures.py            # writes paper/figures/fig{2,3,4}.{pdf,png}

Requires only numpy, pandas and matplotlib. No model inference is performed.
Every value plotted is read from ``artifacts/``; annotations that are printed
in the paper (r values, "29/30 above equality") are recomputed and asserted.

Figure 1 (question-matched pair example) is a text diagram typeset in LaTeX
and is not regenerated here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
SEC = ART / "paper_sections"
SRC = ART / "sources" / "results"
OUT = ROOT / "paper" / "figures"

# Colour-blind safe palette (same hues as the published figures).
BLUE, ORANGE, GREEN, VERMILION, PINK = "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC"
TEXTWIDTH_IN = 6.3  # ACL two-column \textwidth

MODELS = ["Llama-8B", "Qwen-7B", "Mistral-7B", "Qwen-14B", "Ministral-14B"]
DATASETS = ["TriviaQA", "ARC-Easy", "CommonsenseQA"]
ALPHA = 4.0

# Name normalisation across artifact files.
MODEL_ALIASES = {
    "Llama-3.1-8B": "Llama-8B", "Llama-3.1-8B-Instruct": "Llama-8B", "llama": "Llama-8B",
    "llama_3.1_8b": "Llama-8B",
    "Qwen2.5-7B": "Qwen-7B", "Qwen2.5-7B-Instruct": "Qwen-7B", "qwen2.5_7b_corrected": "Qwen-7B",
    "qwen": "Qwen-7B", "qwen2.5_7b": "Qwen-7B",
    "Mistral-7B": "Mistral-7B", "Mistral-7B-Instruct-v0.3": "Mistral-7B", "mistral": "Mistral-7B",
    "mistral_7b_v0.3": "Mistral-7B",
    "Qwen2.5-14B": "Qwen-14B", "Qwen2.5-14B-Instruct": "Qwen-14B", "qwen2.5_14b": "Qwen-14B",
    "Ministral-3-14B": "Ministral-14B", "Ministral-3-14B-Instruct": "Ministral-14B",
    "ministral3_14b": "Ministral-14B", "ministral_3_14b": "Ministral-14B",
}


def norm_model(name: str) -> str:
    return MODEL_ALIASES[name]


def read_csv_tolerant(path: Path) -> pd.DataFrame:
    """Read a CSV whose free-text last column may contain unquoted commas.

    ``d0_d5_canonical.csv`` has two rows (Qwen D4/D5) whose ``notes`` field
    contains an unquoted comma; overflow fields are re-joined into the last column.
    """
    import csv

    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, n = rows[0], len(rows[0])
    fixed = [r[: n - 1] + [",".join(r[n - 1:])] if len(r) > n else r for r in rows[1:]]
    return pd.DataFrame(fixed, columns=header)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "mathtext.fontset": "dejavusans",
    })


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None})  # deterministic output
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Shared inputs
# --------------------------------------------------------------------------
def corrected_qwen7b_triviaqa() -> dict:
    """Corrected Qwen2.5-7B TriviaQA L18 MD effects (offset-corrected extraction).

    The C3/C4 summary CSVs still carry the legacy Qwen-7B TriviaQA MD effect
    (+.126/-.178); the paper figures use the corrected re-execution instead.
    The random-direction null is direction-independent of the probe and is
    identical in both runs (max .01858/.02028).
    """
    by_seed = pd.read_csv(SEC / "C3_intervention" / "c3_qwen_corrected_triviaqa_by_seed.csv")
    agg = by_seed.groupby("condition")[["delta_p_false", "delta_p_true"]].mean()
    summary = json.loads((SRC / "qwen7b_corrected_l18_c3c4_rerun_v1" / "analysis" / "summary.json").read_text())
    nulls = {row["side"]: row for row in summary["job3_null_distribution"]}
    out = {
        "fixed_false": agg.loc["fixed_alpha_4", "delta_p_false"],
        "fixed_true": agg.loc["fixed_alpha_4", "delta_p_true"],
        "matched_false": agg.loc["matched_r_0.10", "delta_p_false"],
        "matched_true": agg.loc["matched_r_0.10", "delta_p_true"],
        "null_max_false": nulls["false"]["random_T_max"],
        "null_max_true": nulls["true"]["random_T_max"],
    }
    assert np.isclose(out["fixed_false"], nulls["false"]["T_probe"])
    assert np.isclose(-out["fixed_true"], nulls["true"]["T_probe"])
    return out


# --------------------------------------------------------------------------
# Figure 2: estimator-induced direction paths on TriviaQA at alpha = 4
# --------------------------------------------------------------------------
def figure2() -> None:
    est_order = ["MD", "1e-4", "1e-2", "1"]
    markers = {"MD": "o", "1e-4": "s", "1e-2": "D", "1": "^"}
    labels = {"MD": "MD", "1e-4": r"Raw $C=10^{-4}$", "1e-2": r"Raw $C=10^{-2}$", "1": r"Raw $C=1$"}

    c2 = SEC / "C2_estimator_sweep"
    d05 = read_csv_tolerant(c2 / "d0_d5_canonical.csv")
    d05 = d05[d05.direction_id.isin(["D0", "D1", "D2", "D3"])].copy()
    d05["est"] = d05.direction_id.map({"D0": "MD", "D1": "1e-4", "D2": "1e-2", "D3": "1"})
    d05["model"] = d05.model.map(norm_model)
    raw = pd.read_csv(c2 / "raw_lr_sweep_completed.csv", dtype={"C": str})
    raw = raw[raw.dataset == "TriviaQA"].copy()
    raw["est"] = raw.C.map({"MD": "MD", "0.0001": "1e-4", "0.01": "1e-2", "1.0": "1"})
    raw["model"] = raw.model.map(norm_model)

    # 7-8B cells: canonical explicit-system D0-D3 table; 14B cells: raw-LR sweep.
    rows = pd.concat([
        d05[d05.model.isin(["Llama-8B", "Qwen-7B", "Mistral-7B"])],
        raw[raw.model.isin(["Qwen-14B", "Ministral-14B"])],
    ])[["model", "est", "cos_with_md", "delta_p_false", "delta_p_true"]]
    # The two sources must agree on the shared (corrected) Qwen-7B cell. The LR rows are
    # independent re-fits of the same estimator (MD rows are identical), so agreement is
    # checked to 1e-3, well below plotting resolution.
    q_a = d05[d05.model == "Qwen-7B"].set_index("est")
    q_b = raw[raw.model == "Qwen-7B"].set_index("est")
    for col in ["cos_with_md", "delta_p_false", "delta_p_true"]:
        assert np.allclose(q_a.loc[est_order, col].astype(float), q_b.loc[est_order, col].astype(float), atol=1e-3), col

    # Residual-relative magnitude r = alpha / median ||h|| at the evaluator's final prefill
    # position (Eq. 7). 7-8B: explicit-system Job-1 baselines; 14B: canonical 15-cell norms.
    j1 = json.loads((SRC / "gpu_queue_explicit_system_v1" / "job1_e8_with" / "analysis" /
                     "with_residual_norm_medians.json").read_text())
    norms = {norm_model(k): v["pooled_median_residual_norm"] for k, v in j1.items()}
    cn = json.loads((SRC / "canonical_residual_norms_15cell" / "analysis" /
                     "canonical_residual_norms_15cell.json").read_text())
    for c in cn["cells"]:
        if c["dataset"] == "triviaqa" and norm_model(c["model"]) in ("Qwen-14B", "Ministral-14B"):
            norms[norm_model(c["model"])] = c["median_pooled"]
    r_vals = {m: ALPHA / norms[m] for m in MODELS}
    paper_r = {"Llama-8B": ".469", "Qwen-7B": ".051", "Mistral-7B": ".845", "Qwen-14B": ".035", "Ministral-14B": ".037"}
    for m in MODELS:
        assert f"{r_vals[m]:.3f}"[1:] == paper_r[m], (m, r_vals[m])

    fig, axes = plt.subplots(1, 5, figsize=(TEXTWIDTH_IN, 2.25), sharey=True)
    for ax, m in zip(axes, MODELS):
        sub = rows[rows.model == m].set_index("est").loc[est_order]
        sub = sub.astype({"cos_with_md": float, "delta_p_false": float, "delta_p_true": float})
        order = sub.sort_values("cos_with_md")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        for col, color in (("delta_p_false", BLUE), ("delta_p_true", VERMILION)):
            ax.plot(order.cos_with_md, order[col], color=color, lw=1.4, zorder=1)
            for est in est_order:  # MD first so overlapping LR markers sit on top, as in the paper
                ax.plot(sub.loc[est, "cos_with_md"], sub.loc[est, col], marker=markers[est], ms=5.2,
                        color=color, mec="white", mew=0.6, ls="none", zorder=2)
        ax.text(0.04, 0.12, f"$r = {r_vals[m]:.3f}$", color="0.35", fontsize=6.3, ha="left", va="bottom")
        ax.set_title(m)
        ax.set_xlim(0.0, 1.05)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylim(-1.0, 0.55)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Signed intervention effect, $\Delta P$")
    fig.supxlabel(r"Alignment with mean-difference direction, $\cos(v_{\mathrm{MD}}, v)$", fontsize=8.5, y=0.0)
    handles = [Line2D([], [], color=BLUE, lw=1.6, label=r"False: $\Delta P_F$"),
               Line2D([], [], color=VERMILION, lw=1.6, label=r"True: $\Delta P_T$")]
    handles += [Line2D([], [], color="0.3", marker=markers[e], ls="none", ms=5.5, label=labels[e]) for e in est_order]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.52, 1.12),
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2_estimator_paths")


# --------------------------------------------------------------------------
# Figure 3: directional specificity (a) and magnitude control (b)
# --------------------------------------------------------------------------
def figure3() -> None:
    ds_colors = {"TriviaQA": BLUE, "ARC-Easy": ORANGE, "CommonsenseQA": GREEN}
    qc = corrected_qwen7b_triviaqa()

    c4 = pd.read_csv(SEC / "C4_random_direction_null" / "c4_random_direction_summary.csv")
    c4["model"] = c4.model.map(norm_model)
    c4["side"] = c4.side.astype(str).map({"False": "false", "True": "true"})
    md = c4.md_delta_p.astype(float).copy()
    nullmax = c4.null_max.astype(float).copy()
    is_qtqa = (c4.model == "Qwen-7B") & (c4.dataset == "TriviaQA")
    for side in ("false", "true"):
        sel = is_qtqa & (c4.side == side)
        md[sel] = qc[f"fixed_{side}"]
        assert np.isclose(nullmax[sel].item(), qc[f"null_max_{side}"]), "Qwen-7B null must match corrected run"
    c4["md_abs"] = md.abs()
    c4["md_signed"] = md
    c4["null_max"] = nullmax

    expected_sign = np.where(c4.side == "false", c4.md_signed > 0, c4.md_signed < 0)
    assert expected_sign.all() and len(c4) == 30, "paper: all 30 sides have the expected sign"
    above = int((c4.md_abs > c4.null_max).sum())
    assert above == 29, above
    exc = c4[c4.md_abs <= c4.null_max]
    assert list(zip(exc.model, exc.dataset, exc.side)) == [("Mistral-7B", "TriviaQA", "false")]

    c3 = pd.read_csv(SEC / "C3_intervention" / "c3_full_intervention_summary.csv")
    c3["model"] = c3.model.map(norm_model)
    q = (c3.model == "Qwen-7B") & (c3.dataset == "TriviaQA")
    for side in ("false", "true"):
        c3.loc[q, f"fixed_delta_p_{side}"] = qc[f"fixed_{side}"]
        c3.loc[q, f"matched_delta_p_{side}"] = qc[f"matched_{side}"]
    assert np.allclose(c3.matched_achieved_r, 0.10)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.45), gridspec_kw={"width_ratios": [1, 1.08]})

    # (a) log-log specificity scatter
    lim = (3e-4, 1.15)
    ax.plot(lim, lim, color="0.4", lw=1.0, zorder=1)
    for _, rrow in c4.iterrows():
        ax.scatter(rrow.null_max, rrow.md_abs, s=22, color=ds_colors[rrow.dataset],
                   marker="o" if rrow.side == "false" else "^", edgecolor="white", linewidth=0.5, zorder=3)
    ex = exc.iloc[0]
    ax.scatter(ex.null_max, ex.md_abs, s=75, facecolor="none", edgecolor="#A9342A", linewidth=1.2, zorder=4)
    ax.annotate("Mistral F", xy=(ex.null_max, ex.md_abs), xytext=(0.48, 0.135), textcoords="data",
                fontsize=6.8, color="0.35", ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=0.7, shrinkA=0, shrinkB=4))
    ax.text(0.04, 0.95, f"{above}/{len(c4)} above equality", transform=ax.transAxes, color="0.35",
            fontsize=7, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.grid(which="major", color="0.88", lw=0.6)
    ax.grid(which="minor", color="0.95", lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"Largest random effect, $\max_j |\Delta P_j|$")
    ax.set_ylabel(r"MD effect, $|\Delta P_{\mathrm{MD}}|$")
    leg = [Line2D([], [], marker="o", color=ds_colors[d], ls="none", ms=4.5, label=d) for d in DATASETS]
    leg += [Line2D([], [], marker="o", color="0.45", ls="none", ms=4.5, label="False"),
            Line2D([], [], marker="^", color="0.45", ls="none", ms=4.5, label="True")]
    ax.legend(handles=leg, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2,
              labelspacing=0.25, fontsize=7)

    # (b) fixed alpha = 4 vs matched r = .10, cross-model ranges
    short = {"TriviaQA": "TriviaQA", "ARC-Easy": "ARC-Easy", "CommonsenseQA": "CSQA"}
    ticks, ticklabels, ranges = [], [], {}
    y = 0.0
    for di, d in enumerate(DATASETS):
        for side, tag in (("false", "F"), ("true", "T")):
            sub = c3[c3.dataset == d]
            fixed = np.abs(sub[f"fixed_delta_p_{side}"].astype(float).values)
            matched = np.abs(sub[f"matched_delta_p_{side}"].astype(float).values)
            ranges[(d, side)] = (np.ptp(fixed), np.ptp(matched))
            for vals, color, mk, dy in ((fixed, BLUE, "o", 0.0), (matched, VERMILION, "D", 0.24)):
                bx.plot([vals.min(), vals.max()], [y + dy, y + dy], color=color, lw=2.6, alpha=0.8,
                        solid_capstyle="round", zorder=1)
                bx.scatter(vals, np.full_like(vals, y + dy), s=20, marker=mk, color=color,
                           edgecolor="white", linewidth=0.5, zorder=2)
            ticks.append(y + 0.12)
            ticklabels.append(f"{short[d]}  {tag}")
            y += 1.0
        if di < len(DATASETS) - 1:
            bx.axhline(y - 0.38, color="0.85", lw=0.7)
    # Paper text: range decreased in 5/6 dataset-side comparisons, increased on CSQA true side.
    shrank = [k for k, (f, mt) in ranges.items() if mt < f]
    grew = [k for k, (f, mt) in ranges.items() if mt >= f]
    assert len(shrank) == 5 and grew == [("CommonsenseQA", "true")], ranges
    bx.set_yticks(ticks)
    bx.set_yticklabels(ticklabels)
    bx.set_ylim(y - 0.55, -0.45)
    bx.set_xlim(-0.015, 0.935)
    bx.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    bx.grid(axis="x", color="0.9", lw=0.6)
    bx.set_axisbelow(True)
    bx.tick_params(axis="y", length=3)
    bx.set_xlabel(r"Hypothesis-aligned effect, $|\Delta P|$")
    bx.legend(handles=[Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=4.5, label=r"Fixed $\alpha = 4$"),
                       Line2D([], [], color=VERMILION, lw=2.4, marker="D", ms=4.2, label=r"Matched $r = .10$")],
              loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.0, fontsize=6.8,
              bbox_to_anchor=(1.0, 1.04), labelspacing=0.3)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
    # Sub-captions sit below the x-axis labels (the max_j subscript needs the room).
    for axis, cap in ((ax, r"(a) Specificity at fixed $\alpha = 4$"),
                      (bx, r"(b) Fixed $\alpha = 4$ vs. matched $r = .10$")):
        box = axis.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.0, cap, ha="center", va="bottom", fontsize=8.5)
    save(fig, "fig3_specificity_and_magnitude")


# --------------------------------------------------------------------------
# Figure 4: excess differential quantization-error energy on TriviaQA
# --------------------------------------------------------------------------
def figure4() -> None:
    rinc = pd.read_csv(ART / "stats_package" / "rinc_global_375.csv")
    rinc = rinc[rinc.dataset == "TriviaQA"].copy()
    rinc["model"] = rinc.model.map(norm_model)
    rinc["excess"] = rinc.observed_R_inc - rinc.null_mean_R_inc
    qmap = {"AWQ W4A16": "AWQ", "GPTQ W4A16": "GPTQ", "RTN W4A16": "RTN",
            "simulated W8A8": "W8A8", "simulated W4A8": "W4A8"}
    rinc["q"] = rinc.quantization.map(qmap)
    assert len(rinc) == 125
    # Paper Sec. 6.2: on TriviaQA every model-quantization-dimension cell has positive excess
    # and is significant after the pooled 375-test BH-FDR correction.
    assert (rinc.excess > 0).all() and rinc.significant_q_lt_0_05.all()

    qstyle = {"AWQ": (BLUE, "o"), "GPTQ": (ORANGE, "s"), "RTN": (GREEN, "D"),
              "W8A8": (PINK, "^"), "W4A8": (VERMILION, "v")}
    ks = [2, 5, 10, 20, 50]
    fig, axes = plt.subplots(1, 5, figsize=(TEXTWIDTH_IN, 2.15), sharey=True)
    for i, (ax, m) in enumerate(zip(axes, MODELS)):
        sub = rinc[rinc.model == m]
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        for qn, (color, mk) in qstyle.items():
            s = sub[sub.q == qn].set_index("k").loc[ks]
            ax.plot(ks, s.excess, color=color, marker=mk, ms=4.2, lw=1.3, mec="white", mew=0.5, label=qn)
        ax.set_title(f"({'abcde'[i]}) {m}")
        ax.set_xticks(ks)
        ax.set_xticklabels(["2", "5", "10", "20", "50"])
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(-0.002, 0.075)
        ax.set_yticks([0.0, 0.02, 0.04, 0.06])
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Excess over activation-matched null")
    fig.supxlabel(r"Task-defined contrastive subspace dimension, $k$", fontsize=8.5, y=0.0)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.52, 1.12),
               handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_quant_error_excess")


def main() -> None:
    style()
    figure2()
    figure3()
    figure4()
    print(f"wrote figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
