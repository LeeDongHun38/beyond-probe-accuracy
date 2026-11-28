#!/usr/bin/env python3
"""Regenerate Figures 2-4 of the EACL 2027 paper from released artifacts.

Run from the repository root:

    python paper/make_figures.py            # writes paper/figures/fig{2,3,4}.{pdf,png}

Requires only numpy, pandas and matplotlib. No model inference is performed.
Every value plotted is read from ``artifacts/``; annotations that are printed
in the paper (r values, "29/30 above equality") are recomputed and asserted.

Figure 1 (question-matched pair example) is a text diagram typeset in LaTeX
and is not regenerated here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
from make_tables import fixed_null, geometry_panel
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
    "Qwen2.5-14B": "Qwen-14B", "Qwen2.5-14B-Instruct": "Qwen-14B", "qwen2.5_14b": "Qwen-14B", "qwen14": "Qwen-14B", "qwen14b": "Qwen-14B",
    "Ministral-3-14B": "Ministral-14B", "Ministral-3-14B-Instruct": "Ministral-14B",
    "ministral3_14b": "Ministral-14B", "ministral_3_14b": "Ministral-14B", "ministral14": "Ministral-14B", "ministral14b": "Ministral-14B",
}


def norm_model(name: str) -> str:
    return MODEL_ALIASES[name]


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "mathtext.fontset": "dejavusans",
    })


def save(fig: plt.Figure, stem: str, out_dir: Path = OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", metadata={"CreationDate": None})  # deterministic output
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: estimator-induced direction paths on TriviaQA at alpha = 4
# --------------------------------------------------------------------------
def figure2(out_dir: Path = OUT) -> None:
    est_order = ["MD", "1e-4", "1e-2", "1"]
    markers = {"MD": "o", "1e-4": "s", "1e-2": "D", "1": "^"}
    labels = {"MD": "MD", "1e-4": r"Raw $C=10^{-4}$", "1e-2": r"Raw $C=10^{-2}$", "1": r"Raw $C=1$"}

    points = pd.read_csv(ART / "figure_inputs/figure2_input.csv", dtype={"side": str})
    points["model"] = points.model.map(norm_model)
    points["est"] = points.direction.map(dict(zip(["D0", "D1", "D2", "D3"], est_order)))
    r_vals = points.groupby("model").r.first()
    rows = points.pivot(index=["model", "est", "alignment"], columns="side", values="delta_p")
    rows = rows.reset_index().rename(columns={"alignment": "cos_with_md", "false": "delta_p_false",
                                             "true": "delta_p_true"})

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
    save(fig, "fig2_estimator_paths", out_dir)


# --------------------------------------------------------------------------
# Figure 3: directional specificity (a) and magnitude control (b)
# --------------------------------------------------------------------------
def figure3(out_dir: Path = OUT) -> None:
    ds_colors = {"TriviaQA": BLUE, "ARC-Easy": ORANGE, "CommonsenseQA": GREEN}
    c4 = fixed_null().rename(columns={"md_effect": "md_signed"})
    c4["md_abs"] = c4.md_signed.abs()
    c4["model"] = c4.model.map(norm_model)
    c4["dataset"] = c4.dataset.map({"triviaqa": "TriviaQA", "arc_easy": "ARC-Easy", "commonsenseqa": "CommonsenseQA"})
    expected_sign = np.where(c4.side == "false", c4.md_signed > 0, c4.md_signed < 0)
    assert expected_sign.all() and len(c4) == 30, "paper: all 30 sides have the expected sign"
    above = int((c4.md_abs > c4.null_max).sum())
    assert above == 29, above
    exc = c4[c4.md_abs <= c4.null_max]
    assert list(zip(exc.model, exc.dataset, exc.side)) == [("Mistral-7B", "TriviaQA", "false")]

    c3 = pd.read_csv(ART / "figure_inputs/figure3b_input.csv", dtype={"side": str})

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
            sub = c3[(c3.dataset == d) & (c3.side == side)]
            fixed = sub[sub.dose == "alpha4"].delta_p.abs().to_numpy()
            matched = sub[sub.dose == "r010"].delta_p.abs().to_numpy()
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
    save(fig, "fig3_specificity_and_magnitude", out_dir)


# --------------------------------------------------------------------------
# Figure 4: enrichment relative to the pair-label permutation null.
def figure4(out_dir: Path = OUT) -> None:
    rows = geometry_panel()
    panels = [("qwen14b", "triviaqa"), ("qwen14b", "arc_easy"), ("qwen14b", "commonsenseqa"),
              ("ministral14b", "arc_easy"), ("ministral14b", "commonsenseqa")]
    styles = [("awq", BLUE, "o"), ("gptq", ORANGE, "s"), ("rtn", GREEN, "^"),
              ("w8a8", VERMILION, "D"), ("w4a8", PINK, "v")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.25), sharey=True)
    for ax, statistic in zip(axes, ("R1", "R50")):
        indexed = rows[rows.statistic == statistic].set_index(["model", "dataset", "quantization"])
        for j, (quant, color, marker) in enumerate(styles):
            for i, (model, dataset) in enumerate(panels):
                row = indexed.loc[model, dataset, quant]
                ax.scatter(i + (j - 2) * .13, row.observed / row.null_mean, marker=marker, s=20,
                           edgecolors=color, facecolors=color if row.q_bh_family < .05 else "white",
                           label=quant.upper() if i == 0 else None)
        ax.axhline(1, color="0.5", ls="--", lw=.8)
        ax.set_xticks(range(5), ["TQA", "ARC-E", "CSQA", "ARC-E", "CSQA"])
        ax.set_xlabel("Qwen2.5-14B           Ministral-3-14B")
        ax.set_title(statistic)
    axes[0].set_ylim(0, 10.5)
    axes[0].set_ylabel("Observed / null mean")
    h, labels = axes[0].get_legend_handles_labels()
    fig.legend(h, labels, ncol=5, loc="upper center", frameon=False)
    fig.text(.98, .01, "Hollow markers indicate BH q ≥ .05", ha="right", fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, .84))
    save(fig, "fig4_quant_error_excess", out_dir)


def main(out_dir: Path = OUT) -> None:
    style()
    figure2(out_dir)
    figure3(out_dir)
    figure4(out_dir)
    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Figures 2–4 from the released experiment outputs.")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    main(parser.parse_args().out_dir)
