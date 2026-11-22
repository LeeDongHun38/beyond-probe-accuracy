#!/usr/bin/env python3
"""Regenerate the numeric tables of EACL_2026.pdf from archived artifacts.

Reads only release-relative files (artifacts/ and data/) and writes one CSV per
paper table to paper/tables/ (or --out-dir).  Cells are formatted at the
precision printed in the paper ('.830', '+.4379', '2,500', ...) so that
verify/verify_paper_numbers.py can compare them cell by cell with the values
transcribed from the PDF in verify/expected/.

No statistic is re-estimated: tables are re-tabulations of stored run outputs.
Where the paper deliberately mixes runs (corrected Qwen2.5-7B TriviaQA rerun,
Mistral-7B explicit-system vs. legacy no-system runs), the source used for each
cell is the one the paper prints; see verify/known_discrepancies.csv.

Usage (from the repository root):
    python paper/make_tables.py [--out-dir paper/tables]
Dependencies: Python 3.9+ standard library only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "artifacts/paper_sections"
SRC = ROOT / "artifacts/sources/results"
STATS = ROOT / "artifacts/stats_package"

MODELS = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B", "Qwen2.5-14B", "Ministral-3-14B"]
DATASETS = ["TriviaQA", "ARC-Easy", "CommonsenseQA"]
QUANTS = ["awq", "gptq", "rtn", "w8a8", "w4a8"]
# Model names as they appear in the stats package (full HF names).
STATS_MODEL = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B", "Qwen2.5-14B-Instruct": "Qwen2.5-14B",
    "Ministral-3-14B-Instruct-2512": "Ministral-3-14B", "Ministral-3-14B-Instruct": "Ministral-3-14B",
}


# ------------------------------------------------------------------ formatting
def num(x: float, d: int) -> str:
    """Paper style: no leading zero, ASCII minus (e.g. .830, -.001, 112.94)."""
    s = f"{float(x):.{d}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    if s in ("-." + "0" * d,):  # avoid negative zero
        s = s[1:]
    return s


def sgn(x: float, d: int) -> str:
    s = num(x, d)
    return s if s.startswith("-") else "+" + s


def cos3(x: float) -> str:
    """Cosine alignment in Table 1: 3 decimals, but 1.000 printed as 1.00."""
    s = num(x, 3)
    return "1.00" if s == "1.000" else s


def thou(n) -> str:
    return f"{int(n):,}"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def write(out: Path, name: str, header: list[str], rows: list[list]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ shared sources
def estimator_rows() -> dict:
    """(model, estimator_key) -> row for the TriviaQA estimator sweep (Tables 1, 13, 14)."""
    out = {}
    dmap = {"D0": "MD", "D1": "raw_C1e-4", "D2": "raw_C1e-2", "D3": "raw_C1", "D4": "std_C1e-4", "D5": "std_C1"}
    for r in read_csv(SEC / "C2_estimator_sweep/d0_d5_canonical.csv"):
        out[(r["model"], dmap[r["direction_id"]])] = dict(r, dir=r["direction_id"])
    cmap = {"MD": "MD", "0.0001": "raw_C1e-4", "0.01": "raw_C1e-2", "1.0": "raw_C1"}
    nmap = {"qwen2.5_14b": "Qwen2.5-14B", "ministral3_14b": "Ministral-3-14B"}
    for r in read_csv(SEC / "C2_estimator_sweep/raw_lr_sweep_completed.csv"):
        if r["model"] in nmap and r["dataset"] == "TriviaQA":
            out[(nmap[r["model"]], cmap[r["C"]])] = r
    return out


def c1_rows() -> dict:
    """(model, dataset) -> C.1 split-half row; Ministral TriviaQA uses canonical L36."""
    dmap = {"triviaqa": "TriviaQA", "arc_easy": "ARC-Easy", "commonsenseqa": "CommonsenseQA"}
    out = {}
    for r in read_csv(SEC / "C1_md_stability/result.csv"):
        key = (r["model"].replace("Ministral-3-14B", "Ministral-3-14B"), dmap[r["dataset"]])
        if key == ("Ministral-3-14B", "TriviaQA") and r["layer"] != "36":
            continue  # L19 row is the Appendix C.3 sensitivity layer, not the canonical cell
        out[key] = r
    return out


def e1_rows() -> dict:
    return {(r["model"], r["dataset"], r["quantization"]): r
            for r in read_csv(SEC / "E1_cross_precision_transfer/e1_cross_precision_auroc.csv")}


def qwen_corrected() -> dict:
    """Corrected Qwen2.5-7B TriviaQA L18 rerun printed in Tables 15-17."""
    return load(SRC / "qwen7b_corrected_l18_c3c4_rerun_v1/analysis/summary.json")


def drop_fmt(x: float) -> str:
    x = float(x)
    if 0 <= x < 0.0005:
        return "<.001"
    return num(x, 3)


# ------------------------------------------------------------------ tables
def table01(out):
    est = estimator_rows()
    rows = []
    for e in ["MD", "raw_C1e-4", "raw_C1e-2", "raw_C1"]:
        for m in MODELS:
            r = est[(m, e)]
            rows.append([e, m, num(r["heldout_auroc"], 3), cos3(r["cos_with_md"]), num(r["split_mean"], 3)])
    write(out, "table01_triviaqa_estimator_sweep.csv", ["estimator", "model", "auc", "align", "stab"], rows)


def table02(out):
    e1 = e1_rows()
    rows = []
    for m in MODELS:
        row = [m, num(e1[(m, "TriviaQA", "awq")]["fp16_auroc"], 3)]
        for q in QUANTS:
            r = e1[(m, "TriviaQA", q)]
            row += [num(r["quantized_auroc"], 3), drop_fmt(r["auroc_drop"])]
        rows.append(row)
    hdr = ["model", "fp16"] + [c for q in QUANTS for c in (q, f"{q}_drop")]
    write(out, "table02_triviaqa_quant_transfer.csv", hdr, rows)


def table03(out):
    paths = {"TriviaQA": "data/pairs/triviaqa", "ARC-Easy": "data/pairs/arc_easy",
             "CommonsenseQA": "data/pairs/commonsenseqa"}
    rows = []
    for d, p in paths.items():
        n = {s: len(load(ROOT / p / f"{s}.json")) for s in ("all", "train", "test")}
        rows.append([d, thou(n["all"]), thou(n["train"]), thou(n["test"])])
    write(out, "table03_dataset_counts.csv", ["dataset", "total", "train", "heldout"], rows)


def table10(out):
    c1 = c1_rows()
    rows = [[m] + [f"L{c1[(m, d)]['layer']}" for d in DATASETS] for m in MODELS]
    write(out, "table10_selected_layers.csv", ["model"] + DATASETS, rows)


def table12(out):
    c1 = c1_rows()
    rows = [[m, d, num(c1[(m, d)]["recomputed_heldout_md_auroc"], 3), num(c1[(m, d)]["split_half_mean"], 3)]
            for m in MODELS for d in DATASETS]
    write(out, "table12_md_auroc_stability.csv", ["model", "dataset", "auroc", "split"], rows)


def _est_row(r):
    return [num(r["heldout_auroc"], 3), num(r["cos_with_md"], 3), num(r["split_mean"], 3),
            num(r["participation_ratio"], 1), sgn(r["delta_p_false"], 4), sgn(r["delta_p_true"], 4)]


def table13(out):
    rows = []
    for r in read_csv(SEC / "C2_estimator_sweep/d0_d5_canonical.csv"):
        rows.append([r["model"], r["direction_id"]] + _est_row(r))
    order = {m: i for i, m in enumerate(MODELS)}
    rows.sort(key=lambda x: (order[x[0]], x[1]))
    write(out, "table13_estimator_sensitivity_7_8b.csv",
          ["model", "dir", "auroc", "align", "split", "pr", "dpf", "dpt"], rows)


def table14(out):
    est = estimator_rows()
    rows = [[m, e] + _est_row(est[(m, e)]) for m in ("Qwen2.5-14B", "Ministral-3-14B")
            for e in ("MD", "raw_C1e-4", "raw_C1e-2", "raw_C1")]
    write(out, "table14_estimator_sensitivity_14b.csv",
          ["model", "estimator", "auroc", "align", "split", "pr", "dpf", "dpt"], rows)


def _c3():
    return {(r["model"], r["dataset"]): r for r in read_csv(SEC / "C3_intervention/c3_full_intervention_summary.csv")}


def table15(out):
    c3, qc = _c3(), qwen_corrected()
    q_d0 = [x for x in qc["job1_d0_d5"] if x["direction_id"] == "D0"][0]
    rows = []
    for d in DATASETS:
        for m in MODELS:
            r = c3[(m, d)]
            dpf, dpt, dmf, dmt = (r["fixed_delta_p_false"], r["fixed_delta_p_true"],
                                  r["fixed_delta_m_false"], r["fixed_delta_m_true"])
            if (m, d) == ("Qwen2.5-7B", "TriviaQA"):  # corrected rerun (paper value)
                dpf, dpt, dmf, dmt = (q_d0["delta_p_false"], q_d0["delta_p_true"],
                                      q_d0["delta_logit_false"], q_d0["delta_logit_true"])
            rows.append([d, m, r["layer"], num(r["median_hidden_norm"], 2), num(r["fixed_r"], 3),
                         num(r["baseline_p_false"], 3), num(r["baseline_p_true"], 3),
                         sgn(dpf, 3), sgn(dpt, 3), sgn(dmf, 2), sgn(dmt, 2)])
    write(out, "table15_md_intervention_alpha4.csv",
          ["dataset", "model", "layer", "median_norm", "r", "base_pf", "base_pt", "dpf", "dpt", "dmf", "dmt"], rows)


def table16(out):
    c3, qc = _c3(), qwen_corrected()
    q_r10 = [x for x in qc["job2_md_by_r"] if abs(x["r_target"] - 0.1) < 1e-12][0]
    rows = []
    for d in DATASETS:
        for m in MODELS:
            r = c3[(m, d)]
            vals = (r["matched_alpha"], r["matched_delta_p_false"], r["matched_delta_p_true"],
                    r["matched_delta_m_false"], r["matched_delta_m_true"])
            if (m, d) == ("Qwen2.5-7B", "TriviaQA"):
                vals = (q_r10["alpha"], q_r10["delta_p_false"], q_r10["delta_p_true"],
                        q_r10["delta_logit_false"], q_r10["delta_logit_true"])
            a, pf, pt, mf, mt = vals
            rows.append([d, m, r["layer"], num(a, 3), sgn(pf, 3), sgn(pt, 3), sgn(mf, 2), sgn(mt, 2)])
    write(out, "table16_md_intervention_matched_r010.csv",
          ["dataset", "model", "layer", "alpha", "dpf", "dpt", "dmf", "dmt"], rows)


def table17(out):
    qc = qwen_corrected()
    qn = {x["side"]: x for x in qc["job3_null_distribution"]}
    cells = {}
    for r in read_csv(SEC / "C4_random_direction_null/c4_random_direction_summary.csv"):
        cells[(r["model"], r["dataset"], r["side"])] = r
    rows = []
    for d in DATASETS:
        for m in MODELS:
            f, t = cells[(m, d, "false")], cells[(m, d, "true")]
            tmd = [abs(float(f["md_delta_p"])), abs(float(t["md_delta_p"]))]
            med = [f["null_median"], t["null_median"]]
            mx = [f["null_max"], t["null_max"]]
            exc = [f["exceedance_count"], t["exceedance_count"]]
            if (m, d) == ("Qwen2.5-7B", "TriviaQA"):
                tmd = [qn["false"]["T_probe"], qn["true"]["T_probe"]]
                med = [qn["false"]["random_T_median"], qn["true"]["random_T_median"]]
                mx = [qn["false"]["random_T_max"], qn["true"]["random_T_max"]]
                exc = [qn["false"]["random_ge_probe_count"], qn["true"]["random_ge_probe_count"]]
            rows.append([d, m, f["layer"].lstrip("L"), num(f["normalized_magnitude"], 3),
                         num(tmd[0], 3), num(tmd[1], 3), num(med[0], 4), num(med[1], 4),
                         num(mx[0], 4), num(mx[1], 4), int(exc[0]), int(exc[1])])
    write(out, "table17_random_direction_alpha4.csv",
          ["dataset", "model", "layer", "r", "tmd_f", "tmd_t", "null_med_f", "null_med_t",
           "null_max_f", "null_max_t", "exc_f", "exc_t"], rows)


def table18(out):
    cells = {}
    for r in read_csv(SEC / "C4_random_direction_null/c4_norm_matched_random_summary.csv"):
        cells[(r["model"], r["side"])] = r
    qn = {x["side"]: x for x in qwen_corrected()["job3_null_distribution"]}
    rows = []
    for m in ("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B", "Qwen2.5-14B"):
        f, t = cells[(m, "false")], cells[(m, "true")]
        tmd = [abs(float(f["md_delta_p"])), abs(float(t["md_delta_p"]))]
        med, mx = [f["null_median"], t["null_median"]], [f["null_max"], t["null_max"]]
        exc = [f["exceedance_count"], t["exceedance_count"]]
        if m == "Qwen2.5-7B":
            # Qwen2.5-7B already has r=.051 at alpha=4, so the paper reuses its corrected
            # alpha=4 row (identical to Table 17); the C4 CSV row carries the legacy MD effect.
            tmd = [qn["false"]["T_probe"], qn["true"]["T_probe"]]
            med = [qn["false"]["random_T_median"], qn["true"]["random_T_median"]]
            mx = [qn["false"]["random_T_max"], qn["true"]["random_T_max"]]
            exc = [qn["false"]["random_ge_probe_count"], qn["true"]["random_ge_probe_count"]]
        rows.append([m, f["layer"].lstrip("L"), num(f["normalized_magnitude"], 3), num(f["alpha"], 3),
                     num(tmd[0], 3), num(tmd[1], 3), num(med[0], 4), num(med[1], 4),
                     num(mx[0], 4), num(mx[1], 4), int(exc[0]), int(exc[1])])
    write(out, "table18_random_direction_r005.csv",
          ["model", "layer", "r", "alpha", "tmd_f", "tmd_t", "null_med_f", "null_med_t",
           "null_max_f", "null_max_t", "exc_f", "exc_t"], rows)


def table19(out):
    side = {"false": "F", "true": "T"}
    rows = []
    for r in read_csv(SEC / "C4_random_direction_null/c4_paired_side_tests.csv"):
        nf, nt = float(r["null_magnitude_false"]), float(r["null_magnitude_true"])
        rows.append([r["model"], r["dataset"], num(r["normalized_magnitude"], 3),
                     num(r["baseline_p_false"], 3), num(r["baseline_p_true"], 3),
                     side.get(r["side_closer_to_half"], r["side_closer_to_half"]), num(nf, 4), num(nt, 4),
                     side.get(r["side_with_larger_null"], r["side_with_larger_null"]),
                     f"{max(nf, nt) / min(nf, nt):.2f}"])
    write(out, "table19_paired_side_comparison.csv",
          ["model", "dataset", "r", "base_pf", "base_pt", "closer", "n_f", "n_t", "larger", "ratio"], rows)


def table20(out):
    rows = []
    for r in read_csv(SEC / "D1_template_length_control/d1_template_length_summary.csv"):
        rows.append([r["model"], r["dataset"], thou(r["n_heldout_pairs"]), thou(r["n_small"]), thou(r["n_large"]),
                     num(r["g_temp"], 4), num(r["g_len"], 4)])
    write(out, "table20_template_length_control.csv",
          ["model", "dataset", "n", "small", "large", "g_temp", "g_len"], rows)


def table21(out):
    sc = load(SRC / "source_matched_control/qwen_answer_only_style_classifier.json")
    rows = [["answer_only_style", "Gold vs. incorrect answer", "-",
             num(sc["canonical_gold_vs_generated"]["test_auroc"], 4),
             num(sc["source_matched_gold_vs_gold"]["test_auroc"], 4)]]
    d2 = {r["model"]: r for r in read_csv(SEC / "D2_source_style_control/d2_source_style_summary.csv")}
    for m in ("Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"):
        r = d2[m]
        rows.append(["hidden_state_transfer", m, r["layer"], num(r["original_auroc"], 4),
                     num(r["gold_donor_transfer_auroc"], 4)])
    write(out, "table21_source_style_control.csv", ["analysis", "model", "layer", "original", "gold_donor"], rows)


def table22(out):
    rows = []
    for r in read_csv(SEC / "D3_owngen_readout/d3_owngen_last_token_summary.csv"):
        if r["dataset"] != "TriviaQA":
            continue  # the ARC-Easy row is reported in text (D.3), not in Table 22
        rows.append([r["model"], r["layer"], thou(r["scored_answers"]), thou(r["correct_answers"]),
                     thou(r["incorrect_answers"]), thou(r["empty_or_excluded"]), num(r["last_token_auroc"], 4),
                     num(r["ci95_low"], 4), num(r["ci95_high"], 4), num(r["random_p"], 4)])
    write(out, "table22_owngen_last_token.csv",
          ["model", "layer", "scored", "correct", "incorrect", "excl", "auroc", "ci_low", "ci_high", "random_p"], rows)


def table23(out):
    rows = []
    for r in read_csv(SEC / "D4_pre_answer_readout/d4_pre_answer_summary.csv"):
        has_null = r["random_direction_count"] not in ("", "0", "NA", "None", "not run")
        exc = r["random_exceedance_count"]
        rows.append([r["model"], r["dataset"], r["layer"], thou(r["outcome_valid_questions"]), thou(r["n_questions"]),
                     r["generations_per_question"], num(r["spearman_rho"], 4), num(r["ci95_low"], 4),
                     num(r["ci95_high"], 4), exc if has_null else "-",
                     num(r["random_p"], 4) if has_null else "-"])
    write(out, "table23_pre_answer_readout.csv",
          ["model", "dataset", "layer", "valid", "total", "gen_per_question", "rho", "ci_low", "ci_high",
           "random_exc", "random_p"], rows)


def table24(out):
    rows = []
    for r in read_csv(SEC / "D5_answer_span_aggregation/d5_answer_span_summary.csv"):
        rows.append([r["model"], r["layer"], thou(r["scored_answers"]), num(r["last_token_auroc"], 4),
                     num(r["last_ci95_low"], 4), num(r["last_ci95_high"], 4), num(r["mean_pooled_auroc"], 4),
                     num(r["mean_ci95_low"], 4), num(r["mean_ci95_high"], 4)])
    write(out, "table24_answer_span_aggregation.csv",
          ["model", "layer", "scored", "last_auroc", "last_ci_low", "last_ci_high", "mean_auroc",
           "mean_ci_low", "mean_ci_high"], rows)


def table25(out):
    e1 = e1_rows()
    rows = []
    for m in MODELS:
        for d in DATASETS:
            first = e1[(m, d, "awq")]
            row = [m, d, f"L{first['layer']}", num(first["fp16_auroc"], 3)]
            for q in QUANTS:
                r = e1[(m, d, q)]
                row += [num(r["quantized_auroc"], 3), drop_fmt(r["auroc_drop"])]
            rows.append(row)
    hdr = ["model", "dataset", "layer", "fp16"] + [c for q in QUANTS for c in (q, f"{q}_drop")]
    write(out, "table25_cross_precision_transfer.csv", hdr, rows)


def table26(out):
    rows_in = read_csv(STATS / "rinc_global_375.csv")
    scopes = [("7-8B models", {"Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"}),
              ("Qwen2.5-14B", {"Qwen2.5-14B"}), ("Ministral-3-14B", {"Ministral-3-14B"})]
    rows = []
    for name, ms in scopes:
        for d in DATASETS:
            sub = [r for r in rows_in if STATS_MODEL[r["model"]] in ms and r["dataset"] == d]
            sig = lambda rs: sum(r["significant_q_lt_0_05"] == "True" for r in rs)
            k2 = [r for r in sub if r["k"] == "2"]
            k5 = [r for r in sub if r["k"] != "2"]
            rows.append([name, d, f"{sig(k2)}/{len(k2)}", f"{sig(k5)}/{len(k5)}", f"{sig(sub)}/{len(sub)}"])
    write(out, "table26_rinc_bh_significance.csv", ["scope", "dataset", "k2", "k_ge5", "all"], rows)


def table27(out):
    rows = []
    order = {m: i for i, m in enumerate(MODELS)}
    for r in read_csv(SEC / "E2_quant_error_geometry/e2_geometry_k50_summary.csv"):
        rows.append([r["model"], r["dataset"], r["quantization"], num(r["R_1"], 3), num(r["R_50"], 3),
                     num(r["null_mean_R_incremental_50"], 3)])
    rows.sort(key=lambda x: (order[x[0]], DATASETS.index(x[1]), QUANTS.index(x[2])))
    write(out, "table27_quant_geometry_k50.csv", ["model", "dataset", "quantization", "r1", "r50", "n50"], rows)


def table28(out):
    m_rows = read_csv(STATS / "m_global_375.csv")
    a_rows = read_csv(STATS / "a_global_375.csv")
    margins = read_csv(SEC / "E3_mean_error_and_margin/e3_margin_flip_summary.csv")
    rows = []
    for m in MODELS:
        for d in DATASETS:
            sub = [r for r in m_rows if STATS_MODEL[r["model"]] == m and r["dataset"] == d]
            sub_a = [r for r in a_rows if STATS_MODEL[r["model"]] == m and r["dataset"] == d]
            sig = lambda rs: sum(r["significant_q_lt_0_05"] == "True" for r in rs)
            # M and A must give identical decisions (A = M / fixed positive constant).
            key = lambda r: (r["quantization"], r["k"], r["significant_q_lt_0_05"])
            assert sorted(map(key, sub)) == sorted(map(key, sub_a)), (m, d)
            k2 = [r for r in sub if r["k"] == "2"]
            k5 = [r for r in sub if r["k"] != "2"]
            mg = [r for r in margins if r["model"] == m and r["dataset"] == d]
            shr = [100 * float(r["relative_shrinkage"]) for r in mg]
            flp = [100 * float(r["margin_flip_rate"]) for r in mg]
            rows.append([m, d, mg[0]["layer"], f"{sig(k2)}/{len(k2)}", f"{sig(k5)}/{len(k5)}",
                         f"{sig(sub)}/{len(sub)}", f"{min(shr):.1f}", f"{max(shr):.1f}",
                         f"{min(flp):.1f}", f"{max(flp):.1f}"])
    write(out, "table28_mean_error_margin.csv",
          ["model", "dataset", "layer", "sig_k2", "sig_k_ge5", "sig_all", "shrink_min_pct", "shrink_max_pct",
           "flip_min_pct", "flip_max_pct"], rows)


TABLES = [table01, table02, table03, table10, table12, table13, table14, table15, table16, table17,
          table18, table19, table20, table21, table22, table23, table24, table25, table26, table27, table28]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "paper/tables")
    a = ap.parse_args()
    for fn in TABLES:
        fn(a.out_dir)
    print(f"wrote {len(TABLES)} tables to {a.out_dir}")


if __name__ == "__main__":
    main()
