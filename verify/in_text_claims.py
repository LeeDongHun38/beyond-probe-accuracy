"""Calculators for verify/expected/in_text_claims.csv.

Each function recomputes one number quoted in the paper text from release
artifacts and returns it formatted like the paper.  Called by
verify/verify_paper_numbers.py; standard library only.
"""
from __future__ import annotations

import csv
import json
import statistics
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "artifacts/paper_sections"
SRC = ROOT / "artifacts/sources/results"
STATS = ROOT / "artifacts/stats_package"
DATASETS = ["TriviaQA", "ARC-Easy", "CommonsenseQA"]


def _read(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x: float, d: int) -> str:
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def _steer():
    return _read(STATS / "steering_fixed_alpha_global_30.csv")


def _c3():
    rows = {(r["model"], r["dataset"]): dict(r) for r in _read(SEC / "C3_intervention/c3_full_intervention_summary.csv")}
    q = json.loads((SRC / "qwen7b_corrected_l18_c3c4_rerun_v1/analysis/summary.json").read_text())
    d0 = [x for x in q["job1_d0_d5"] if x["direction_id"] == "D0"][0]
    r10 = [x for x in q["job2_md_by_r"] if abs(x["r_target"] - 0.1) < 1e-12][0]
    rows[("Qwen2.5-7B", "TriviaQA")].update(fixed_delta_p_false=d0["delta_p_false"], fixed_delta_p_true=d0["delta_p_true"],
                                            matched_delta_p_false=r10["delta_p_false"],
                                            matched_delta_p_true=r10["delta_p_true"])
    return rows


def _e1():
    return _read(SEC / "E1_cross_precision_transfer/e1_cross_precision_auroc.csv")


def _c1():
    return [r for r in _read(SEC / "C1_md_stability/result.csv")
            if not (r["model"] == "Ministral-3-14B" and r["dataset"] == "triviaqa" and r["layer"] != "36")]


# ---------------------------------------------------------------- Section 5 / App C
def md_exceeds_null_max():
    s = _steer()
    return f"{sum(abs(float(r['md_delta_p'])) > float(r['null_max_abs_effect']) for r in s)}/{len(s)}"


def expected_sign_alpha4():
    s = _steer()
    ok = sum((float(r["md_delta_p"]) > 0) == (r["side"] == "false") for r in s)
    return f"{ok}/{len(s)}"


def fixed_alpha_bh_significant():
    s = _steer()
    return f"{sum(r['significant_q_lt_0_05'] == 'True' for r in s)}/{len(s)}"


def mistral_tqa_false_p():
    r = [r for r in _steer() if r["model"].startswith("Mistral") and r["dataset"] == "TriviaQA" and r["side"] == "false"][0]
    return _f(r["p_mc"], 6)


def r_ratio_range():
    """Per-dataset max/min r at alpha=4.  For Mistral-7B TriviaQA the paper's text uses the
    explicit-system run (r=.845, as in Figure 2), not the legacy r=.855 of Table 15."""
    c3 = _c3()
    norms = json.loads((SRC / "gpu_queue_explicit_system_v1/job1_e8_with/analysis/with_residual_norm_medians.json").read_text())
    ratios = []
    for d in DATASETS:
        rs = {m: float(v["fixed_r"]) for (m, dd), v in c3.items() if dd == d}
        if d == "TriviaQA":
            rs["Mistral-7B"] = 4.0 / norms["mistral"]["pooled_median_residual_norm"]
        ratios.append(max(rs.values()) / min(rs.values()))
    return f"{min(ratios):.1f}-{max(ratios):.1f}"


def matched_r_range_shrinks():
    c3 = _c3()
    n = shrink = 0
    for d in DATASETS:
        for side in ("false", "true"):
            fixed = [abs(float(v[f"fixed_delta_p_{side}"])) for (m, dd), v in c3.items() if dd == d]
            matched = [abs(float(v[f"matched_delta_p_{side}"])) for (m, dd), v in c3.items() if dd == d]
            n += 1
            shrink += (max(matched) - min(matched)) < (max(fixed) - min(fixed))
    return f"{shrink}/{n}"


def r005_zero_exceedances():
    rows = _read(SEC / "C4_random_direction_null/c4_norm_matched_random_summary.csv")
    return f"{sum(int(r['exceedance_count']) == 0 for r in rows)}/{len(rows)}"


def md_auroc_range():
    a = [float(r["recomputed_heldout_md_auroc"]) for r in _c1()]
    return f"{_f(min(a), 3)}-{_f(max(a), 3)}"


def md_split_range():
    a = [float(r["split_half_mean"]) for r in _c1()]
    return f"{_f(min(a), 3)}-{_f(max(a), 3)}"


def paired_side_sign_test():
    rows = _read(SEC / "C4_random_direction_null/c4_paired_side_tests.csv")
    n, k = len(rows), sum(r["comparison_correct"] == "True" for r in rows)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
    return f"{k}/{n}/{_f(p, 4)}"


def paired_side_median_ratio():
    rows = _read(SEC / "C4_random_direction_null/c4_paired_side_tests.csv")
    ratios = []
    for r in rows:
        a, b = float(r["null_magnitude_false"]), float(r["null_magnitude_true"])
        ratios.append(max(a, b) / min(a, b))
    return f"{statistics.median(ratios):.2f}"


def fig2_r_values():
    norms = json.loads((SRC / "gpu_queue_explicit_system_v1/job1_e8_with/analysis/with_residual_norm_medians.json").read_text())
    c3 = _c3()
    vals = [4.0 / norms[k]["pooled_median_residual_norm"] for k in ("llama", "qwen", "mistral")]
    vals += [float(c3[(m, "TriviaQA")]["fixed_r"]) for m in ("Qwen2.5-14B", "Ministral-3-14B")]
    return "/".join(_f(v, 3) for v in vals)


# ---------------------------------------------------------------- App D
def _d1():
    return _read(SEC / "D1_template_length_control/d1_template_length_summary.csv")


def d1_max_template_range():
    return _f(max(float(r["g_temp"]) for r in _d1()), 4)


def d1_max_length_gap():
    return _f(max(float(r["g_len"]) for r in _d1()), 4)


def d3_auroc_range():
    a = [float(r["last_token_auroc"]) for r in _read(SEC / "D3_owngen_readout/d3_owngen_last_token_summary.csv")
         if r["dataset"] == "TriviaQA"]
    return f"{_f(min(a), 4)}-{_f(max(a), 4)}"


def _d5(col):
    a = [float(r[col]) for r in _read(SEC / "D5_answer_span_aggregation/d5_answer_span_summary.csv")]
    return f"{_f(min(a), 4)}-{_f(max(a), 4)}"


def d5_last_range():
    return _d5("last_token_auroc")


def d5_mean_range():
    return _d5("mean_pooled_auroc")


def _d6(split, cond, acc_d, p_d):
    r = [r for r in _read(SEC / "D6_direct_generation/d5_direct_generation_summary.csv")
         if r["split"] == split and r["condition"] == cond][0]
    p = float(r["mcnemar_exact_two_sided_p"])
    ps = "1.0" if p == 1.0 else _f(p, p_d)
    return (f"{_f(r['baseline_accuracy'], acc_d)}->{_f(r['condition_accuracy'], acc_d)}/"
            f"{r['baseline_wrong_condition_correct']}/{r['baseline_correct_condition_wrong']}/{ps}")


def d6_initial():
    return _d6("initial_200", "plus_v", 3, 2)


def d6_confirm_pos():
    return _d6("confirmation_400", "plus_v", 4, 2)


def d6_confirm_neg():
    return _d6("confirmation_400", "minus_v", 4, 2)


# ---------------------------------------------------------------- App E
def max_auroc_drop():
    return _f(max(float(r["auroc_drop"]) for r in _e1()), 3)


def max_auroc_drop_cell():
    r = max(_e1(), key=lambda r: float(r["auroc_drop"]))
    return f"{r['model']}/{r['dataset']}/{r['quantization']}"


def w8a8_mean_drop():
    a = [float(r["auroc_drop"]) for r in _e1() if r["quantization"] == "w8a8"]
    return _f(sum(a) / len(a), 4)


def max_drop_qwen14b():
    return _f(max(float(r["auroc_drop"]) for r in _e1() if r["model"] == "Qwen2.5-14B"), 3)


def max_drop_ministral14b():
    return _f(max(float(r["auroc_drop"]) for r in _e1() if r["model"] == "Ministral-3-14B"), 3)


def negative_drops_llama():
    neg = [r for r in _e1() if float(r["auroc_drop"]) < 0]
    assert all(r["model"] == "Llama-3.1-8B" and r["quantization"] == "w8a8" for r in neg), neg
    return str(len(neg))


def _sig(rows):
    return sum(r["significant_q_lt_0_05"] == "True" for r in rows)


def rinc_significant():
    r = _read(STATS / "rinc_global_375.csv")
    return f"{_sig(r)}/{len(r)}"


def rinc_nonsig_cells():
    r = [x for x in _read(STATS / "rinc_global_375.csv") if x["significant_q_lt_0_05"] != "True"]
    if not all(x["dataset"] == "CommonsenseQA" and x["k"] == "2" and x["model"].startswith("Qwen") for x in r):
        return "not all Qwen CommonsenseQA k=2"
    out = {}
    for x in r:
        m = "Qwen2.5-7B" if x["model"].startswith("Qwen2.5-7B") else "Qwen2.5-14B"
        out[m] = out.get(m, 0) + 1
    return ";".join(f"{k}:{out[k]}" for k in ("Qwen2.5-7B", "Qwen2.5-14B") if k in out)


def rinc_k_ge5():
    r = [x for x in _read(STATS / "rinc_global_375.csv") if x["k"] != "2"]
    return f"{_sig(r)}/{len(r)}"


def _rinc_cell(quant):
    r = [x for x in _read(STATS / "rinc_global_375.csv") if x["model"].startswith("Qwen2.5-7B")
         and x["dataset"] == "CommonsenseQA" and x["k"] == "2" and x["quantization"] == quant][0]
    return (f"{_f(r['observed_R_inc'], 6)}/{_f(r['null_mean_R_inc'], 6)}/"
            f"{_f(r['p_raw'], 5)}/{_f(r['q_bh_global_375'], 5)}")


def rinc_qwen7_csqa_gptq_k2():
    return _rinc_cell("GPTQ W4A16")


def rinc_qwen7_csqa_w8a8_k2():
    return _rinc_cell("simulated W8A8")


def m_significant():
    r = _read(STATS / "m_global_375.csv")
    return f"{_sig(r)}/{len(r)}"


def m_k2():
    r = [x for x in _read(STATS / "m_global_375.csv") if x["k"] == "2"]
    return f"{_sig(r)}/{len(r)}"


def m_k_ge5():
    r = [x for x in _read(STATS / "m_global_375.csv") if x["k"] != "2"]
    return f"{_sig(r)}/{len(r)}"


def ministral_tqa_m():
    r = [x for x in _read(STATS / "m_global_375.csv") if x["model"].startswith("Ministral") and x["dataset"] == "TriviaQA"]
    return f"{_sig(r)}/{len(r)}"


def ministral_tqa_awq_k5():
    r = [x for x in _read(STATS / "m_global_375.csv") if x["model"].startswith("Ministral")
         and x["dataset"] == "TriviaQA" and x["quantization"] == "AWQ W4A16" and x["k"] == "5"][0]
    return f"{_f(r['p_raw'], 4)}/{_f(r['q_bh_global_375'], 4)}"


def _margins():
    return _read(SEC / "E3_mean_error_and_margin/e3_margin_flip_summary.csv")


def margin_contracts():
    m = _margins()
    return f"{sum(float(r['mean_margin_change']) < 0 for r in m)}/{len(m)}"


def shrinkage_range():
    a = [100 * float(r["relative_shrinkage"]) for r in _margins()]
    return f"{min(a):.1f}-{max(a):.1f}"


def flip_range():
    a = [100 * float(r["margin_flip_rate"]) for r in _margins()]
    return f"{min(a):.1f}-{max(a):.1f}"


def flip_max_cell():
    m = _margins()
    a = max(m, key=lambda r: float(r["relative_shrinkage"]))
    b = max(m, key=lambda r: float(r["margin_flip_rate"]))
    if (a["model"], a["dataset"], a["quantization"]) != (b["model"], b["dataset"], b["quantization"]):
        return "maxima in different cells"
    return f"{a['model']}/{a['dataset']}/{a['quantization']}"
