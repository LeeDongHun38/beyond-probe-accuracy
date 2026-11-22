#!/usr/bin/env python3
"""Build the derived paper-section summaries that have no single upstream CSV.

Reads only release-relative files under artifacts/ and writes:

* D1_template_length_control/d1_template_length_summary.csv   (Table 20)
* D5_answer_span_aggregation/d5_answer_span_summary.csv         (Table 24)
* E1_cross_precision_transfer/e1_cross_precision_auroc.csv      (Tables 2, 25)

No values are re-estimated here: every number is read from an archived run
output (see the ``source`` column of each CSV). The only computation is the
Small/Large group counts of Table 20, which follow the original rule in
train_and_validate_probe.py::length_confound_test
(small: |len+ - len-| <= median over held-out pairs; large: > median).

Run from the repository root:  python artifacts/paper_sections/build_derived_sections.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts/sources/results"
SEC = ROOT / "artifacts/paper_sections"

QUANTS = ("awq", "gptq", "rtn", "w8a8", "w4a8")


def load(path: Path):
    with open(path) as fh:
        return json.load(fh)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- D1
D1_7B = [
    # model, dataset, layer, level12 json, input-length csv
    ("Llama-3.1-8B", "TriviaQA", 14, "ccs_probes/llama_level12_results.json"),
    ("Llama-3.1-8B", "ARC-Easy", 14, "ccs_probes_arc_easy/llama_level12_results.json"),
    ("Llama-3.1-8B", "CommonsenseQA", 14, "ccs_probes_commonsenseqa/llama_level12_results.json"),
    ("Qwen2.5-7B", "TriviaQA", 18, "ccs_probes_offset/qwen_level12_results.json"),
    ("Qwen2.5-7B", "ARC-Easy", 18, "ccs_probes_arc_easy_offset/qwen_level12_results.json"),
    ("Qwen2.5-7B", "CommonsenseQA", 17, "ccs_probes_commonsenseqa_offset/qwen_level12_results.json"),
    ("Mistral-7B", "TriviaQA", 13, "ccs_probes/mistral_level12_results.json"),
    ("Mistral-7B", "ARC-Easy", 15, "ccs_probes_arc_easy/mistral_level12_results.json"),
    ("Mistral-7B", "CommonsenseQA", 17, "ccs_probes_commonsenseqa/mistral_level12_results.json"),
]
DS_KEY = {"TriviaQA": "triviaqa", "ARC-Easy": "arc_easy", "CommonsenseQA": "commonsenseqa"}


def layer_entry(block, layer):
    return block[str(layer)] if isinstance(block, dict) else block[layer]


def build_d1() -> list[dict]:
    rows = []
    for model, dataset, layer, rel in D1_7B:
        res = load(SRC / rel)
        ts, lc = layer_entry(res["template_shuffle"], layer), layer_entry(res["length_confound"], layer)
        lens = SEC / "D1_template_length_control/inputs" / f"{model}_{dataset}_heldout_input_lengths.csv"
        with open(lens) as fh:
            diffs = [abs(int(r["input_len_plus"]) - int(r["input_len_minus"])) for r in csv.DictReader(fh)]
        med = statistics.median(diffs)
        rows.append({
            "model": model, "dataset": dataset, "layer": layer, "n_heldout_pairs": len(diffs),
            "n_small": sum(d <= med for d in diffs), "n_large": sum(d > med for d in diffs),
            "median_abs_input_length_difference": med,
            "template_aurocs": ";".join(f"{ts[k]:.6f}" for k in sorted(ts) if k.startswith("template_")),
            "g_temp": ts["auroc_range"], "auroc_small": lc["small_len_diff"], "auroc_large": lc["large_len_diff"],
            "g_len": lc["auroc_gap"], "template_protocol": "3 templates assigned across held-out pairs",
            "source": f"artifacts/sources/results/{rel}; {lens.relative_to(ROOT)}",
        })
    q14 = load(SRC / "qwen14b/validation/phase2.json")["structural"]
    m14 = load(SRC / "ministral3_14b/analysis/crossmodel_qwen14_comparison.json")["datasets"]
    for model, getter, src in (
        ("Qwen2.5-14B", lambda ds: q14[ds], "artifacts/sources/results/qwen14b/validation/phase2.json"),
        ("Ministral-3-14B", lambda ds: m14[ds]["level2"],
         "artifacts/sources/results/ministral3_14b/analysis/crossmodel_qwen14_comparison.json"),
    ):
        for dataset, key in DS_KEY.items():
            s = getter(key)
            ln = s["length"]
            tmpl = s.get("template_aurocs") or [t["auroc"] for t in s.get("templates", [])]
            rows.append({
                "model": model, "dataset": dataset, "layer": s["layer"],
                "n_heldout_pairs": ln["small_group_n"] + ln["large_group_n"],
                "n_small": ln["small_group_n"], "n_large": ln["large_group_n"],
                "median_abs_input_length_difference": ln["median_abs_input_length_difference"],
                "template_aurocs": ";".join(f"{x:.6f}" for x in tmpl), "g_temp": s["template_range"],
                "auroc_small": ln["small_group_auroc"], "auroc_large": ln["large_group_auroc"],
                "g_len": ln["auroc_gap"], "template_protocol": "same 200 held-out pairs under 5 system messages",
                "source": src,
            })
    return rows


# --------------------------------------------------------------------------- D5
def build_d5() -> list[dict]:
    rows = []
    for model, key, layer in (("Qwen2.5-7B", "qwen", 18), ("Llama-3.1-8B", "llama", 14), ("Mistral-7B", "mistral", 13)):
        base = SRC / "three_model_owngen_position_robustness_v1" / key
        last, mean = load(base / "last_metrics.json"), load(base / "mean_metrics.json")
        lp, mp = last["primary"], mean["primary"]
        assert lp["n_answers"] == mp["n_answers"]
        rows.append({
            "model": model, "dataset": "TriviaQA", "layer": layer, "scored_answers": lp["n_answers"],
            "n_exact_correct": lp["n_exact_correct"], "n_incorrect": lp["n_incorrect"],
            "last_token_auroc": lp["auroc"], "last_ci95_low": lp["auroc_ci"][0], "last_ci95_high": lp["auroc_ci"][1],
            "last_random_p": last["random_direction_control"]["empirical_p_upper"],
            "mean_pooled_auroc": mp["auroc"], "mean_ci95_low": mp["auroc_ci"][0], "mean_ci95_high": mp["auroc_ci"][1],
            "mean_random_p": mean["random_direction_control"]["empirical_p_upper"],
            "random_direction_count": last["random_direction_control"]["n_random_directions"],
            "source": f"artifacts/sources/results/three_model_owngen_position_robustness_v1/{key}/{{last,mean}}_metrics.json",
        })
    return rows


# --------------------------------------------------------------------------- E1
def build_e1() -> list[dict]:
    rows = []
    # 7-8B and 14B TriviaQA: question-paired bootstrap table (point estimates are the
    # full-sample AUROCs of the fixed FP16 direction).
    boot = SRC / "paper_auroc_bootstrap_v1/table_2.csv"
    names = {"llama": ("Llama-3.1-8B", 14), "qwen": ("Qwen2.5-7B", 18), "mistral": ("Mistral-7B", 13)}
    with open(boot) as fh:
        b = [r for r in csv.DictReader(fh) if r["model_key"] in names]
    fp = {r["model_key"]: float(r["auroc_estimate"]) for r in b if r["quantization"] == "fp16"}
    for r in b:
        if r["quantization"] == "fp16":
            continue
        model, layer = names[r["model_key"]]
        q = float(r["auroc_estimate"])
        rows.append({"model": model, "dataset": "TriviaQA", "layer": layer, "quantization": r["quantization"],
                     "n_heldout_pairs": int(r["n_pairs"]), "fp16_auroc": fp[r["model_key"]], "quantized_auroc": q,
                     "auroc_drop": fp[r["model_key"]] - q, "source": str(boot.relative_to(ROOT))})
    # 7-8B ARC-Easy / CommonsenseQA frozen transfer.
    t7 = SRC / "quant_transfer_arc_csqa_7b_v1/analysis/table2_frozen_transfer.csv"
    disp = {"llama": "Llama-3.1-8B", "qwen": "Qwen2.5-7B", "mistral": "Mistral-7B"}
    with open(t7) as fh:
        for r in csv.DictReader(fh):
            rows.append({"model": disp[r["model_key"]], "dataset": {"arc_easy": "ARC-Easy", "commonsenseqa": "CommonsenseQA"}[r["dataset"]],
                         "layer": int(r["layer"]), "quantization": r["quantization"], "n_heldout_pairs": int(r["n_heldout_pairs"]),
                         "fp16_auroc": float(r["fp16_auroc"]), "quantized_auroc": float(r["quantized_auroc"]),
                         "auroc_drop": float(r["auroc_drop"]), "source": str(t7.relative_to(ROOT))})
    # Qwen2.5-14B quantization grid.
    q14p = SRC / "qwen14b/quant_grid_v1/analysis.json"
    for c in load(q14p)["cells"]:
        rows.append({"model": "Qwen2.5-14B", "dataset": {"triviaqa": "TriviaQA", "arc_easy": "ARC-Easy", "commonsenseqa": "CommonsenseQA"}[c["dataset"]],
                     "layer": c["layer"], "quantization": c["condition"].replace("rtn_w4a16", "rtn"),
                     "n_heldout_pairs": c["n_test_pairs"], "fp16_auroc": c["fp16_auroc"], "quantized_auroc": c["quant_auroc"],
                     "auroc_drop": c["auroc_degradation"], "source": str(q14p.relative_to(ROOT))})
    # Ministral-3-14B quantization grid.
    m14p = SRC / "ministral3_14b/quant_grid_v1/analysis/job_a_results.json"
    for c in load(m14p)["section_6_1_and_6_3_cells"]:
        rows.append({"model": "Ministral-3-14B", "dataset": {"triviaqa": "TriviaQA", "arc_easy": "ARC-Easy", "commonsenseqa": "CommonsenseQA"}[c["dataset"]],
                     "layer": c["layer"], "quantization": c["condition"], "n_heldout_pairs": c["n_test_pairs"],
                     "fp16_auroc": c["fp16_auroc"], "quantized_auroc": c["quantized_auroc"],
                     "auroc_drop": c["auroc_degradation"], "source": str(m14p.relative_to(ROOT))})
    order_m = ["Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B", "Qwen2.5-14B", "Ministral-3-14B"]
    order_d = ["TriviaQA", "ARC-Easy", "CommonsenseQA"]
    rows.sort(key=lambda r: (order_m.index(r["model"]), order_d.index(r["dataset"]), QUANTS.index(r["quantization"])))
    assert len(rows) == 75, len(rows)
    return rows


def main() -> None:
    write(SEC / "D1_template_length_control/d1_template_length_summary.csv", build_d1())
    write(SEC / "D5_answer_span_aggregation/d5_answer_span_summary.csv", build_d5())
    write(SEC / "E1_cross_precision_transfer/e1_cross_precision_auroc.csv", build_e1())
    print("derived section CSVs written")


if __name__ == "__main__":
    main()
