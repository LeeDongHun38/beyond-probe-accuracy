"""Build the camera-ready numerical tables from the released experiment outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODELS = dict(llama="Llama-3.1-8B", qwen="Qwen2.5-7B", mistral="Mistral-7B",
              qwen14="Qwen2.5-14B", ministral14="Ministral-3-14B")
DATASETS = dict(triviaqa="TriviaQA", arc_easy="ARC-Easy", commonsenseqa="CommonsenseQA")


def read(relative):
    return pd.read_csv(ART / relative, dtype={"side": str}, float_precision="round_trip")


def benjamini_hochberg(p):
    """Return adjusted p-values in the input row order."""
    ordered = p.sort_values()
    ranked = ordered * len(p) / pd.Series(range(1, len(p) + 1), index=ordered.index)
    return ranked.iloc[::-1].cummin().iloc[::-1].clip(upper=1).reindex(p.index)


def fixed_null():
    keys = ["model", "dataset", "side"]
    cells = pd.concat([read("steering/fixed_alpha/arc_csqa/md_effects.csv"), read("steering/fixed_alpha/triviaqa/md_effects.csv")])
    draws = pd.concat([read("steering/fixed_alpha/arc_csqa/random_direction_effects.csv"),
                       read("steering/fixed_alpha/triviaqa/random_direction_effects.csv")])
    draws = draws.merge(cells[keys + ["md_effect"]], on=keys, validate="many_to_one")
    draws["absolute"] = draws.delta_p.abs()
    draws["exceeds"] = draws.absolute >= draws.md_effect.abs()
    summary = draws.groupby(keys).agg(md_effect=("md_effect", "first"), null_median=("absolute", "median"),
                                      null_max=("absolute", "max"), exceedance=("exceeds", "sum"),
                                      n=("direction_index", "nunique")).reset_index()
    summary["p"] = (1 + summary.exceedance) / (1 + summary.n)
    summary["q"] = benjamini_hochberg(summary.p)
    return summary


def geometry_panel():
    """Recompute the two BH families directly from the 10,000 stored permutations."""
    protocol = json.loads((ART / "geometry/protocol.json").read_text())
    rows = []
    for cell, layer in protocol["cells"].items():
        model, dataset = cell.split("/")
        with np.load(ART / "geometry" / f"{model}_{dataset}.npz") as data:
            obs, null = data["observed"], data["null"]
        if null.shape != (10000, 5, 6) or obs.shape != (5, 6):
            raise ValueError(f"Invalid permutation panel shape for {cell}")
        for j, quant in enumerate(protocol["quantizations"]):
            stats = [("R1", obs[j, 0], null[:, j, 0]), ("R50", obs[j, -1], null[:, j, -1]),
                     ("Rinc50", obs[j, -1] - obs[j, 0], null[:, j, -1] - null[:, j, 0])]
            for label, value, draws in stats:
                p = (1 + np.count_nonzero(draws >= value)) / (len(draws) + 1)
                rows.append(dict(model=model, dataset=dataset, layer=layer, quantization=quant,
                                 statistic=label, observed=value, null_mean=draws.mean(), p_upper=p))
    panel = pd.DataFrame(rows)
    panel["q_bh_family"] = panel.groupby(panel.statistic == "Rinc50").p_upper.transform(benjamini_hochberg)
    return panel


def build(out):
    out.mkdir(parents=True, exist_ok=True)

    def save(name, rows):
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False, float_format="%.12g")

    raw = read("probes/estimator_sweep.csv").rename(columns={"model_name": "model"})
    raw["model"] = raw.model.replace({"Ministral-14B": "Ministral-3-14B"})
    raw = raw[["model", "layer", "estimator", "auc", "align", "stab"]]
    save("table01_triviaqa_estimator_sweep", raw)
    transfer = read("transfer/triviaqa/results.csv")
    transfer["model"] = transfer.model.map(MODELS)
    full = transfer[["model", "precision", "full_test_auroc", "fp16_auroc_drop"]].copy()
    save("table02_triviaqa_quant_transfer", full)
    counts = []
    for ds, label in DATASETS.items():
        base = ROOT / "data/pairs" / ds
        counts.append({"dataset": label, **{s: len(json.loads((base / f"{s}.json").read_text()))
                                            for s in ("all", "train", "test")}})
    save("table03_dataset_counts", counts)
    md = raw[raw.estimator == "D0"][["model", "layer", "auc", "stab"]].assign(dataset="TriviaQA")
    other = pd.read_csv(ART / "probes/arc_csqa_md_stability.csv")
    other = other[other.dataset != "triviaqa"].rename(columns={"recomputed_heldout_md_auroc": "auc",
                                                               "split_half_mean": "stab"})
    other["dataset"] = other.dataset.map(DATASETS)
    md = pd.concat([md, other])[["model", "dataset", "layer", "auc", "stab"]]
    save("table09_selected_layers", md[["model", "dataset", "layer"]])
    save("table11_md_auroc_stability", md)
    save("table12_scale_normalized_lr", read("probes/scale_normalized_lr.csv"))

    pr = read("probes/standardized_lr/participation_ratio.csv")
    ext = []
    for path in sorted((ART / "probes/standardized_lr").glob("*/*.json")):
        fit = json.loads(path.read_text())
        ext.append(dict(model=MODELS[fit["model"]], estimator=fit["direction"],
                        auc=fit["auc"], align=fit["align"], stab=fit["stab"]))
    fits = pd.concat([raw.drop(columns="layer"), pd.DataFrame(ext)])
    effects = read("figure_inputs/figure2_input.csv")
    std = read("probes/standardized_lr/standardized_steering.csv")
    std["direction"] = std.condition.str.split("_").str[0]
    effects = pd.concat([effects, std]).pivot(index=["model", "direction"], columns="side", values="delta_p")
    fits_all = pr.join(effects, on=["model", "direction"]).rename(
        columns={"direction": "estimator", "false": "delta_p_false", "true": "delta_p_true"})
    fits_all["model"] = fits_all.model.map(MODELS)
    fits_all = fits.merge(fits_all, on=["model", "estimator"], validate="one_to_one")
    fits_all = fits_all.merge(md[md.dataset == "TriviaQA"][["model", "layer"]], on="model")
    large = fits_all.model.isin([MODELS["qwen14"], MODELS["ministral14"]])
    save("table13_estimator_sensitivity_7_8b", fits_all[~large])
    save("table14_estimator_sensitivity_14b", fits_all[large])
    doses = read("figure_inputs/figure3b_input.csv").merge(md[["model", "dataset", "layer"]], on=["model", "dataset"])
    doses = doses[["model", "dataset", "layer", "side", "dose", "alpha", "r", "delta_p"]]
    fixed = doses[doses.dose == "alpha4"].copy()
    fixed["median_norm"] = fixed.alpha / fixed.r
    save("table15_md_intervention_alpha4", fixed)
    save("table16_md_intervention_matched_r010", doses[doses.dose == "r010"])
    matched = read("steering/matched_dose/matched_results.csv")
    save("table17_estimator_matched_dose", matched[matched.condition.str.startswith(("D0_", "D1_", "D2_", "D3_"))])
    null = fixed_null()
    null["model"] = null.model.replace({"qwen14b": "qwen14", "ministral14b": "ministral14"}).map(MODELS)
    null["dataset"] = null.dataset.map(DATASETS)
    save("table18_random_direction_alpha4", null.merge(fixed[["model", "dataset", "side", "layer", "r"]],
                                                      on=["model", "dataset", "side"], validate="one_to_one"))
    nulls = read("steering/matched_dose/null_results.csv")
    iso = nulls[nulls.family == "isotropic"][["model", "side", "md_effect", "exceedance", "raw_p"]]
    iso = iso.merge(matched[matched.condition == "D0_r0.05"][["model", "side", "alpha"]], on=["model", "side"])
    for side in ("false", "true"):
        row = json.loads((ART / f"steering/matched_dose/qwen14/isotropic_{side}.json").read_text())
        iso.loc[len(iso)] = ["qwen14", side, row["md"], row["exceedance"], row["p"], row["alpha"]]
    iso["q"] = benjamini_hochberg(iso.raw_p)
    save("table19_random_direction_r005", iso)
    save("table20_covariance_and_dose_controls", nulls[nulls.family != "isotropic"])

    template = pd.read_csv(ART / "controls/template_length/arc_csqa.csv")
    template = template[template.dataset != "TriviaQA"]
    rows = []
    for key, model in MODELS.items():
        r = json.loads((ART / f"controls/template_length/triviaqa/{key}.json").read_text())
        n = r["length"]
        rows.append(dict(model=model, dataset="TriviaQA", n_heldout_pairs=n["small_n"] + n["large_n"],
                         n_small=n["small_n"], n_large=n["large_n"], g_temp=r["template_range"], g_len=n["gap"]))
    save("table21_template_length_control", pd.concat([pd.DataFrame(rows), template])[list(rows[0])])
    style = json.loads((ART / "controls/gold_donor/style.json").read_text())
    donor = read("controls/gold_donor/results.csv").rename(columns={"auc": "gold_donor"})
    donor["model"] = donor.model.map(MODELS)
    donor = donor.merge(md[md.dataset == "TriviaQA"][["model", "auc"]], on="model").rename(columns={"auc": "original"})
    donor = donor[["model", "layer", "original", "gold_donor"]]
    donor.loc[len(donor)] = ["Answer-only style", None, style["original"]["test_auroc"], style["gold_donor"]["test_auroc"]]
    save("table22_source_style_control", donor)
    own = read("controls/own_generation/triviaqa/results.csv")
    own["model"] = own.model.map(MODELS)
    own["incorrect"] = own.n_retained_answers - own.n_positive
    own["excluded"] = 16 * own.n_questions - own.n_retained_answers
    save("table23_owngen_last_token", own[["model", "layer", "n_retained_answers", "n_positive", "incorrect",
                                          "excluded", "auc", "auc_ci_low", "auc_ci_high", "auc_random_p"]])
    own["dataset"] = "TriviaQA"
    own["generations_per_question"] = 16
    own["valid"] = own.n_questions
    own["random_exc"] = (own.rho_random_p * 1001 - 1).round()
    arc = pd.read_csv(ART / "controls/own_generation/arc_easy_pre_answer.csv")
    arc = arc[arc.dataset == "ARC-Easy"].rename(columns={"spearman_rho": "rho", "ci95_low": "rho_ci_low",
        "ci95_high": "rho_ci_high", "outcome_valid_questions": "valid"})
    save("table24_pre_answer_readout", pd.concat([own, arc])[["model", "dataset", "layer", "valid", "n_questions",
         "generations_per_question", "rho", "rho_ci_low", "rho_ci_high", "random_exc", "rho_random_p"]])
    quant = pd.read_csv(ART / "transfer/arc_csqa_auroc.csv")
    quant = quant[quant.dataset != "TriviaQA"].rename(columns={"quantization": "precision",
        "quantized_auroc": "full_test_auroc", "auroc_drop": "fp16_auroc_drop"})
    fp = quant[["model", "dataset", "fp16_auroc"]].drop_duplicates().rename(columns={"fp16_auroc":"full_test_auroc"})
    all_quant = pd.concat([full.assign(dataset="TriviaQA"), quant, fp.assign(precision="fp16", fp16_auroc_drop=0)])
    save("table25_cross_precision_transfer", all_quant[["model","dataset","precision","full_test_auroc","fp16_auroc_drop"]]
         .merge(md[["model","dataset","layer"]], on=["model","dataset"], validate="many_to_one"))
    geo = geometry_panel()
    save("table26_label_permuted_geometry", geo[["model","dataset","layer","quantization","statistic",
                                                "observed","null_mean","p_upper","q_bh_family"]])
    margins = read("transfer/margins/margins_75.csv")
    ranges = margins.groupby(["model", "dataset", "layer"])[["relative_shrinkage", "margin_flip_rate"]].agg(["min", "max"]) * 100
    ranges.columns = ["_".join(col) + "_pct" for col in ranges.columns]
    save("table27_projection_margins", ranges.reset_index())
    save("table28_operating_points", transfer[["model","precision","evaluation_auroc","threshold","fpr","fnr",
         "balanced_accuracy","flip_numerator","flip_denominator","decision_flip_rate","decision_flip_rate_ci_low","decision_flip_rate_ci_high"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper/tables")
    build(parser.parse_args().out_dir)
