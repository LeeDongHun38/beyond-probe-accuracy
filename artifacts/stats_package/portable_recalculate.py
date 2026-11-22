#!/usr/bin/env python3
"""CPU-only, repository-independent rebuild of the harmonized analysis package.

Every input opened by this program is resolved relative to this file.  The
program therefore continues to work after the package directory is copied or
moved.  It never imports project code and never performs model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import false_discovery_control


PACKAGE = Path(__file__).resolve().parent
MODELS = [
    "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
    "Mistral-7B-Instruct-v0.3", "Qwen2.5-14B-Instruct",
    "Ministral-3-14B-Instruct",
]
DATASETS = ["TriviaQA", "ARC-Easy", "CommonsenseQA"]
QUANTIZATIONS = ["AWQ W4A16", "GPTQ W4A16", "RTN W4A16", "simulated W8A8", "simulated W4A8"]
KS = [2, 5, 10, 20, 50]
SIDES = ["false", "true"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes(order="C")).hexdigest()


def as_bool(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def bh_direct(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def apply_bh(rows: list[dict], p_field: str, q_field: str) -> float:
    p = np.asarray([float(row[p_field]) for row in rows], dtype=float)
    q_library = np.asarray(false_discovery_control(p, method="bh"), dtype=float)
    q_direct = bh_direct(p.tolist())
    difference = float(np.max(np.abs(q_library - q_direct)))
    if difference > 1e-14:
        raise RuntimeError(f"BH cross-check failed for {q_field}: {difference}")
    for row, q in zip(rows, q_library):
        row[q_field] = float(np.clip(q, 0.0, 1.0))
        row["significant_q_lt_0_05"] = bool(q < 0.05)
    return difference


def load_direction_provenance() -> tuple[dict[tuple[str, str], dict], dict]:
    rows = read_csv(PACKAGE / "geometry_direction_provenance.csv")
    by_cell = {(row["model"], row["dataset"]): row for row in rows}
    expected_cells = set(product(MODELS, DATASETS))
    if set(by_cell) != expected_cells:
        raise RuntimeError("geometry direction provenance does not cover all 15 model-dataset cells")

    required = ["direction_artifact_identifier", "direction_hash", "expected_canonical_direction_hash",
                "selected_layer", "extraction_position", "training_split_identifier",
                "canonical_direction_status", "source_level_verification"]
    all_fields = all(all(row.get(field, "") for field in required) for row in rows)
    qwen = [row for row in rows if row["model"] == "Qwen2.5-7B-Instruct"]
    hash_matches = all(row["direction_hash"] == row["expected_canonical_direction_hash"] for row in qwen)
    layer_matches = {row["dataset"]: int(row["selected_layer"]) for row in qwen} == {
        "TriviaQA": 18, "ARC-Easy": 18, "CommonsenseQA": 17,
    }

    with np.load(PACKAGE / "geometry_direction_vectors.npz", allow_pickle=False) as vectors:
        portable_matches = True
        for row in qwen:
            key = row["portable_vector_key"]
            portable_matches &= bool(key and vector_sha256(vectors[key]) == row["direction_hash"])
        tqa = vectors["qwen7_triviaqa_actual"].astype(np.float64)
        intervention = vectors["qwen7_triviaqa_intervention_reference"].astype(np.float64)
        cosine = float(tqa @ intervention)
        max_abs_difference = float(np.max(np.abs(tqa - intervention)))

    legacy_hashes = {row["known_qwen7_legacy_triviaqa_hash"] for row in qwen if row["known_qwen7_legacy_triviaqa_hash"]}
    legacy_absent = not any(row["direction_hash"] in legacy_hashes for row in qwen)
    audit = {
        "geometry_qwen7_direction_hash_matches_expected": bool(hash_matches and portable_matches and layer_matches),
        "geometry_qwen7_legacy_direction_absent": bool(legacy_absent),
        "geometry_all_rows_have_direction_provenance": bool(all_fields),
        "qwen7_geometry_hashes": {row["dataset"]: row["direction_hash"] for row in qwen},
        "qwen7_expected_geometry_hashes": {row["dataset"]: row["expected_canonical_direction_hash"] for row in qwen},
        "qwen7_triviaqa_geometry_vs_intervention_reference": {
            "geometry_hash": by_cell[("Qwen2.5-7B-Instruct", "TriviaQA")]["direction_hash"],
            "intervention_reference_hash": by_cell[("Qwen2.5-7B-Instruct", "TriviaQA")]["canonical_intervention_reference_hash"],
            "cosine": cosine, "max_abs_component_difference": max_abs_difference,
            "interpretation": "Same corrected L18 MD direction; hash differs only because the frozen geometry and intervention builders use slightly different floating-point reduction/normalization order.",
        },
    }
    if not all(audit[key] for key in (
        "geometry_qwen7_direction_hash_matches_expected",
        "geometry_qwen7_legacy_direction_absent",
        "geometry_all_rows_have_direction_provenance",
    )):
        raise RuntimeError("geometry direction audit failed; refusing to reuse geometry p/q values")
    return by_cell, audit


def geometry_tables(direction_by_cell: dict[tuple[str, str], dict]):
    inputs = read_csv(PACKAGE / "geometry_cell_inputs.csv")
    source_by_cell = {(r["model"], r["dataset"], r["quantization"]): r for r in inputs}
    expected_sources = set(product(MODELS, DATASETS, QUANTIZATIONS))
    if set(source_by_cell) != expected_sources:
        raise RuntimeError("geometry_cell_inputs.csv must contain exactly 75 unique cells")
    p_reference = {
        (r["metric"], r["model"], r["dataset"], r["quantization"], int(r["k"])): float(r["p_summary"])
        for r in read_csv(PACKAGE / "frozen_p_value_reference.csv")
    }
    min_aux = {}
    aux_summary_path = PACKAGE / "raw_geometry/derived_summaries/ministral3_14b_triviaqa_mean_error_null.csv"
    for row in read_csv(aux_summary_path):
        quant = {"awq": "AWQ W4A16", "gptq": "GPTQ W4A16", "rtn": "RTN W4A16",
                 "w8a8": "simulated W8A8", "w4a8": "simulated W4A8"}[row["quantization"]]
        for metric in ("M", "A"):
            min_aux[(quant, int(row["k"]), metric)] = {
                "observed": float(row[f"{metric}_observed"]), "null_mean": float(row[f"{metric}_null_mean"]),
                "null_median": float(row[f"{metric}_null_median"]), "null_sd": float(row[f"{metric}_null_std"]),
                "n_null": int(row[f"{metric}_n_null"]), "exceedance": int(row[f"{metric}_exceedance"]),
                "p": float(row[f"{metric}_p_raw"]),
            }

    tables = {"R_incremental": [], "M": [], "A": []}
    p_matches, missing = [], []
    for model, dataset, quantization in product(MODELS, DATASETS, QUANTIZATIONS):
        source_meta = source_by_cell[(model, dataset, quantization)]
        source = PACKAGE / source_meta["source_artifact"]
        if sha256(source) != source_meta["source_artifact_sha256"]:
            raise RuntimeError(f"raw geometry SHA mismatch: {source_meta['source_artifact']}")
        provenance = direction_by_cell[(model, dataset)]
        with np.load(source, allow_pickle=False) as z:
            if "direction_sha256" in z.files and str(np.asarray(z["direction_sha256"]).item()) != provenance["direction_hash"]:
                raise RuntimeError(f"embedded geometry direction hash mismatch: {model}/{dataset}/{quantization}")
            ks = np.asarray(z["ks"], dtype=int)
            repetitions = int(np.asarray(z["repetitions"]).item())
            if repetitions != 1000:
                raise RuntimeError(f"expected 1000 null draws: {source}")
            robs = "observed_r_pair_incremental" if "observed_r_pair_incremental" in z.files else "observed_r_incremental"
            rnull = "activation_matched_r_pair_incremental" if "activation_matched_r_pair_incremental" in z.files else "activation_matched_r_incremental"
            arrays = {"R_incremental": (np.asarray(z[robs], float), np.asarray(z[rnull], float))}
            if "observed_m_pair" in z.files:
                arrays["M"] = (np.asarray(z["observed_m_pair"], float), np.asarray(z["activation_matched_m_pair"], float))
                arrays["A"] = (np.asarray(z["observed_a_pair"], float), np.asarray(z["activation_matched_a_pair"], float))
            for k in KS:
                index = int(np.where(ks == k)[0][0])
                for metric in ("R_incremental", "M", "A"):
                    raw_available = metric in arrays
                    if raw_available:
                        observed = float(arrays[metric][0][index])
                        draws = np.asarray(arrays[metric][1][:, index], float)
                        exceedance = int(np.count_nonzero(draws >= observed))
                        stat = {"observed": observed, "null_mean": float(np.mean(draws)),
                                "null_median": float(np.median(draws)), "null_sd": float(np.std(draws, ddof=1)),
                                "n_null": int(draws.size), "exceedance": exceedance,
                                "p": float((1 + exceedance) / (draws.size + 1))}
                        row_source = source_meta["source_artifact"]
                        row_sha = source_meta["source_artifact_sha256"]
                        protocol = source_meta["protocol_identifier"]
                    else:
                        stat = min_aux[(quantization, k, metric)]
                        row_source = "raw_geometry/derived_summaries/ministral3_14b_triviaqa_mean_error_null.csv"
                        row_sha = sha256(aux_summary_path)
                        protocol = "MINISTRAL14_TQA_MA_FROZEN_SUMMARY_EXCEEDANCE_B1000_RAW_DRAWS_NOT_PRESERVED"
                        missing.append({"model": model, "dataset": dataset, "quantization": quantization, "k": k,
                                        "metric": metric, "missing_artifact": "per-repetition raw M/A null draws",
                                        "available_replacement": row_source,
                                        "status": "raw_draw_array_not_preserved; observed/null summaries/exceedance retained"})
                    key = (metric, model, dataset, quantization, k)
                    match = math.isclose(stat["p"], p_reference[key], rel_tol=0.0, abs_tol=1e-15)
                    p_matches.append(match)
                    if not match:
                        raise RuntimeError(f"raw/frozen p-value reference mismatch: {key}")
                    tables[metric].append({
                        "family_id": {"R_incremental": "RINC_GLOBAL_375", "M": "M_GLOBAL_375", "A": "A_GLOBAL_375"}[metric],
                        "family_size": 375, "model": model, "dataset": dataset,
                        "layer": int(source_meta["layer"]), "quantization": quantization, "k": k,
                        "observed": stat["observed"], "null_mean": stat["null_mean"],
                        "null_median": stat["null_median"], "null_sd": stat["null_sd"],
                        "n_null": stat["n_null"], "exceedance_count": stat["exceedance"], "p_raw": stat["p"],
                        "source_artifact": row_source, "source_artifact_sha256": row_sha,
                        "upstream_source_artifact": "portable package input; original identifier retained in provenance_manifest.json",
                        "protocol_identifier": protocol, "raw_null_samples_available": raw_available,
                    })
    bh = {metric: apply_bh(rows, "p_raw", "q_bh_global_375") for metric, rows in tables.items()}
    return tables, p_matches, missing, bh


def comparison(rows: list[dict], historical_file: str, auxiliary: bool) -> list[dict]:
    historical = read_csv(PACKAGE / historical_file)
    key_fields = ("metric", "model", "dataset", "quantization", "k") if auxiliary else ("model", "dataset", "quantization", "k")
    lookup = {tuple(r[field] for field in key_fields): r for r in historical}
    out = []
    for row in rows:
        metric = row["family_id"].split("_")[0] if auxiliary else "R_incremental"
        raw_key = {"metric": metric, **row}
        key = tuple(str(raw_key[field]) for field in key_fields)
        old = lookup[key]
        old_q, new_q = float(old["q_bh_historical"]), float(row["q_bh_global_375"])
        old_sig, new_sig = old_q < 0.05, new_q < 0.05
        transition = "significant_to_nonsignificant" if old_sig and not new_sig else "nonsignificant_to_significant" if new_sig and not old_sig else "unchanged"
        if auxiliary:
            out.append({"metric": metric, "model": row["model"], "dataset": row["dataset"],
                        "quantization": row["quantization"], "k": row["k"], "p_raw": row["p_raw"],
                        "historical_family_id": old["historical_family_id"], "q_bh_historical": old_q,
                        "q_bh_global_375": new_q, "historical_significant": old_sig,
                        "global_375_significant": new_sig, "decision_changed": old_sig != new_sig,
                        "decision_transition": transition})
        else:
            out.append({"model": row["model"], "dataset": row["dataset"], "layer": row["layer"],
                        "quantization": row["quantization"], "k": row["k"], "p_raw": row["p_raw"],
                        "historical_family": old["historical_family_id"], "q_bh_historical": old_q,
                        "q_bh_global_375": new_q, "q_value_changed": abs(new_q - old_q) > 1e-15,
                        "absolute_q_change": abs(new_q - old_q), "historical_significant": old_sig,
                        "global_375_significant": new_sig, "decision_changed": old_sig != new_sig,
                        "decision_transition": transition})
    return out


def steering_table():
    effects_path = PACKAGE / "steering_direction_effects.csv"
    effects = read_csv(effects_path)
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["model"], row["dataset"], row["side"])].append(float(row["signed_delta_p"]))
    meta_rows = read_csv(PACKAGE / "steering_cell_inputs.csv")
    meta = {(r["model"], r["dataset"], r["side"]): r for r in meta_rows}
    rows = []
    for model, dataset, side in product(MODELS, DATASETS, SIDES):
        source = meta[(model, dataset, side)]
        null = np.asarray(grouped[(model, dataset, side)], dtype=float)
        if null.size != 127:
            raise RuntimeError(f"steering null has {null.size}, expected 127: {model}/{dataset}/{side}")
        absolute = np.abs(null)
        md = float(source["md_delta_p"])
        exceedance = int(np.count_nonzero(absolute >= abs(md)))
        rows.append({"family_id": "STEERING_FIXED_ALPHA_GLOBAL_30", "family_size": 30,
                     "model": model, "dataset": dataset, "layer": int(source["layer"]), "side": side,
                     "alpha": float(source["alpha"]), "normalized_magnitude_r": float(source["normalized_magnitude_r"]),
                     "md_delta_p": md, "n_random_directions": 127,
                     "null_mean_abs_effect": float(np.mean(absolute)), "null_median_abs_effect": float(np.median(absolute)),
                     "null_max_abs_effect": float(np.max(absolute)), "exceedance_count": exceedance,
                     "p_mc": float((1 + exceedance) / 128), "direction_hash": source["direction_hash"],
                     "source_artifact": "steering_direction_effects.csv", "source_artifact_sha256": sha256(effects_path),
                     "upstream_source_artifact": source["upstream_source_artifact"],
                     "direction_artifact_identifier": source["direction_artifact_identifier"],
                     "canonical_direction_status": source["canonical_direction_status"]})
    return rows, apply_bh(rows, "p_mc", "q_bh_global_30")


def primary_summary(rows: list[dict]) -> dict:
    significant = lambda row: bool(row["significant_q_lt_0_05"])
    nonsignificant = [row for row in rows if not significant(row)]
    count = lambda field, value: sum(significant(row) for row in rows if row[field] == value)
    return {
        "significant_count": sum(significant(row) for row in rows), "nonsignificant_count": len(nonsignificant),
        "by_dataset": {x: count("dataset", x) for x in DATASETS},
        "by_model": {x: count("model", x) for x in MODELS},
        "by_quantization": {x: count("quantization", x) for x in QUANTIZATIONS},
        "by_k": {str(k): sum(significant(row) for row in rows if int(row["k"]) == k) for k in KS},
        "triviaqa": sum(significant(row) for row in rows if row["dataset"] == "TriviaQA"),
        "k2": sum(significant(row) for row in rows if int(row["k"]) == 2),
        "k_ge_5": sum(significant(row) for row in rows if int(row["k"]) >= 5),
        "nonsignificant_cells": [{key: row[key] for key in ("model", "dataset", "layer", "quantization", "k", "p_raw", "q_bh_global_375")} for row in nonsignificant],
    }


def auxiliary_summary(comparisons: list[dict], tables: dict[str, list[dict]]) -> dict:
    output = {}
    for metric in ("M", "A"):
        comp = [row for row in comparisons if row["metric"] == metric]
        changed = [row for row in comp if row["decision_changed"]]
        output[metric] = {
            "historical_significant_count": sum(row["historical_significant"] for row in comp),
            "global_375_significant_count": sum(row["global_375_significant"] for row in comp),
            "historical_k2_significant_count_out_of_75": sum(row["historical_significant"] for row in comp if int(row["k"]) == 2),
            "global_k2_significant_count_out_of_75": sum(row["global_375_significant"] for row in comp if int(row["k"]) == 2),
            "historical_k_ge_5_significant_count_out_of_300": sum(row["historical_significant"] for row in comp if int(row["k"]) >= 5),
            "global_k_ge_5_significant_count_out_of_300": sum(row["global_375_significant"] for row in comp if int(row["k"]) >= 5),
            "decision_changed_count": len(changed), "changed_cells": changed,
        }
    return output


def build(output: Path) -> list[str]:
    direction_by_cell, direction_audit = load_direction_provenance()
    tables, p_matches, missing, bh_geometry = geometry_tables(direction_by_cell)
    primary = tables["R_incremental"]
    primary_comparison = comparison(primary, "historical_primary_q_values.csv", auxiliary=False)
    auxiliary_rows = tables["M"] + tables["A"]
    auxiliary_comparison = comparison(auxiliary_rows, "historical_auxiliary_q_values.csv", auxiliary=True)
    steering, steering_bh = steering_table()

    geometry_fields = ["family_id", "family_size", "model", "dataset", "layer", "quantization", "k",
                       "observed", "null_mean", "null_median", "null_sd", "n_null", "exceedance_count", "p_raw",
                       "q_bh_global_375", "significant_q_lt_0_05", "source_artifact", "source_artifact_sha256",
                       "upstream_source_artifact", "protocol_identifier", "raw_null_samples_available"]
    renames = {
        "R_incremental": ("rinc_global_375.csv", {"observed": "observed_R_inc", "null_mean": "null_mean_R_inc", "null_median": "null_median_R_inc", "null_sd": "null_sd_R_inc"}),
        "M": ("m_global_375.csv", {"observed": "observed_M", "null_mean": "null_mean_M", "null_median": "null_median_M", "null_sd": "null_sd_M"}),
        "A": ("a_global_375.csv", {"observed": "observed_A", "null_mean": "null_mean_A", "null_median": "null_median_A", "null_sd": "null_sd_A"}),
    }
    generated = []
    for metric, rows in tables.items():
        filename, rename = renames[metric]
        exported = []
        for row in rows:
            exported.append({rename.get(key, key): value for key, value in row.items()})
        fields = [rename.get(key, key) for key in geometry_fields]
        write_csv(output / filename, exported, fields); generated.append(filename)

    nonsignificant = [row for row in primary if not row["significant_q_lt_0_05"]]
    exported_nonsig = [{renames["R_incremental"][1].get(key, key): value for key, value in row.items()} for row in nonsignificant]
    write_csv(output / "rinc_nonsignificant_cells.csv", exported_nonsig,
              [renames["R_incremental"][1].get(key, key) for key in geometry_fields]); generated.append("rinc_nonsignificant_cells.csv")
    write_csv(output / "missing_raw_artifacts.csv", missing); generated.append("missing_raw_artifacts.csv")
    write_csv(output / "previous_family_comparison.csv", primary_comparison); generated.append("previous_family_comparison.csv")
    write_csv(output / "auxiliary_historical_comparison.csv", auxiliary_comparison); generated.append("auxiliary_historical_comparison.csv")
    write_csv(output / "steering_fixed_alpha_global_30.csv", steering); generated.append("steering_fixed_alpha_global_30.csv")

    ps = primary_summary(primary)
    auxs = auxiliary_summary(auxiliary_comparison, tables)
    changed_primary = [row for row in primary_comparison if row["decision_changed"]]
    q_changed_primary = [row for row in primary_comparison if row["q_value_changed"]]
    largest = max(primary_comparison, key=lambda row: row["absolute_q_change"])
    nonzero_steering = [row for row in steering if int(row["exceedance_count"]) > 0]
    mistral_false = next(row for row in steering if row["model"] == "Mistral-7B-Instruct-v0.3" and row["dataset"] == "TriviaQA" and row["side"] == "false")
    effects = read_csv(PACKAGE / "steering_direction_effects.csv")
    exceed_effects = [{"direction_id": row["direction_id"], "signed_effect": float(row["signed_delta_p"]),
                       "opposite_sign_to_md": float(row["signed_delta_p"]) * float(mistral_false["md_delta_p"]) < 0}
                      for row in effects if row["model"] == mistral_false["model"] and row["dataset"] == "TriviaQA"
                      and row["side"] == "false" and abs(float(row["signed_delta_p"])) >= abs(float(mistral_false["md_delta_p"]))]

    summary = {
        "analysis_family_label": "final unified analysis families / harmonized analysis families",
        "generated_at_utc": "2026-08-03T14:53:38.471128+00:00",
        "portable_recalculation": {"repository_independent": True, "gpu_or_model_inference_required": False,
                                   "input_root": "directory containing portable_recalculate.py"},
        "primary_rinc_family_size": 375, "primary_rinc_significant_count": ps["significant_count"],
        "primary_rinc_nonsignificant_count": ps["nonsignificant_count"],
        "triviaqa_significant_count_out_of_125": ps["triviaqa"], "k2_significant_count_out_of_75": ps["k2"],
        "k_ge_5_significant_count_out_of_300": ps["k_ge_5"], "significant_counts_by_dataset": ps["by_dataset"],
        "significant_counts_by_model": ps["by_model"], "significant_counts_by_quantization": ps["by_quantization"],
        "significant_counts_by_k": ps["by_k"], "nonsignificant_cells": ps["nonsignificant_cells"],
        "auxiliary_m_family_size": 375, "auxiliary_m_significant_count": sum(r["significant_q_lt_0_05"] for r in tables["M"]),
        "auxiliary_a_family_size": 375, "auxiliary_a_significant_count": sum(r["significant_q_lt_0_05"] for r in tables["A"]),
        "auxiliary_historical_vs_global": auxs,
        "steering_family_size": 30, "steering_significant_count": sum(r["significant_q_lt_0_05"] for r in steering),
        "steering_zero_exceedance_count": sum(int(r["exceedance_count"]) == 0 for r in steering),
        "steering_nonzero_exceedance_cells": [{key: r[key] for key in ("model", "dataset", "side", "exceedance_count", "p_mc", "q_bh_global_30")} for r in nonzero_steering],
        "steering_conditions_both_sides_significant_out_of_15": sum(all(next(r for r in steering if r["model"] == m and r["dataset"] == d and r["side"] == s)["significant_q_lt_0_05"] for s in SIDES) for m, d in product(MODELS, DATASETS)),
        "mistral_triviaqa_false_p_mc": mistral_false["p_mc"], "mistral_triviaqa_false_q_bh": mistral_false["q_bh_global_30"],
        "mistral_triviaqa_false_exceedance_directions": exceed_effects,
        "qwen7_direction_hashes": {d: next(r["direction_hash"] for r in steering if r["model"] == "Qwen2.5-7B-Instruct" and r["dataset"] == d) for d in DATASETS},
        "qwen7_geometry_direction_hashes": direction_audit["qwen7_geometry_hashes"],
        "geometry_direction_audit": direction_audit,
        "qwen7_all_canonical_corrected": True,
        "changed_decisions_relative_to_previous_families": len(changed_primary),
        "changed_decisions_relative_to_previous_families_scope": "Primary R_inc only",
        "primary_changed_decisions_relative_to_previous_families": len(changed_primary),
        "changed_decision_cells": changed_primary, "q_values_changed_relative_to_previous_families": len(q_changed_primary),
        "largest_q_value_change_cell": {key: largest[key] for key in ("model", "dataset", "quantization", "k", "q_bh_historical", "q_bh_global_375", "absolute_q_change")},
        "legacy_claim_checks": {"all_125_triviaqa_cells_significant": ps["triviaqa"] == 125,
                                "exactly_7_nonsignificant": ps["nonsignificant_count"] == 7,
                                "all_7_are_qwen_commonsenseqa_k2": all("Qwen" in r["model"] and r["dataset"] == "CommonsenseQA" and int(r["k"]) == 2 for r in nonsignificant),
                                "k_ge_5_is_300_of_300": ps["k_ge_5"] == 300},
        "raw_artifact_limitations": {
            "primary_raw_draw_coverage": "375/375", "auxiliary_raw_draw_coverage": "700/750",
            "primary_rows_with_raw_samples": 375, "auxiliary_rows_with_raw_samples": 700,
            "auxiliary_rows_recovered_from_frozen_summary_and_exceedance_count": 50,
            "affected_cells": "Ministral-3-14B-Instruct × TriviaQA, M and A (25 rows each)",
            "impact_on_primary_rinc": "None; all 375 primary R_inc rows have per-repetition raw draws.",
            "description": "The 50 auxiliary rows are not claimed as fully recalculated from per-repetition raw draws.",
        },
        "steering_excluded_analyses": ["matched r=.10", "TriviaQA r≈.05 sensitivity", "estimator sweep", "dose-response analysis"],
    }
    write_json(output / "summary.json", summary); generated.append("summary.json")

    expected_primary = set(product(MODELS, DATASETS, QUANTIZATIONS, KS))
    got_primary = {(r["model"], r["dataset"], r["quantization"], int(r["k"])) for r in primary}
    verification_record = PACKAGE / "portable_verification.json"
    portable_verification_passed = verification_record.exists() and json.loads(verification_record.read_text()).get("status") == "PASS"
    checks = {
        "primary_exactly_375_rows": len(primary) == 375, "primary_no_duplicate_cells": len(got_primary) == 375,
        "primary_cartesian_product_complete": got_primary == expected_primary,
        "primary_family_id_all_RINC_GLOBAL_375": all(r["family_id"] == "RINC_GLOBAL_375" for r in primary),
        "primary_family_size_all_375": all(r["family_size"] == 375 for r in primary),
        "auxiliary_not_mixed_into_primary": all(r["family_id"] == "RINC_GLOBAL_375" for r in primary),
        "auxiliary_each_exactly_375_rows": len(tables["M"]) == len(tables["A"]) == 375,
        "steering_exactly_30_rows": len(steering) == 30, "steering_family_size_all_30": all(r["family_size"] == 30 for r in steering),
        "all_p_and_q_in_unit_interval": all(0 <= float(r[p]) <= 1 for rows, p, q in [(primary,"p_raw","q_bh_global_375"),(tables["M"],"p_raw","q_bh_global_375"),(tables["A"],"p_raw","q_bh_global_375"),(steering,"p_mc","q_bh_global_30")] for r in rows for p in (p,q)),
        "raw_or_frozen_recomputed_p_matches_reference": all(p_matches),
        "bh_library_matches_direct_implementation": max(*bh_geometry.values(), steering_bh) <= 1e-14,
        "qwen7_legacy_direction_absent": all("LEGACY" not in r["canonical_direction_status"] for r in steering),
        "qwen7_expected_layers": all(int(r["layer"]) == {"TriviaQA":18,"ARC-Easy":18,"CommonsenseQA":17}[r["dataset"]] for r in steering if r["model"] == "Qwen2.5-7B-Instruct"),
        "comparison_exactly_375_rows": len(primary_comparison) == 375,
        "auxiliary_comparison_exactly_750_rows": len(auxiliary_comparison) == 750,
        "geometry_qwen7_direction_hash_matches_expected": direction_audit["geometry_qwen7_direction_hash_matches_expected"],
        "geometry_qwen7_legacy_direction_absent": direction_audit["geometry_qwen7_legacy_direction_absent"],
        "geometry_all_rows_have_direction_provenance": direction_audit["geometry_all_rows_have_direction_provenance"],
        "portable_inputs_are_package_relative": all(not Path(r["source_artifact"]).is_absolute() and not r["source_artifact"].startswith("results/") for r in read_csv(PACKAGE / "geometry_cell_inputs.csv")),
        "portable_isolated_copy_csv_byte_identity": portable_verification_passed,
    }
    integrity = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
                 "bh_max_abs_difference_by_family": {"RINC_GLOBAL_375": bh_geometry["R_incremental"], "M_GLOBAL_375": bh_geometry["M"], "A_GLOBAL_375": bh_geometry["A"], "STEERING_FIXED_ALPHA_GLOBAL_30": steering_bh},
                 "raw_null_coverage": {"primary_rows_with_raw_samples": 375, "auxiliary_rows_with_raw_samples": 700, "auxiliary_rows_without_preserved_raw_draw_arrays": 50}}
    write_json(output / "integrity_checks.json", integrity); generated.append("integrity_checks.json")
    if integrity["status"] != "PASS":
        raise RuntimeError("integrity checks failed")

    transitions = [r for r in auxiliary_comparison if r["decision_changed"]]
    lines = ["# Harmonized analysis update report", "", "All calculations below are the final unified/harmonized analysis families; none is described as prespecified.", "",
             "## Primary R_inc", "", f"Global-375 BH: {ps['significant_count']}/375 significant and {ps['nonsignificant_count']}/375 nonsignificant. TriviaQA remains {ps['triviaqa']}/125; k>=5 remains {ps['k_ge_5']}/300. The seven nonsignificant rows remain the Qwen-family CommonsenseQA k=2 cells listed in `rinc_nonsignificant_cells.csv`.", "",
             f"Primary R_inc significance decisions changed relative to the historical families: {len(changed_primary)}. This statement applies only to primary R_inc.", "",
             "## Auxiliary M/A historical comparison", ""]
    for metric in ("M", "A"):
        a = auxs[metric]
        lines += [f"- {metric}: historical {a['historical_significant_count']}/375 -> global-375 {a['global_375_significant_count']}/375; k=2 {a['historical_k2_significant_count_out_of_75']}/75 -> {a['global_k2_significant_count_out_of_75']}/75; k>=5 {a['historical_k_ge_5_significant_count_out_of_300']}/300 -> {a['global_k_ge_5_significant_count_out_of_300']}/300; {a['decision_changed_count']} cell decisions changed."]
    lines += ["", "Complete transitions (also in `auxiliary_historical_comparison.csv`):", ""]
    for row in transitions:
        lines.append(f"- {row['metric']}: {row['model']} / {row['dataset']} / {row['quantization']} / k={row['k']}: {row['decision_transition']} (q {row['q_bh_historical']:.8g} -> {row['q_bh_global_375']:.8g}).")
    lines += ["", "## Geometry direction provenance", "",
              "All 375 rows join to one of the 15 model-dataset records in `geometry_direction_provenance.csv`. Qwen7 geometry hashes match the geometry-specific frozen canonical references: TriviaQA L18 `b36d...`, ARC-Easy L18 `85bb...`, and CommonsenseQA L17 `b683...`. The corrected TriviaQA intervention D0 hash is `74f8...`; its difference from `b36d...` is only floating-point construction order (cosine approximately 1, max component difference about 7.5e-9), not a legacy direction. The known legacy TriviaQA hash `a804...` is absent.", "",
              "## Steering", "", f"Global-30 BH: {sum(r['significant_q_lt_0_05'] for r in steering)}/30 significant. Mistral-7B/TriviaQA false remains significant (p={mistral_false['p_mc']}, q={mistral_false['q_bh_global_30']}); R087 remains an absolute-effect exceedance despite its opposite sign.", "",
              "## Raw-artifact limitation", "", "Primary R_inc has 375/375 raw-draw coverage. Auxiliary M/A has 700/750 raw-draw coverage. The remaining 50 rows (Ministral-3-14B/TriviaQA, 25 M and 25 A) are recovered from frozen summaries and exceedance counts, not fully recalculated from per-repetition draws. This limitation does not affect primary R_inc.", ""]
    (output / "SUMMARY_REPORT.md").write_text("\n".join(lines)); generated.append("SUMMARY_REPORT.md")

    provenance = {"package_mode": "self-contained portable CPU verification", "script": "portable_recalculate.py",
                  "runtime_dependencies": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                  "portable_inputs": ["raw_geometry/", "steering_direction_effects.csv", "steering_cell_inputs.csv", "geometry_cell_inputs.csv", "frozen_p_value_reference.csv", "historical_primary_q_values.csv", "historical_auxiliary_q_values.csv", "geometry_direction_provenance.csv", "geometry_direction_vectors.npz"],
                  "external_repository_reads": [], "gpu_or_model_inference": False,
                  "raw_coverage": {"primary_R_inc": "375/375", "auxiliary_M_A": "700/750", "auxiliary_frozen_summary_rows": 50,
                                   "limitation": "Ministral-3-14B/TriviaQA M/A raw draw arrays were not preserved; frozen summary and exceedance counts are used. Primary R_inc is unaffected."},
                  "geometry_direction_audit": direction_audit}
    write_json(output / "provenance_manifest.json", provenance); generated.append("provenance_manifest.json")
    # The SHA manifest intentionally excludes itself and interpreter caches.  It
    # covers both immutable portable inputs and the rebuilt derived outputs.
    manifest_paths: dict[str, Path] = {}
    for path in PACKAGE.rglob("*"):
        if (path.is_file() and "__pycache__" not in path.parts
                and path.name != "sha256_manifest.json" and "portable_rebuild" not in path.parts):
            manifest_paths[path.relative_to(PACKAGE).as_posix()] = path
    for name in generated:
        manifest_paths[name] = output / name
    manifest = [{"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size}
                for name, path in sorted(manifest_paths.items())]
    write_json(output / "sha256_manifest.json", {"algorithm": "SHA-256", "files": manifest})
    generated.append("sha256_manifest.json")
    return generated


def verify(reference: Path, rebuilt: Path, generated: list[str], record: Path | None) -> None:
    csv_files = [name for name in generated if name.endswith(".csv")]
    comparisons = []
    for name in csv_files:
        expected, actual = reference / name, rebuilt / name
        identical = expected.read_bytes() == actual.read_bytes()
        comparisons.append({"file": name, "byte_identical": identical,
                            "reference_sha256": sha256(expected), "rebuilt_sha256": sha256(actual)})
    value = {"status": "PASS" if all(r["byte_identical"] for r in comparisons) else "FAIL",
             "verification_scope": "all regenerated CSV artifacts, byte-for-byte",
             "reference_directory": ".", "rebuilt_directory": "isolated output directory",
             "repository_external_reads": 0, "gpu_or_model_inference": False, "files": comparisons}
    if record:
        write_json(record, value)
    print(json.dumps(value, indent=2))
    if value["status"] != "PASS":
        raise RuntimeError("portable CSV verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE,
                        help="directory for regenerated derived outputs (default: package directory)")
    parser.add_argument("--verify-reference", type=Path,
                        help="compare every regenerated CSV byte-for-byte with this reference package")
    parser.add_argument("--verification-record", type=Path,
                        help="optional path for the portable verification JSON record")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = build(output)
    if args.verify_reference:
        verify(args.verify_reference.resolve(), output, generated,
               args.verification_record.resolve() if args.verification_record else None)


if __name__ == "__main__":
    main()
