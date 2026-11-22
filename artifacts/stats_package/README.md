# Portable harmonized analysis package

This directory is a self-contained, machine-readable package for the final unified/harmonized analysis families. It can be copied or moved without the original repository. No GPU, model weights, inference, or network access is required.

## Run from the package directory

```bash
cd harmonized_analysis_families_2026-08-03
python portable_recalculate.py
```

`recalculate_harmonized_analysis.py` is a backward-compatible wrapper for the same portable entry point. All input paths are resolved from the script location, not the current working directory or repository root.

To rebuild into a separate directory and compare every regenerated CSV byte-for-byte with the package reference:

```bash
python portable_recalculate.py \
  --output-dir ./portable_rebuild \
  --verify-reference . \
  --verification-record ./portable_verification.json
```

The completed isolated-copy test is recorded in `portable_verification.json`.

## Portable inputs

- `raw_geometry/`: the 75 raw cell artifacts covering all 375 geometry cells; every primary R_inc row has 1,000 saved null draws.
- `steering_direction_effects.csv`: 127 signed random-direction effects for each of the 30 steering cells (3,810 rows).
- `steering_cell_inputs.csv`: observed canonical MD effects and fixed cell metadata; no empirical p/q values are taken from it.
- `geometry_cell_inputs.csv`: package-relative index for the 75 geometry artifacts.
- `geometry_direction_provenance.csv` and `geometry_direction_vectors.npz`: 15 model-dataset provenance records and portable Qwen7 vector-level hash evidence.
- `frozen_p_value_reference.csv`: independent frozen p-value reference used for integrity comparisons.
- `historical_primary_q_values.csv` and `historical_auxiliary_q_values.csv`: historical q-values used only for comparison, never for final global decisions.

## Families

- `RINC_GLOBAL_375`: 375 primary R_inc tests.
- `M_GLOBAL_375`: 375 auxiliary M tests.
- `A_GLOBAL_375`: 375 auxiliary A tests.
- `STEERING_FIXED_ALPHA_GLOBAL_30`: 30 fixed-alpha directional-specificity tests.

Each final q-vector is recalculated from its complete raw-p vector. BH uses `scipy.stats.false_discovery_control(method="bh")` and is cross-checked against an independent direct implementation.

## Dependencies and resources

- Python 3.11.14
- NumPy 2.4.3
- SciPy 1.17.1
- CPU only; typical runtime is about 1–2 seconds on this server.

The portable script itself imports only the Python standard library, NumPy, and SciPy. `safetensors` is not required for package verification.

## Historical comparisons

`previous_family_comparison.csv` applies only to primary R_inc; its significance-decision change count is 0.

`auxiliary_historical_comparison.csv` contains all 750 M/A historical-vs-global rows. For each metric, the historical result is 270/375 significant and global-375 is 268/375; k=2 changes 29/75 to 28/75 and k>=5 changes 241/300 to 240/300. Each metric has four cell-level decision transitions, all listed in the CSV and `SUMMARY_REPORT.md`.

## Geometry direction audit

All 375 rows join to the 15 records in `geometry_direction_provenance.csv`. The Qwen2.5-7B geometry axes pass content-hash comparisons against geometry-specific canonical references at TriviaQA L18, ARC-Easy L18, and CommonsenseQA L17. The known legacy TriviaQA hash is absent. See the report for why the corrected TriviaQA intervention D0 and geometry-axis hashes differ by harmless floating-point construction order despite cosine approximately 1.

## Raw-artifact limitation

- Primary R_inc: 375/375 rows have per-repetition raw draws.
- Auxiliary M/A: 700/750 rows have per-repetition raw draws.
- The remaining 50 rows are Ministral-3-14B-Instruct × TriviaQA M/A and are recovered from frozen observed/null summaries and exceedance counts.
- Those 50 rows are not described as fully recalculated from raw draws. This limitation does not affect primary R_inc.

## Key results

- Primary R_inc: 368/375 significant.
- Auxiliary M and A: 268/375 significant each.
- Fixed-alpha steering: 30/30 significant.
- Mistral-7B–TriviaQA false side: p=0.015625, q=0.015625.
