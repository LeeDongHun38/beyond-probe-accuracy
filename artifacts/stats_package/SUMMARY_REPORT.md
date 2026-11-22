# Harmonized analysis update report

All calculations below are the final unified/harmonized analysis families; none is described as prespecified.

## Primary R_inc

Global-375 BH: 368/375 significant and 7/375 nonsignificant. TriviaQA remains 125/125; k>=5 remains 300/300. The seven nonsignificant rows remain the Qwen-family CommonsenseQA k=2 cells listed in `rinc_nonsignificant_cells.csv`.

Primary R_inc significance decisions changed relative to the historical families: 0. This statement applies only to primary R_inc.

## Auxiliary M/A historical comparison

- M: historical 270/375 -> global-375 268/375; k=2 29/75 -> 28/75; k>=5 241/300 -> 240/300; 4 cell decisions changed.
- A: historical 270/375 -> global-375 268/375; k=2 29/75 -> 28/75; k>=5 241/300 -> 240/300; 4 cell decisions changed.

Complete transitions (also in `auxiliary_historical_comparison.csv`):

- M: Llama-3.1-8B-Instruct / TriviaQA / GPTQ W4A16 / k=2: significant_to_nonsignificant (q 0.049204527 -> 0.05950728).
- M: Llama-3.1-8B-Instruct / TriviaQA / simulated W4A8 / k=5: significant_to_nonsignificant (q 0.04995005 -> 0.061766191).
- M: Llama-3.1-8B-Instruct / CommonsenseQA / RTN W4A16 / k=5: nonsignificant_to_significant (q 0.053517911 -> 0.048924956).
- M: Ministral-3-14B-Instruct / CommonsenseQA / GPTQ W4A16 / k=5: significant_to_nonsignificant (q 0.046909612 -> 0.050135738).
- A: Llama-3.1-8B-Instruct / TriviaQA / GPTQ W4A16 / k=2: significant_to_nonsignificant (q 0.049204527 -> 0.05950728).
- A: Llama-3.1-8B-Instruct / TriviaQA / simulated W4A8 / k=5: significant_to_nonsignificant (q 0.04995005 -> 0.061766191).
- A: Llama-3.1-8B-Instruct / CommonsenseQA / RTN W4A16 / k=5: nonsignificant_to_significant (q 0.053517911 -> 0.048924956).
- A: Ministral-3-14B-Instruct / CommonsenseQA / GPTQ W4A16 / k=5: significant_to_nonsignificant (q 0.046909612 -> 0.050135738).

## Geometry direction provenance

All 375 rows join to one of the 15 model-dataset records in `geometry_direction_provenance.csv`. Qwen7 geometry hashes match the geometry-specific frozen canonical references: TriviaQA L18 `b36d...`, ARC-Easy L18 `85bb...`, and CommonsenseQA L17 `b683...`. The corrected TriviaQA intervention D0 hash is `74f8...`; its difference from `b36d...` is only floating-point construction order (cosine approximately 1, max component difference about 7.5e-9), not a legacy direction. The known legacy TriviaQA hash `a804...` is absent.

## Steering

Global-30 BH: 30/30 significant. Mistral-7B/TriviaQA false remains significant (p=0.015625, q=0.015625); R087 remains an absolute-effect exceedance despite its opposite sign.

## Raw-artifact limitation

Primary R_inc has 375/375 raw-draw coverage. Auxiliary M/A has 700/750 raw-draw coverage. The remaining 50 rows (Ministral-3-14B/TriviaQA, 25 M and 25 A) are recovered from frozen summaries and exceedance counts, not fully recalculated from per-repetition draws. This limitation does not affect primary R_inc.
