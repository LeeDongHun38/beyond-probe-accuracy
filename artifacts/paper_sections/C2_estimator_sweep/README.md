# C.2 artifact index

- `d0_d5_canonical.csv`: canonical 7--8B TriviaQA D0--D5 table.  Llama and
  Mistral use the explicit-system Job-1 aggregation; Qwen uses the corrected
  reexecution.  Corrected-Qwen D4/D5 split-half values are independently
  recomputed and recorded in `qwen_corrected_d4_d5_split_half.{csv,json}`.
- `raw_lr_sweep_completed.csv`: every completed frozen raw-LR estimator in the
  2026-08-01 14B sweep.  Its actual grid is MD, 1e-4, 1e-2, and 1.
- `qwen_legacy_corrected.csv` and `provenance.json`: pre-correction versus
  corrected Qwen D0 comparison and definitions.
- `raw_lr_sweep_intervention_metadata.json`: intervention seeds, sample-ID
  alignment, prompt, alpha, and aggregation implementation for the frozen
  raw-LR sweep.

Participation ratio is `(sum_i v_i^2)^2 / sum_i v_i^4`.  Scores use correct
representations as class 1, directions are oriented from training data, and
held-out AUROC has neither sign flipping nor `max(AUC, 1-AUC)`.
