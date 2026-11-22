# Paper figures

Regenerate from the repository root (CPU only, needs `numpy`, `pandas`, `matplotlib`):

```bash
python paper/make_figures.py
```

The original plotting code and figure files were not preserved, so these
figures are reconstructions from the released artifacts. Every plotted value is
read from `artifacts/`; the script asserts the annotations printed in the
paper (the five `r` values in Figure 2, "29/30 above equality" and the single
Mistral-7B–TriviaQA false-side exception in Figure 3, the 5-of-6 range
reduction in Figure 3(b), and positive, BH-significant excess for all 125
TriviaQA cells in Figure 4). The data points were checked visually against the
published PDF; typography and spacing are close but not pixel-identical.

| File | Paper | Inputs |
| --- | --- | --- |
| `fig2_estimator_paths.{pdf,png}` | Figure 2 | `paper_sections/C2_estimator_sweep/d0_d5_canonical.csv` (D0–D3, 7–8B), `raw_lr_sweep_completed.csv` (14B); residual norms from `sources/results/gpu_queue_explicit_system_v1/job1_e8_with/analysis/with_residual_norm_medians.json` (7–8B) and `sources/results/canonical_residual_norms_15cell/` (14B) |
| `fig3_specificity_and_magnitude.{pdf,png}` | Figure 3 | `paper_sections/C4_random_direction_null/c4_random_direction_summary.csv`, `paper_sections/C3_intervention/c3_full_intervention_summary.csv`, corrected Qwen-7B TriviaQA effects from `c3_qwen_corrected_triviaqa_by_seed.csv` and `sources/results/qwen7b_corrected_l18_c3c4_rerun_v1/analysis/summary.json` |
| `fig4_quant_error_excess.{pdf,png}` | Figure 4 | `stats_package/rinc_global_375.csv` (TriviaQA rows; excess = `observed_R_inc − null_mean_R_inc`) |

Figure 1 (the Ottawa/Toronto question-matched pair) is a text diagram typeset
in LaTeX and is not generated here.

## Source-selection notes (reproduce the published figures exactly)

- **Qwen2.5-7B TriviaQA.** The C3/C4 summary CSVs still list the legacy
  Qwen-7B TriviaQA MD effect (+.126 / −.178). The published figures use the
  offset-corrected re-execution (+.0848 / −.1004 at α=4; +.1440 / −.2345 at
  r=.10), so the script substitutes those values. The random-direction null
  maxima (.0186 / .0203) are identical in both runs.
- **Mistral-7B TriviaQA, two prompt protocols.** Figure 2 and Figure 3(a) use
  the explicit-system-prompt run (MD ΔP = +.419 / −.910; r = 4/4.7318 = .845).
  Figure 3(b) uses the C3 fixed-α values, which for this cell come from the
  legacy no-system-prompt run (+.473 / −.880; norm 4.6787, r = .855). This
  mirrors the protocol mix documented for Tables 13/15/17/19 and is reproduced
  as published, not corrected.
- `d0_d5_canonical.csv` contains two rows (Qwen D4/D5) whose free-text `notes`
  field has an unquoted comma; the script reads it with a tolerant parser.
