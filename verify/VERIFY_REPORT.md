# Camera-ready verification

The reference PDF has SHA-256 `207a6cce60fa0efef236cb32d5c9fa4f7b61b7fb28213b809a6b0797b6918c15`.

`verify_paper_numbers.py` rebuilds 22 numerical result tables in a temporary directory, checks the released exports and checks 2,091 printed numeric entries against the PDF transcriptions in `expected/`. The five entries in `known_discrepancies.csv` are display-precision differences in Table 21 on the original ARC-Easy and CommonsenseQA runs. Four template ranges were rounded to three decimal places and printed with a trailing zero. The Mistral ARC-Easy length gap is 0.0101499203801 in the source and .0102 in print. The exports retain source precision.

The same command checks the question split, layer choices, stored projection AUROCs, calibration thresholds and changed-decision counts. Table 18 rebuilds the 30-condition fixed-α family from 3,810 direction-by-side effects. Table 26 rebuilds the permutation p-values and BH families from all five arrays of 10,000 draws. GPU inference is a separate workflow.

**Table 24 bootstrap unit.** The caption calls the resampling unit an item, and Appendix D.4 defines an item as one held-out pair row. For the three TriviaQA intervals, the experiment resampled 2,203 normalized questions with replacement and carried all corresponding rows together. Repeating 1,000 draws with seed 42 from the [released pre-answer scores](../artifacts/controls/own_generation/triviaqa) and [question keys](../artifacts/probes/split_manifest.csv) reproduces all three printed intervals.

`check_checksums.py` checks every file listed in the release manifest and runs the numerical verification above. Use `--skip-stats` to check file hashes separately.
