# Camera-ready verification

The reference PDF has SHA-256 `207a6cce60fa0efef236cb32d5c9fa4f7b61b7fb28213b809a6b0797b6918c15`.

`verify_paper_numbers.py` rebuilds 22 numerical result tables in a temporary directory, checks the released exports and checks 2,091 printed numeric entries against the PDF transcriptions in `expected/`. The five entries in `known_discrepancies.csv` are display-precision differences in Table 21 on the original ARC-Easy and CommonsenseQA runs. Four template ranges were rounded to three decimal places and printed with a trailing zero. The Mistral ARC-Easy length gap is 0.0101499203801 in the source and .0102 in print. The exports retain source precision.

The same command checks the question split, layer choices, stored projection AUROCs, calibration thresholds and changed-decision counts. Table 18 rebuilds the 30-condition fixed-α family from 3,810 direction-by-side effects. Table 26 rebuilds the permutation p-values and BH families from all five arrays of 10,000 draws. GPU inference is a separate workflow.

`check_checksums.py` validates the release manifest and runs the retained review-version statistics-package integrity check.
