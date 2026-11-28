# Beyond Probe Accuracy

**Estimation Stability, Intervention Specificity, and Representation Geometry of Probe Directions**

DongHun Lee and Misuk Kim · EACL 2027

[Final paper (PDF)](paper/Beyond%20Probe%20Accuracy.pdf) · [Numerical tables](paper/tables) · [Figures](paper/figures) · [Experiment artifacts](artifacts) · [Data card](data/DATA_CARD.md)

The paper compares probe estimators on the same model, layer and training representations. It measures how well each probe separates correct from incorrect answers, how consistently its direction can be estimated, and how its intervention effect compares with random directions. Five instruction-tuned models are evaluated on TriviaQA, ARC-Easy and CommonsenseQA.

| Analysis | Paper result |
|---|---|
| Direction stability (Tables 1 and 12) | On TriviaQA, MD split-half cosine ranges from .956 to .980. Scale-normalized LR has higher held-out AUROC and lower split-half cosine than MD in all 66 comparisons across 11 model–layer settings. The paired AUROC interval excludes zero in 64. |
| Intervention specificity (Figure 3, Tables 18–20) | At α = 4, MD exceeds all 127 isotropic directions in 29 of 30 model–dataset–side comparisons. One random direction exceeds MD on the Mistral-7B TriviaQA false side. All 30 tests pass BH correction. At matched dose r = .05, the result depends on the null family. |
| Quantization (Figure 4, Tables 2 and 25–28) | The largest TriviaQA AUROC drop is .036. At a fixed FP16 threshold, 2.34–14.30% of evaluation decisions change. In the 25-condition 14B geometry panel, the MD-axis statistic is significant in 25/25 settings and the incremental 49-component statistic in 0/25. |

## Rebuild and verify the results

From a source checkout with Python 3.10 or newer, run the following on CPU. The scripts read the released scores, model-level summaries and null draws.

```bash
python -m pip install -e '.[paper,dev]'
python paper/make_tables.py
python paper/make_figures.py
python verify/verify_paper_numbers.py
python verify/check_checksums.py --skip-stats
python -m pytest -q
```

The table builder exports the 22 numerical result tables using the numbering of the final PDF. Tables 4–8 describe prompts, examples, models and estimators. Table 10 defines the statistical families. The CSVs retain extra precision and use one row per experimental condition or side. The verifier checks them against the printed values, including the documented rounding differences in Table 21. [Verification details](verify/VERIFY_REPORT.md)

## Reuse the experiments

The final TriviaQA split assigns all rows with the same normalized question to the same partition. There are 2,508 training pairs and 2,491 held-out pairs, with 2,203 distinct questions on each side. Training-only layer selection gives L16 for Llama-3.1-8B, L18 for Qwen2.5-7B, L13 for Mistral-7B, L29 for Qwen2.5-14B and L36 for Ministral-3-14B. [Split manifest](artifacts/probes/split_manifest.csv) · [Configuration](configs/experiment.yaml)

The package provides model-level commands for pair construction, extraction, probe fitting, steering and quantization. For an inference run, install the GPU extras and obtain the relevant checkpoint. Quantized checkpoint loaders may require additional platform-specific dependencies. [Model revisions and environment](environment/MODELS.md)

```bash
python -m pip install -e '.[gpu]'
probedir extract --model llama-3.1-8b --dataset triviaqa
probedir probes --model llama-3.1-8b --dataset triviaqa
probedir estimator-sweep --model llama-3.1-8b --dataset triviaqa
probedir steer --model llama-3.1-8b --dataset triviaqa --alpha 4 --random-null
```

After extracting FP16 and the five quantized precisions for a cell, use `probedir transfer` for AUROC and margins. Use `probedir geometry-permutation` for the within-pair label-permutation experiment in Figure 4. The command refits MD and residual PCA for each permutation. The CPU table builder can reproduce the panel statistics directly from its stored draws.

## Repository layout

| Directory | Contents |
|---|---|
| [paper/](paper) | Manuscript PDF, numerical tables, figure files and the two scripts that rebuild them |
| [artifacts/](artifacts) | Fitted directions, held-out scores, intervention effects and permutation draws, indexed by paper table and figure |
| [src/probedir/](src/probedir) | Python implementation and CLI for data preparation, probe fitting, steering, transfer and controls |
| [data/](data) | Contrastive pairs and split assignments, generation records and gold-donor controls |
| [configs/](configs) | Model layers, quantization conditions, doses, seeds and statistical settings |
| [environment/](environment) | Model checkpoint identifiers and Python dependency versions for inference |
| [verify/](verify) | Printed reference values, release checks and numerical verification |
| [tests/](tests) | Unit tests for the analysis code |

## License

The code uses the MIT License. Data and derived artifacts retain their source terms. [License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
