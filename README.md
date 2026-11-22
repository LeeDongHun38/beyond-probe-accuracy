# Beyond Probe Accuracy: Estimation Stability, Intervention Specificity, and Representation Geometry of Probe Directions

Code, data, and results for the EACL 2026 paper.

Linear probes with similar held-out AUROC can point in very different directions. With the model,
dataset, layer, and training representations held fixed, we vary only the probe estimator and
evaluate each direction on three properties:

- **predictive separability:** held-out AUROC
- **estimation stability:** split-half cosine
- **directional intervention specificity:** steering against norm-matched random directions

We then transfer the FP16 mean-difference direction to quantized models and measure where the
differential quantization error concentrates.

**Setting:** 5 instruction-tuned models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B, Qwen2.5-14B,
Ministral-3-14B) × 3 QA datasets (TriviaQA, ARC-Easy, CommonsenseQA) × 5 quantization conditions
(AWQ, GPTQ, RTN W4A16; W8A8; W4A8).

## Installation

```bash
pip install -e .                 # core: numpy, scipy, scikit-learn, pyyaml, safetensors
pip install -e ".[gpu]"          # + torch, transformers (extraction, steering)
pip install -e ".[gpu,quant]"    # + AWQ / GPTQ loaders
pip install -e ".[paper]"        # + pandas, matplotlib (tables, figures)
```

## Reproducing the paper's tables and figures (CPU, under a minute)

All reported results are included under `artifacts/`, so the tables and figures can be rebuilt
without a GPU:

```bash
python paper/make_tables.py      # paper/tables/table01 ... table28
python paper/make_figures.py     # paper/figures/fig2, fig3, fig4
python verify/verify_paper_numbers.py   # compare every table cell and in-text number with the paper
python verify/check_checksums.py        # file hashes + exact rebuild of all multiple-testing families
```

## Running the experiments

Every stage is a `probedir` command. Fixed settings (models, selected layers, seeds, intervention
magnitudes) are in `configs/experiment.yaml`. Outputs go to `results/`.

```bash
# 0. Contrastive pairs (released in data/pairs/; this rebuilds them, Sec. 4, App. A.1)
probedir pairs --dataset arc_easy --out data/pairs/arc_easy

# 1. Hidden states at the last token of each proposed answer (Sec. 3.1)
probedir extract --model llama-3.1-8b --dataset triviaqa                    # FP16, all layers
probedir extract --model llama-3.1-8b --dataset triviaqa --precision awq    # selected layer

# 2. Layer selection, MD direction, split-half stability (Sec. 3.2-3.3; Tables 10, 12)
probedir probes --model llama-3.1-8b --dataset triviaqa

# 3. Estimator sweep: MD vs. L2 logistic regression (Table 1; Fig. 2)
probedir estimator-sweep --model llama-3.1-8b --dataset triviaqa

# 4. Bidirectional steering and random-direction specificity (Sec. 3.3-3.4; Fig. 3)
probedir steer --model llama-3.1-8b --dataset triviaqa --alpha 4 --random-null
probedir steer --model llama-3.1-8b --dataset triviaqa --r 0.10
probedir null-family

# 5. Quantization transfer and error geometry (Sec. 3.5; Table 2; Fig. 4)
probedir transfer --model llama-3.1-8b --dataset triviaqa \
    --direction artifacts/directions/llama-3.1-8b__triviaqa__transfer.npy
probedir geometry --model llama-3.1-8b --dataset triviaqa \
    --direction artifacts/directions/llama-3.1-8b__triviaqa__geometry.npy
probedir geometry-bh

# 6. Validity controls (App. D)
probedir template-control --model qwen2.5-14b --dataset triviaqa
probedir controls gold-donor
probedir controls readout --model llama-3.1-8b --dataset triviaqa \
    --answers data/owngen/triviaqa_llama.json --output results/controls/readout_llama.json
```

`transfer` and `geometry` cover all five quantization conditions of a cell, so extract each
`--precision` (awq, gptq, rtn, w8a8, w4a8) first. With
`--direction`, they use the exact direction vectors from the paper's runs
(`artifacts/directions/`); without it, they re-estimate the MD direction from the FP16 cache.
`probedir <command> --help` lists the options for each command. Models are loaded from the
Hugging Face Hub at the revisions in `environment/MODELS.md`. 7–8B models run on a single
24 GB GPU; the 14B models were run on two.

## Repository layout

| Path | Contents |
|---|---|
| `src/probedir/` | The package. One module per method component: `data`, `prompts`, `models`, `extract`, `probes`, `steering`, `nulls`, `transfer`, `geometry`, `controls`, `stats` |
| `configs/experiment.yaml` | Models, datasets, selected layers, seeds, intervention and geometry settings |
| `data/` | Contrastive pairs, gold-donor control, self-generated answers (`data/DATA_CARD.md`) |
| `artifacts/paper_sections/` | Results for each appendix section (C.1–E.3) that the tables and figures are built from |
| `artifacts/stats_package/` | Raw null draws and per-direction effects; recomputes every BH family |
| `artifacts/directions/` | MD direction vectors used for the quantization analyses |
| `artifacts/sources/` | Run summaries that the table builders and claim checks read |
| `paper/` | Table and figure builders and their outputs |
| `verify/` | Paper values and the verification scripts |
| `tests/` | Unit tests (`pytest`, CPU, no downloads) |
| `environment/` | Recorded package versions and model checkpoints |

## License

Code is released under the MIT License (`LICENSE`). Data and derived artifacts follow the licenses
of their sources (`THIRD_PARTY_NOTICES.md`). Model weights are not redistributed.
