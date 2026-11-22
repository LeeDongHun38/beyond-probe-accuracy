# Published mean-difference directions

The exact unit-normalized MD direction vectors used in the paper's quantization analyses,
one per model-dataset cell (FP16 training split, selected layer):

- `<model>__<dataset>__transfer.npy`: Table 2, App. E.1 and E.3 (cross-precision AUROC, pair margins)
- `<model>__<dataset>__geometry.npy`: Fig. 4, App. E.2-E.3 (error-geometry subspace and null)

Pass one with `--direction` to reproduce the published values exactly, e.g.

    probedir geometry --model llama-3.1-8b --dataset triviaqa \
        --direction artifacts/directions/llama-3.1-8b__triviaqa__geometry.npy

Without `--direction`, the commands re-estimate the MD direction from the cached hidden
states in float64.
