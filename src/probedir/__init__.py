"""probedir: evaluating linear probe directions beyond predictive accuracy.

Modules
-------
config     experiment configuration (configs/experiment.yaml)
io         hidden-state cache format
stats      AUROC, empirical p-values, Benjamini-Hochberg, bootstrap
data       answer-conditional contrastive pairs
prompts    exact prompt formats and chat templating
models     FP16 / quantized model loading and simulated quantization
extract    answer-final hidden-state extraction
probes     mean-difference and logistic-regression directions, layer selection, split-half stability
steering   activation steering and the P(Yes) correctness-judgment readout
nulls      norm-matched isotropic random-direction nulls
transfer   fixed-direction cross-precision transfer (AUROC, pair margins)
geometry   differential quantization-error geometry and activation-matched null subspaces
controls   validity controls (App. D)
cli        command-line entry point
"""

__version__ = "1.0.0"
