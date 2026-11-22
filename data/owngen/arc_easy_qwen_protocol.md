# ARC-Easy Qwen own-generation main experiment

This main experiment is pre-specified before generation. It evaluates the
corrected Qwen2.5-7B L18 ARC-Easy CCS direction on independently generated
closed-book answers. The two earlier 100-question pilots (seeds 42 and 43)
are excluded by question ID.

The deterministic manifest selects 500 remaining ARC-Easy train and 500
remaining test questions (seed 20260725), then makes two computational shards
of 250 train and 250 test questions for GPU 0 and GPU 1. Each question has 32
samples, temperature 0.8, top-p 0.95, and at most 24 generated tokens. The
RNG seed is set once per shard. Labels use normalized exact alias matching;
span matching is disabled for this main result. No generated result is used to
change these settings.

Hidden states are extracted at Qwen layer 18 with the pre-registered
`external_user` answer context. The primary outcome is question-clustered
bootstrap AUROC of the fixed ARC-Easy CCS direction on generated-answer last
tokens in the independent test subset. Pre-answer Spearman and train results
are secondary diagnostics. The two shards are concatenated only after
generation and ID-overlap validation.
