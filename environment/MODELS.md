# Model checkpoints

No weights are redistributed. Download each checkpoint at the pinned revision.
`probedir.models.checkpoint` loads `$PROBEDIR_MODELS/<directory>` (default `./models/`)
when that directory exists, and otherwise the Hugging Face ID (hub cache).

## Reference (FP16) models

| Paper name | Hugging Face ID | Revision | Local directory (optional) |
|---|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | — |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | — |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | `c170c708c41dac9275d15a8fff4eca08d52bab71` | — |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | `models/Qwen2.5-14B-Instruct` |
| Ministral-3-14B-Instruct-2512 | `mistralai/Ministral-3-14B-Instruct-2512-BF16` | `3cea74c1ebaf5ce5f5a2553de470e2ceab825142` | `models/Ministral-3-14B-Instruct-2512-BF16` (BF16 weights cast to FP16) |

## Quantized conditions

AWQ and GPTQ use public pre-quantized checkpoints (downloaded, not produced locally).
RTN W4A16, W8A8 and W4A8 are simulated in-process from the FP16 model by the
extraction code (symmetric per-group weight quantization, group size 128;
W*A8 adds per-token dynamic INT8 activation quantize-dequantize on each
transformer-block output; no SmoothQuant smoothing).

| Model | AWQ W4A16 checkpoint (revision) | GPTQ W4A16 checkpoint (revision) |
|---|---|---|
| Llama-3.1-8B | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (`db1f81ad4b8c7e39777509fac66c652eb0a52f91`) | `hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4` (`f184d2b18e2b7c257af79aa0bc53ef3ee505caa2`) |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B-Instruct-AWQ` (`b25037543e9394b818fdfca67ab2a00ecc7dd641`) | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` (`e9c932ac1893a49ae0fc497ad6e1e86e2e39af20`) |
| Mistral-7B | `solidrust/Mistral-7B-Instruct-v0.3-AWQ` (`95b1295ddd1a8673117cdc7bd2a4da2a457bb3f7`) | `thesven/Mistral-7B-Instruct-v0.3-GPTQ` (`2d24a7f41b3d93fba3aef0cde15a77f09e99437c`) |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B-Instruct-AWQ` (`539535859b135b0244c91f3e59816150c8056698`) -> `models/Qwen2.5-14B-Instruct-AWQ` | `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` (`08fdbde5eea15cd2aae884125fbfd604dec19818`) -> `models/Qwen2.5-14B-Instruct-GPTQ-Int4` |
| Ministral-3-14B | `cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit` (`d4401438aba6c4e523b929c1a8878d3fe8cc0bca`, group size 32) -> `models/Ministral-3-14B-Instruct-2512-AWQ-4bit` | `inference-optimization/Ministral-3-14B-Instruct-2512.w4a16` (`7933499c9816b90b7c337f4c5ffb494f0927b588`, group size 128) -> `models/Ministral-3-14B-Instruct-2512-GPTQ-W4A16` |

For Ministral-3-14B the AWQ/GPTQ weights are reconstructed from these public
checkpoints when the model is loaded (`probedir.models.load_model`; App. E.1).

## Other external model

TriviaQA incorrect answers were generated with OpenAI `gpt-4.1`
(`probedir pairs --dataset triviaqa --generate`, key read from `OPENAI_API_KEY`). The
generated pairs are released in `data/pairs/triviaqa/`; regeneration is not deterministic.

## Hardware

7-8B models were run on one GPU each and 14B models on two GPUs (model sharded with
`device_map="auto"`).
