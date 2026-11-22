# Third-party notices

The code in this repository is released under the license in `LICENSE`. Data and model
artifacts derived from third-party resources remain subject to their original licenses
and terms, listed below. Model weights are **not** redistributed; the code downloads them
from the Hugging Face Hub under their own licenses.

## Datasets

| Resource | Used for | License |
|---|---|---|
| TriviaQA (Joshi et al., 2017), `rc.nocontext` validation | `data/pairs/triviaqa/`, gold-donor control, own-generation | Apache License 2.0 |
| ARC-Easy (Clark et al., 2018) | `data/pairs/arc_easy/`, own-generation | CC BY-SA 4.0. Derived files are shared under the same license. |
| CommonsenseQA (Talmor et al., 2019) | `data/pairs/commonsenseqa/` | MIT License |

TriviaQA incorrect answers (`false_answer`) were generated with OpenAI GPT-4.1 and are
released as research data, in accordance with OpenAI's terms of use.

## Models (not redistributed)

| Checkpoint | License |
|---|---|
| meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 Community License |
| Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct | Apache License 2.0 |
| mistralai/Mistral-7B-Instruct-v0.3 | Apache License 2.0 |
| mistralai/Ministral-3-14B-Instruct-2512-BF16 | See the model card |

Quantized checkpoints, and the toolchains used to produce or simulate them, are listed in
`environment/MODELS.md` with their sources and licenses.

Derived artifacts in `artifacts/` (probe directions, per-cell statistics) are computed from
these models' hidden states. Using them is subject to each model's license, including the
Llama 3.1 Community License's attribution and acceptable-use terms.
