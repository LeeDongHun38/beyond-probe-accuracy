# Data card

All data used in the paper. Paths below are relative to `data/`.

## Answer-conditional contrastive pairs (Sec. 3.1, 4; App. A.1, Table 3)

| Dataset | Path | Pairs | Train | Held-out | Incorrect answer `a-` |
|---|---|---:|---:|---:|---|
| TriviaQA (`rc.nocontext` validation, 5,000 sampled with seed 42) | `pairs/triviaqa/` | 4,999 | 2,499 | 2,500 | Generated with GPT-4.1 |
| ARC-Easy (all official splits) | `pairs/arc_easy/` | 5,197 | 2,598 | 2,599 | One of three provided distractors (seed 42) |
| CommonsenseQA (official train + validation) | `pairs/commonsenseqa/` | 10,962 | 5,481 | 5,481 | One of four provided distractors (seed 42) |

Each question contributes one pair and appears in exactly one split (`split` = `train` or `test`;
`test` is the held-out split). Each dataset folder has `all.json`, `train.json`, `test.json`
and `stats.json` (counts and mean answer lengths).

Record fields:

- `id`, `question`, `true_answer` (a+), `false_answer` (a-), `ground_truth_aliases`
- `split`, `template_id` (prompt template used for the template control, App. D.1)
- TriviaQA only: `llama_fp16` / `qwen_fp16` hold the models' own greedy answers recorded when the
  pairs were built. Correctness labels for the pair analyses come from `true_answer` / `false_answer`.

## Gold-donor source/style control (App. D.2, Table 21)

`controls/gold_donor/` replaces each GPT-4.1 TriviaQA distractor with a gold answer
from another question (same train/held-out split). `source_matching_audit.json` records donor
selection; `manual_review_sample.csv` is the 100-pair semantic audit sample.

## Self-generated answers (App. D.3-D.5, Tables 22-24)

`owngen/` holds short answers sampled from each model under the five-shot generation prompt of
App. A.2 (temperature 0.8, top-p 0.95, at most 24 new tokens), with one greedy answer per
question. Each file is `{"metadata": ..., "records": [...]}`; a record keeps the question `id`,
`split`, `template_id`, gold aliases, the labeled `samples` (`label` = correct / incorrect /
excluded, `match_type`), and `p_correct` = correct / (correct + incorrect).

| File | Model | Questions | Samples | Used in |
|---|---|---|---|---|
| `triviaqa_{llama,mistral,qwen}.json` | each 7-8B model | all 4,999 TriviaQA pair questions (held-out half evaluated) | 16, seed 42 | D.3, D.4 |
| `arc_easy_qwen.json` | Qwen2.5-7B | 1,000 ARC-Easy questions (`arc_easy_qwen_questions.json`) | 32, seeds 20260725 / 20260726 (two shards) | D.3, D.4 |
| `span_control_{llama,mistral,qwen}.json` | each 7-8B model | 2,500 TriviaQA questions disjoint from the pair data (`span_control_questions.json`) | 16, seed 20260728 | D.5 |

- TriviaQA answer files were labeled with span matching enabled (`metadata.allow_span_match`);
  the readouts use exact-match positives only where the protocol requires it (D.5).
- `arc_easy_qwen_questions.json` lists the selected questions in shard order and the 200 pilot
  question ids that were excluded before selection (`probedir.controls.select_arc_owngen_questions`
  reproduces the selection). `arc_easy_qwen_protocol.md` is the note written before generation.
- `span_control_questions.json` keeps the original question ids.

## Pair-construction inputs

`raw/triviaqa_sample5000.json` is the 5,000-question TriviaQA `rc.nocontext` validation sample
(seed 42) from which the TriviaQA pairs were built, and `raw/triviaqa_greedy_{llama,qwen}.json`
are the models' greedy answers merged into the pair records (`probedir pairs`). They are only
needed to rebuild the pairs.

## Licenses

Derived data inherit the licenses of their sources; see `../THIRD_PARTY_NOTICES.md`.
In particular, ARC-derived files are CC BY-SA 4.0 (share-alike).
