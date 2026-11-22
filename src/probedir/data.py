"""Answer-conditional contrastive pairs (Sec. 3.1, Sec. 4, App. A.1).

Each question q_i contributes one pair: a correct answer a+ and a plausible
incorrect answer a-. Pairs are split 50:50 into training and held-out sets at the
question level, and each pair is assigned one of three pair-prompt templates.

* TriviaQA (rc.nocontext validation, 5,000 sampled with seed 42): a- is generated
  by GPT-4.1 (:func:`generate_triviaqa_distractors`).
* ARC-Easy (all official splits) and CommonsenseQA (train + validation): a- is one
  of the provided distractors, chosen with seed 42.
* Gold-donor control (App. D.2): each TriviaQA a- is replaced by the gold answer of
  another question in the same split (:func:`build_gold_donor_control`).

Released pair files: ``data/pairs/<dataset>/{all,train,test}.json``.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .config import pairs_path

SPLIT_SEED = 42
N_TEMPLATES = 3
GOLD_DONOR_SEED = 20260723
GOLD_DONOR_VERSION = "triviaqa_gold_source_matched_v1_2026-07-23"

Pair = dict[str, Any]


# --------------------------------------------------------------------------- loading

def load_pairs(dataset: str, split: str = "all") -> list[Pair]:
    """Released pairs in canonical order; ``split`` is ``all``, ``train`` or ``test``."""
    return json.loads(Path(pairs_path(dataset, split)).read_text())


def save_pairs(pairs: list[Pair], out_dir: Path, stats: dict[str, Any] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "all.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2))
    for split in ("train", "test"):
        subset = [p for p in pairs if p["split"] == split]
        (out_dir / f"{split}.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2))
    if stats is not None:
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- split and templates

def assign_split_and_templates(pairs: list[Pair], seed: int = SPLIT_SEED,
                               train_ratio: float = 0.5) -> list[Pair]:
    """Question-level 50:50 split and uniform template assignment (in place).

    The RNG call sequence matches the original construction: the split shuffle
    uses ``random.seed(seed)``; templates use ``random.seed(seed + 1)``.
    """
    random.seed(seed)
    _split(pairs, train_ratio)
    _templates(pairs, seed)
    return pairs


def _split(pairs: list[Pair], train_ratio: float) -> None:
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    train = set(indices[:int(len(pairs) * train_ratio)])
    for i, pair in enumerate(pairs):
        pair["split"] = "train" if i in train else "test"


def _templates(pairs: list[Pair], seed: int) -> None:
    random.seed(seed + 1)
    for pair in pairs:
        pair["template_id"] = random.randint(0, N_TEMPLATES - 1)


# --------------------------------------------------------------------------- TriviaQA

DISTRACTOR_PROMPT = """You are helping construct a research dataset for studying truth representations in language models.

Given a trivia question and its correct answer, generate ONE plausible but WRONG answer that:
1. Is the SAME type/category as the correct answer (if correct is a person name, give a person name; if a year, give a year; if a city, give a city)
2. Is believable — someone who doesn't know the answer might think it's correct
3. Is clearly factually wrong (not an alternative correct answer or alias)
4. Is similar in length to the correct answer

Question: {question}
Correct answer: {correct_answer}

Respond with ONLY the wrong answer, nothing else. No quotes, no explanation."""


def triviaqa_records(sample: list[dict], greedy: dict[str, list[dict]] | None = None) -> list[dict]:
    """Normalize the 5,000-question TriviaQA sample (``data/raw/triviaqa_sample5000.json``).

    ``greedy`` maps a model name ("llama", "qwen") to its greedy-answer records
    (``data/raw/triviaqa_greedy_<model>.json``); they are carried into the pair
    records as ``<model>_fp16`` fields, aligned by id (or by position).
    """
    records = {}
    for i, item in enumerate(sample):
        gt = item.get("ground_truth", item.get("answer"))
        if isinstance(gt, dict):
            gt = gt.get("value", gt.get("aliases", [str(gt)]))
        if isinstance(gt, list) and gt:
            primary, all_ = str(gt[0]), [str(a) for a in gt]
        elif isinstance(gt, str):
            primary, all_ = gt, [gt]
        else:
            primary = str(gt) if gt is not None else None
            all_ = [primary] if primary else []
        qid = item.get("id", item.get("_key", f"q_{i:04d}"))
        # The released pairs carry the gold answer field only (``[primary]`` for the
        # string-valued sample); the sample's separate ``answer_aliases`` lists are not used.
        records[qid] = {"id": qid, "question": item["question"], "ground_truth": primary,
                        "ground_truth_aliases": all_}
    for model, items in (greedy or {}).items():
        keys = list(records)
        for i, item in enumerate(items):
            qid = item.get("id", item.get("_key", f"q_{i:04d}"))
            if qid not in records:
                if i >= len(records):
                    continue
                qid = keys[i]
            sfx = f"{model}_fp16"
            prob = item.get(f"first_token_prob_{sfx}")
            records[qid][f"{model}_answer"] = item.get(f"model_answer_{sfx}")
            records[qid][f"{model}_correct"] = item.get(f"correct_{sfx}")
            records[qid][f"{model}_prob"] = float(prob) if prob is not None else None
    return list(records.values())


def generate_triviaqa_distractors(records: Iterable[dict], client: Any, max_retries: int = 3,
                                  sleep: Callable[[float], None] = time.sleep) -> dict[str, str | None]:
    """Generate one plausible wrong answer per question with GPT-4.1 (App. A.2).

    ``client`` is an ``openai.OpenAI`` instance. Settings: model gpt-4.1,
    temperature 0.7, max_tokens 100, no system message. A response equal to the
    gold answer or an alias is retried; after ``max_retries`` the item is dropped.
    Generation is not deterministic, so the released pairs are canonical.
    """
    out: dict[str, str | None] = {}
    for record in records:
        correct = record["ground_truth"]
        wrong = None
        if correct:
            for attempt in range(max_retries):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4.1", max_tokens=100, temperature=0.7,
                        messages=[{"role": "user", "content": DISTRACTOR_PROMPT.format(
                            question=record["question"], correct_answer=correct)}])
                    wrong = resp.choices[0].message.content.strip()
                    aliases = [a.lower() for a in record.get("ground_truth_aliases", [correct])]
                    if wrong.lower() in aliases or wrong.lower() == correct.lower():
                        wrong = None
                        if attempt < max_retries - 1:
                            continue
                    break
                except Exception:  # network/API errors: back off and retry
                    if attempt < max_retries - 1:
                        sleep(2 ** attempt)
        out[record["id"]] = wrong
    return out


def build_triviaqa_pairs(records: list[dict], distractors: dict[str, str | None],
                         max_samples: int = 5000, seed: int = SPLIT_SEED) -> list[Pair]:
    """TriviaQA pairs from normalized records and generated distractors.

    Records without a gold answer or without a distractor are dropped
    (5,000 questions -> 4,999 pairs).
    """
    if len(records) > max_samples:
        random.seed(seed)
        records = random.sample(records, max_samples)
    pairs = []
    for r in records:
        if not r.get("ground_truth") or not distractors.get(r["id"]):
            continue
        pairs.append({
            "id": r["id"], "question": r["question"], "true_answer": r["ground_truth"],
            "false_answer": distractors[r["id"]],
            "ground_truth_aliases": r.get("ground_truth_aliases", []),
            "llama_fp16": {"answer": r.get("llama_answer"), "correct": r.get("llama_correct"),
                           "prob": r.get("llama_prob")},
            "qwen_fp16": {"answer": r.get("qwen_answer"), "correct": r.get("qwen_correct"),
                          "prob": r.get("qwen_prob")},
        })
    return assign_split_and_templates(pairs, seed)


def triviaqa_stats(pairs: list[Pair]) -> dict[str, Any]:
    n = len(pairs)
    return {"total_pairs": n,
            "train": sum(p["split"] == "train" for p in pairs),
            "test": sum(p["split"] == "test" for p in pairs),
            "avg_question_len": sum(len(p["question"].split()) for p in pairs) / n,
            "avg_true_answer_len": sum(len(p["true_answer"].split()) for p in pairs) / n,
            "avg_false_answer_len": sum(len(p["false_answer"].split()) for p in pairs) / n}


# --------------------------------------------------------------------------- ARC-Easy / CommonsenseQA

MC_SOURCES = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", ("train", "test", "validation"), "arc"),
    "commonsenseqa": ("tau/commonsense_qa", None, ("train", "validation"), "csqa"),
}


def load_mc_records(dataset: str) -> list[dict]:
    """All multiple-choice items of ARC-Easy or CommonsenseQA, in official split order."""
    from datasets import load_dataset

    repo, config, splits, prefix = MC_SOURCES[dataset]
    records = []
    for split in splits:
        ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
        for i, item in enumerate(ds):
            labels, texts = item["choices"]["label"], item["choices"]["text"]
            if item["answerKey"] not in labels:
                continue
            k = labels.index(item["answerKey"])
            records.append({"id": f"{prefix}_{split}_{i:05d}", "question": item["question"],
                            "correct_answer": texts[k],
                            "wrong_answers": [t for j, t in enumerate(texts) if j != k],
                            "source_dataset": dataset})
    return records


def build_mc_pairs(records: list[dict], seed: int = SPLIT_SEED) -> list[Pair]:
    """One provided distractor per item (``random.choice``), then split and templates.

    The RNG state is shared: ``random.seed(seed)`` precedes the distractor choices,
    and the split shuffle continues from the same stream (as in the original runs).
    """
    random.seed(seed)
    pairs = []
    for r in records:
        pairs.append({"id": r["id"], "question": r["question"], "true_answer": r["correct_answer"],
                      "false_answer": random.choice(r["wrong_answers"]),
                      "ground_truth_aliases": [r["correct_answer"]],
                      "source_dataset": r["source_dataset"]})
    _split(pairs, 0.5)
    _templates(pairs, seed)
    return pairs


def mc_stats(pairs: list[Pair], dataset: str) -> dict[str, Any]:
    n = len(pairs)
    return {"dataset_name": dataset, "total_pairs": n,
            "train": sum(p["split"] == "train" for p in pairs),
            "test": sum(p["split"] == "test" for p in pairs),
            "avg_true_len": sum(len(p["true_answer"].split()) for p in pairs) / n,
            "avg_false_len": sum(len(p["false_answer"].split()) for p in pairs) / n}


# --------------------------------------------------------------------------- gold-donor control (App. D.2)

_WORD = re.compile(r"\S+")
_YEAR = re.compile(r"^\d{4}$")
_NUMBER = re.compile(r"^[\d,.]+(?:st|nd|rd|th|%)?$")


def _norm(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def answer_profile(value: str) -> dict[str, Any]:
    """Surface form of an answer: number class, letter case, length, punctuation."""
    text = value.strip()
    if _YEAR.fullmatch(text):
        number = "year"
    elif _NUMBER.fullmatch(text):
        number = "numeric"
    elif any(c.isdigit() for c in text):
        number = "contains_digit"
    else:
        number = "none"
    letters = "".join(c for c in text if c.isalpha())
    if letters and letters.isupper():
        case = "upper"
    elif letters and letters.islower():
        case = "lower"
    elif text.istitle():
        case = "title"
    else:
        case = "mixed"
    return {"tokens": len(_WORD.findall(text)), "chars": len(text), "number_kind": number,
            "case_kind": case, "has_ampersand": "&" in text, "has_hyphen": "-" in text,
            "has_comma": "," in text}


def match_cost(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Form-mismatch cost between two answer profiles (lower is better)."""
    cost = 60.0 * (a["number_kind"] != b["number_kind"]) + 16.0 * (a["case_kind"] != b["case_kind"])
    cost += 7.0 * abs(a["tokens"] - b["tokens"]) + abs(a["chars"] - b["chars"]) / max(a["chars"], 1)
    return cost + sum(3.0 * (a[k] != b[k]) for k in ("has_ampersand", "has_hyphen", "has_comma"))


def _eligible(target: Pair, donor: Pair) -> bool:
    aliases = {_norm(a) for a in target.get("ground_truth_aliases", [])} | {_norm(target["true_answer"])}
    d = _norm(donor["true_answer"])
    return donor["id"] != target["id"] and bool(d) and d not in aliases and d not in _norm(target["question"])


def gold_donor_assignments(pairs: list[Pair], seed: int = GOLD_DONOR_SEED) -> list[dict[str, Any]]:
    """Assign each TriviaQA question a donor gold answer from the same split.

    Donors are used once, must not be an alias of the target answer or occur in
    its question, and are matched by answer form (exact bucket -> coarse bucket ->
    whole split; rarer forms assigned first; ties broken by the seeded RNG).
    """
    rng = random.Random(seed)
    by_split: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_split[p["split"]].append(p)
    out = []
    for split, rows in sorted(by_split.items()):
        prof = {r["id"]: answer_profile(r["true_answer"]) for r in rows}
        exact: dict[tuple, list[Pair]] = defaultdict(list)
        coarse: dict[tuple, list[Pair]] = defaultdict(list)
        for r in rows:
            p = prof[r["id"]]
            exact[(p["number_kind"], p["case_kind"], p["tokens"])].append(r)
            coarse[(p["number_kind"], p["case_kind"])].append(r)
        for bucket in [*exact.values(), *coarse.values()]:
            rng.shuffle(bucket)

        def ekey(r: Pair) -> tuple:
            p = prof[r["id"]]
            return (p["number_kind"], p["case_kind"], p["tokens"])

        order = sorted(rows, key=lambda r: (len(exact[ekey(r)]), rng.random()))
        unused = {r["id"]: r for r in rows}
        for target in order:
            p = prof[target["id"]]
            reused = False
            cands = [d for d in exact[ekey(target)] if d["id"] in unused and _eligible(target, d)]
            if not cands:
                cands = [d for d in coarse[(p["number_kind"], p["case_kind"])]
                         if d["id"] in unused and _eligible(target, d)]
            if not cands:
                cands = [d for d in rows if d["id"] in unused and _eligible(target, d)]
            if not cands:
                cands, reused = [d for d in rows if _eligible(target, d)], True
            best_cost = min(match_cost(p, prof[d["id"]]) for d in cands)
            donor = rng.choice([d for d in cands if match_cost(p, prof[d["id"]]) == best_cost])
            unused.pop(donor["id"], None)
            out.append({"target_id": target["id"], "donor_id": donor["id"], "split": split,
                        "cost": best_cost, "donor_reused": reused, "target_profile": p,
                        "donor_profile": prof[donor["id"]]})
    return out


def build_gold_donor_control(pairs: list[Pair], seed: int = GOLD_DONOR_SEED,
                             review_size: int = 100) -> tuple[list[Pair], list[dict], list[dict]]:
    """Gold-vs-gold TriviaQA control (App. D.2).

    Returns (pairs, assignments, review_sample). ``review_sample`` is the random
    100-pair semantic audit sample (seed + 1).
    """
    assignments = gold_donor_assignments(pairs, seed)
    by_id = {p["id"]: p for p in pairs}
    by_target = {a["target_id"]: a for a in assignments}
    if len(by_target) != len(pairs):
        raise RuntimeError("assignment count does not match pair count")
    out = []
    for p in pairs:
        a = by_target[p["id"]]
        q = dict(p)
        q["false_answer"] = by_id[a["donor_id"]]["true_answer"]
        q["control_metadata"] = {"control_version": GOLD_DONOR_VERSION,
                                 "true_answer_source": "triviaqa_gold",
                                 "false_answer_source": "triviaqa_gold_donor",
                                 "donor_id": a["donor_id"], "match_cost": a["cost"],
                                 "donor_reused": a["donor_reused"]}
        out.append(q)
    review = random.Random(seed + 1).sample(assignments, min(review_size, len(assignments)))
    return out, assignments, review


def split_counts(pairs: list[Pair]) -> dict[str, int]:
    return dict(Counter(p["split"] for p in pairs))


# --------------------------------------------------------------------------- CLI

def add_cli(sub: Any) -> None:
    p = sub.add_parser("pairs", help="build contrastive pairs or the gold-donor control (Sec. 4, App. A.1, D.2)")
    p.add_argument("--dataset", required=True, choices=["triviaqa", "arc_easy", "commonsenseqa", "gold_donor"])
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--generate", action="store_true",
                   help="TriviaQA: call GPT-4.1 (needs OPENAI_API_KEY) instead of reusing the released distractors")
    p.set_defaults(func=_cli)


def _cli(args: Any) -> None:
    import csv
    import os

    from .config import REPO_ROOT

    if args.dataset == "gold_donor":
        pairs, assignments, review = build_gold_donor_control(load_pairs("triviaqa"))
        save_pairs(pairs, args.out)
        rows = {p["id"]: p for p in load_pairs("triviaqa")}
        with open(args.out / "manual_review_sample.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["target_id", "question", "true_answer", "donor_id", "false_answer", "split", "match_cost"])
            for a in review:
                t, d = rows[a["target_id"]], rows[a["donor_id"]]
                w.writerow([t["id"], t["question"], t["true_answer"], d["id"], d["true_answer"], a["split"], a["cost"]])
        (args.out / "assignments.json").write_text(json.dumps(assignments, ensure_ascii=False, indent=2))
    elif args.dataset == "triviaqa":
        raw = REPO_ROOT / "data" / "raw"
        records = triviaqa_records(json.loads((raw / "triviaqa_sample5000.json").read_text()),
                                   {m: json.loads((raw / f"triviaqa_greedy_{m}.json").read_text())
                                    for m in ("llama", "qwen")})
        if args.generate:
            from openai import OpenAI

            distractors = generate_triviaqa_distractors(records, OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        else:
            distractors = {p["id"]: p["false_answer"] for p in load_pairs("triviaqa")}
        pairs = build_triviaqa_pairs(records, distractors)
        save_pairs(pairs, args.out, triviaqa_stats(pairs))
    else:
        pairs = build_mc_pairs(load_mc_records(args.dataset))
        save_pairs(pairs, args.out, mc_stats(pairs, args.dataset))
    print(f"wrote {args.out}")
