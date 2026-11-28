"""Validity controls (App. D.2-D.6).

D.2  Gold-donor source/style control: answer-only character TF-IDF classifier, and
     transfer of the fixed original MD direction to gold-vs-gold pairs (Table 21).
D.3  Correctness readout from the final token of self-generated answers (Table 22).
D.4  Pre-answer readout: Spearman correlation of the MD projection at the final
     generation-prompt token with sampled-answer accuracy (Table 23).
D.5  Answer-span aggregation: last answer token vs. mean over the answer span (Table 24).
D.6  Direct answer-generation steering with exact-match scoring and McNemar tests.

The canonical MD direction is never refit and its sign is never adjusted here; it is
passed in by the caller (see :func:`probedir.probes.md_direction`).

Resampling schemes deliberately differ between analyses because they reproduce the
procedures used for the paper: D.3/D.4 use ``random.Random(seed)`` question-clustered
bootstraps and a ``torch.manual_seed`` Gaussian-direction stream; D.5 uses
``numpy.random.default_rng`` bootstraps and a ``torch.Generator`` direction stream.
"""
from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, rankdata

from .config import REPO_ROOT, pairs_path, results_dir
from .io import write_json
from .stats import auroc as _auroc_pos_neg

OWNGEN_DIR = REPO_ROOT / "data" / "owngen"
GOLD_DONOR_DIR = REPO_ROOT / "data" / "controls" / "gold_donor"

# Own-generation sampling (App. D.3-D.5)
OWNGEN_TEMPERATURE = 0.8
OWNGEN_TOP_P = 0.95
OWNGEN_MAX_NEW_TOKENS = 24
OWNGEN_SAMPLES = {"triviaqa": 16, "arc_easy": 32, "span_control": 16}
OWNGEN_SEEDS = {"triviaqa": 42, "arc_easy": (20260725, 20260726), "span_control": 20260728}

# Readout statistics
N_BOOT = 1000
BOOT_SEED = {"triviaqa": 42, "arc_easy": 20260725}   # D.3/D.4 question-clustered bootstrap
N_RANDOM = 1000
RANDOM_SEED = 123              # D.3/D.4 Gaussian-direction stream (torch.manual_seed)
SPAN_SEED = 20260728           # D.5: bootstrap seed; random directions use SPAN_SEED + 1

# Direct answer-generation steering (App. D.6)
DIRECT_LAYER = 18
DIRECT_ALPHA = 4.0
DIRECT_RUNS = {"initial": (200, 20260725), "confirmation": (400, 20260726)}  # (n questions, selection seed)


# ---------------------------------------------------------------------------
# Answer parsing and exact-match labels
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPACE = re.compile(r"\s+")
# Own-generation runs strip a chatty prefix only when followed by whitespace;
# the direct-generation run also strips it when directly followed by the answer.
_CHATTY_OWNGEN = re.compile(r"^(?:the\s+answer\s+is|answer\s+is|answer|it\s+is|it's|that\s+is)\s*[:\-]?\s+", re.I)
_CHATTY_DIRECT = re.compile(r"^(?:the\s+answer\s+is|answer\s+is|answer|it\s+is|it's|that\s+is)\s*[:\-]?\s*", re.I)
_NEGATIONS = {"no", "not", "none", "neither", "wrong", "incorrect", "false"}


def normalize_answer(text: Any) -> str:
    """SQuAD-style normalization: lowercase, punctuation to spaces, drop articles, collapse spaces."""
    if text is None:
        return ""
    text = "".join(" " if ch in string.punctuation else ch for ch in str(text).lower())
    return _SPACE.sub(" ", _ARTICLES.sub(" ", text)).strip()


def parse_short_answer(raw: Any, direct: bool = False) -> str:
    """First line of a generation, without a leading 'The answer is'-style prefix."""
    text = str(raw or "").strip()
    if not direct:
        if not text:
            return ""
        for sep in ("\n", "\r"):
            if sep in text:
                text = text.split(sep, 1)[0].strip()
        return _CHATTY_OWNGEN.sub("", text).strip()
    text = text.split("\n", 1)[0].strip()
    return _CHATTY_DIRECT.sub("", text).strip()


def _span_match(pred: str, alias: str) -> tuple[bool, bool]:
    """(alias occurs as an independent token span, a negation word is within 3 tokens of any occurrence)."""
    p, a = pred.split(), alias.split()
    hits = [i for i in range(len(p) - len(a) + 1) if a and p[i:i + len(a)] == a]
    negated = any(t in _NEGATIONS for i in hits for t in p[max(0, i - 3): i + len(a) + 3])
    return bool(hits), negated


def label_answer(answer: str, aliases: Sequence[str], allow_span_match: bool = False) -> dict:
    """Label an answer as correct (normalized exact alias match), incorrect, or excluded (empty).

    ``allow_span_match`` additionally accepts an alias appearing as an independent token
    span, excluding it when a negation word is nearby.
    """
    pred = normalize_answer(answer)
    norms = [a for a in (normalize_answer(x) for x in aliases or []) if a]
    if not answer or not pred:
        return {"answer": answer, "correct": None, "label": "excluded", "match_type": "empty",
                "exclude_reason": "empty"}
    if pred in norms:
        return {"answer": answer, "correct": True, "label": "correct", "match_type": "exact"}
    if allow_span_match:
        for alias in norms:
            found, negated = _span_match(pred, alias)
            if found and negated:
                return {"answer": answer, "correct": None, "label": "excluded",
                        "match_type": "span_negation_context", "exclude_reason": "ambiguous"}
            if found:
                return {"answer": answer, "correct": True, "label": "correct", "match_type": "span"}
        return {"answer": answer, "correct": False, "label": "incorrect", "match_type": "no_match"}
    return {"answer": answer, "correct": False, "label": "incorrect", "match_type": "no_exact_match"}


def p_correct(samples: Iterable[dict]) -> float | None:
    """Fraction of correct samples among correct + incorrect (excluded samples dropped)."""
    labels = [s["label"] for s in samples]
    denom = labels.count("correct") + labels.count("incorrect")
    return None if denom == 0 else labels.count("correct") / denom


def load_records(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["records"] if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# Question selection for own-generation runs
# ---------------------------------------------------------------------------
def select_arc_owngen_questions(pairs: list[dict], excluded_ids: Iterable[str], seed: int = 20260725,
                                per_split: int = 500) -> list[list[dict]]:
    """ARC-Easy own-generation questions (App. D.3): 500 train + 500 held-out questions
    not used in earlier pilots, returned as two computational shards of 250 + 250."""
    excluded = set(excluded_ids)
    available = [p for p in pairs if p["id"] not in excluded]
    selected = []
    for offset, split in enumerate(("train", "test")):
        candidates = [p for p in available if p.get("split") == split]
        selected.extend(random.Random(seed + offset).sample(candidates, per_split))
    shards: list[list[dict]] = [[], []]
    for k, split in enumerate(("train", "test")):
        rows = [p for p in selected if p.get("split") == split]
        random.Random(seed + 100 + k).shuffle(rows)
        shards[0].extend(rows[: len(rows) // 2])
        shards[1].extend(rows[len(rows) // 2:])
    return shards


def select_direct_generation_questions(pairs: list[dict], n: int, seed: int,
                                       exclude_ids: Iterable[str] = ()) -> list[dict]:
    """Held-out questions for App. D.6, sorted by id; the confirmation run excludes the initial run."""
    excluded = set(exclude_ids)
    pool = [p for p in pairs if p["id"] not in excluded]
    return sorted(random.Random(seed).sample(pool, n), key=lambda p: p["id"])


# ---------------------------------------------------------------------------
# Prompts and answer-token positions (own-generation contexts)
# ---------------------------------------------------------------------------
def readout_messages(question: str, answer: str, template_id: int, context: str,
                     model_key: str) -> list[dict]:
    """Messages for reading out a generated answer.

    ``assistant``: the generation prompt with the answer teacher-forced as the assistant turn
    (TriviaQA D.3, D.5). ``external_user``: the pair-representation prompt ``Q: .. A: ..``
    (ARC-Easy D.3), identical to the representation used for probe training.
    """
    from . import prompts

    if context == "assistant":
        return [*prompts.short_answer_messages(question, template_id, model_key),
                {"role": "assistant", "content": answer}]
    if context == "external_user":
        return prompts.pair_messages(question, answer, template_id=template_id)
    raise ValueError(context)


def answer_token_positions(tokenizer, rendered: str, answer: str) -> tuple[list[int], str]:
    """Token positions of the (last occurrence of the) answer in a rendered prompt.

    Uses character offsets of a fast tokenizer: every token overlapping the answer's
    character span (a BPE token may start on the preceding space). Falls back to
    searching for the answer's token ids, then to the final token.
    """
    if getattr(tokenizer, "is_fast", False):
        start = rendered.rfind(answer)
        if start >= 0:
            end = start + len(answer)
            offsets = tokenizer(rendered, return_offsets_mapping=True)["offset_mapping"]
            span = [i for i, (s, e) in enumerate(offsets) if e > s and e > start and s < end]
            if span:
                return span, "offset_full"
    ids = tokenizer(rendered)["input_ids"]
    target = tokenizer.encode(answer, add_special_tokens=False)
    if not target:
        return [], "empty"
    for s in range(len(ids) - len(target), -1, -1):
        if ids[s:s + len(target)] == target:
            return list(range(s, s + len(target))), "full"
    for k in range(min(3, len(target)), 0, -1):
        for s in range(len(ids) - k, -1, -1):
            if ids[s:s + k] == target[-k:]:
                return list(range(s, s + k)), f"partial_{k}"
    if tokenizer.eos_token_id and ids and ids[-1] == tokenizer.eos_token_id:
        return [len(ids) - 2], "fallback_eos_prev"
    return [len(ids) - 1], "fallback_last"


# ---------------------------------------------------------------------------
# Own-generation: sampling and hidden states (GPU)
# ---------------------------------------------------------------------------
def _stop_ids(tokenizer) -> list[int]:
    ids = [tokenizer.eos_token_id]
    nl = tokenizer.encode("\n", add_special_tokens=False)
    if len(nl) == 1:
        ids.append(nl[0])
    return list(dict.fromkeys(i for i in ids if i is not None))


def generate_answers(model, tokenizer, model_key: str, pairs: list[dict], n_samples: int, seed: int,
                     temperature: float = OWNGEN_TEMPERATURE, top_p: float = OWNGEN_TOP_P,
                     max_new_tokens: int = OWNGEN_MAX_NEW_TOKENS, allow_span_match: bool = False) -> list[dict]:
    """Sample ``n_samples`` short answers plus one greedy answer per question and label them.

    Seeds are set once per run (not per question); for each question the samples are
    drawn before the greedy answer, so the sampling RNG stream does not depend on it.
    """
    import torch
    from transformers import set_seed

    from . import prompts
    from .models import input_device

    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    stop = _stop_ids(tokenizer)
    device = input_device(model)
    records = []
    for pair in pairs:
        aliases = pair.get("ground_truth_aliases") or [pair.get("true_answer", "")]
        template_id = pair.get("template_id", 0)
        prompt = prompts.render(tokenizer, model_key, prompts.short_answer_messages(pair["question"], template_id, model_key),
                                add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        n0 = inputs["input_ids"].shape[1]
        with torch.no_grad():
            sampled = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p,
                                     num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
                                     pad_token_id=tokenizer.eos_token_id, eos_token_id=stop)
            greedy = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=stop)
        decode = lambda row: parse_short_answer(tokenizer.decode(row[n0:], skip_special_tokens=True).strip())
        samples = []
        for k, row in enumerate(sampled):
            samples.append({**label_answer(decode(row), aliases, allow_span_match), "sample_id": k})
        counts = Counter(s["label"] for s in samples)
        uniq = [normalize_answer(s["answer"]) for s in samples if normalize_answer(s["answer"])]
        records.append({
            "id": pair["id"], "question": pair["question"], "ground_truth_aliases": aliases,
            "split": pair.get("split", "unknown"), "template_id": template_id,
            "greedy": label_answer(decode(greedy[0]), aliases, allow_span_match), "samples": samples,
            "n_correct": counts["correct"], "n_incorrect": counts["incorrect"], "n_excluded": counts["excluded"],
            "n_total": len(samples), "p_correct": p_correct(samples),
            "sample_uniqueness_ratio": len(set(uniq)) / max(len(uniq), 1),
        })
    return records


def extract_owngen_states(model, tokenizer, model_key: str, records: list[dict], layer: int,
                          context: str = "assistant", positions: Sequence[str] = ("last",),
                          pre_answer: bool = True) -> dict[str, Any]:
    """Hidden states at the selected layer for own-generated answers (batch size 1).

    Returns ``pre`` (final generation-prompt token per question; D.4) and, for each
    labeled sample (correct/incorrect), the answer ``first``/``last`` token state or the
    ``mean`` over the answer span (D.3, D.5). Samples whose answer cannot be located are
    skipped and counted in ``match_types``.
    """
    import torch

    from . import prompts
    from .models import blocks, input_device

    device = input_device(model)
    block = blocks(model)[layer]
    captured: dict[str, Any] = {}

    def run(text: str, pos: dict[str, Any]) -> dict[str, np.ndarray]:
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            for name, p in pos.items():
                v = h[0, p, :].mean(dim=0) if name == "mean" else h[0, p, :]
                captured[name] = v.detach().float().cpu().numpy().astype(np.float16)
        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(**tokenizer(text, return_tensors="pt").to(device))
        finally:
            handle.remove()
        return dict(captured)

    pre, pre_meta = [], {"ids": [], "p_correct": [], "split": [], "template_id": []}
    states = {p: [] for p in positions}
    meta = {"ids": [], "sample_id": [], "labels": [], "split": [], "template_id": []}
    match_types: Counter = Counter()
    for rec in records:
        tid = rec.get("template_id", 0)
        if pre_answer:
            text = prompts.render(tokenizer, model_key, prompts.short_answer_messages(rec["question"], tid, model_key),
                                  add_generation_prompt=True)
            n = len(tokenizer(text)["input_ids"])
            pre.append(run(text, {"last": n - 1})["last"])
            for k, v in (("ids", rec["id"]), ("p_correct", rec.get("p_correct")), ("split", rec.get("split")),
                         ("template_id", tid)):
                pre_meta[k].append(v)
        for s in rec.get("samples", []):
            if s.get("label") not in ("correct", "incorrect"):
                continue
            text = prompts.render(tokenizer, model_key, readout_messages(rec["question"], s["answer"], tid, context, model_key))
            span, mtype = answer_token_positions(tokenizer, text, s["answer"])
            match_types[mtype] += 1
            if not span:
                continue
            want = {"first": span[0], "last": span[-1], "mean": span}
            got = run(text, {p: want[p] for p in positions})
            for p in positions:
                states[p].append(got[p])
            for k, v in (("ids", rec["id"]), ("sample_id", s.get("sample_id")),
                         ("labels", int(s["label"] == "correct")), ("split", rec.get("split")),
                         ("template_id", tid)):
                meta[k].append(v)
    out: dict[str, Any] = {"layer": layer, "context": context, "match_types": dict(match_types),
                           "answer": {"h": {p: np.stack(v) for p, v in states.items() if v}, **meta}}
    if pre_answer:
        out["pre"] = {"h": np.stack(pre), **pre_meta}
    return out


# ---------------------------------------------------------------------------
# Statistics helpers reproducing the original resampling schemes
# ---------------------------------------------------------------------------
def project(h: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """float32 projections h @ v. Uses torch when installed, which reproduces the
    original scores bit for bit (NumPy's BLAS sums in a different order, ~1e-7 AUROC)."""
    try:
        import torch
    except ImportError:
        return np.asarray(h, np.float32) @ np.asarray(direction, np.float32)
    return (torch.as_tensor(np.asarray(h, np.float32)) @ torch.as_tensor(np.asarray(direction, np.float32))).numpy()


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels, scores = np.asarray(labels), np.asarray(scores, float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return _auroc_pos_neg(scores[labels == 1], scores[labels == 0])


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation (average ranks for ties)."""
    if len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _percentile_ci(values: list) -> list[float | None]:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def bootstrap_ci_python(ids: np.ndarray, stat: Callable[[np.ndarray], float | None],
                        n_boot: int = N_BOOT, seed: int = 42) -> list[float | None]:
    """Question-clustered percentile CI; questions drawn with ``random.Random(seed).choice`` (D.3, D.4)."""
    rng = random.Random(seed)
    ids = np.asarray(ids)
    unique = sorted(set(ids.tolist()))
    rows = {q: np.where(ids == q)[0] for q in unique}
    vals = []
    for _ in range(n_boot):
        picked = [rng.choice(unique) for _ in unique]
        vals.append(stat(np.concatenate([rows[q] for q in picked])))
    return _percentile_ci(vals)


def bootstrap_auc_ci_numpy(ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                           n_boot: int = N_BOOT, seed: int = SPAN_SEED) -> list[float]:
    """Question-clustered percentile AUROC CI with ``numpy.random.default_rng(seed)`` (D.5);
    resamples containing a single class are skipped."""
    rng = np.random.default_rng(seed)
    unique = np.unique(ids)
    rows = {q: np.flatnonzero(ids == q) for q in unique}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[q] for q in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(labels[idx])) == 2:
            vals.append(_auc(labels[idx], scores[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gaussian_directions_sequential(d: int, n: int = N_RANDOM, seed: int = RANDOM_SEED) -> np.ndarray:
    """Unit Gaussian directions drawn one at a time after ``torch.manual_seed(seed)`` (D.3, D.4)."""
    import torch

    torch.manual_seed(seed)
    out = np.empty((n, d), np.float32)
    for k in range(n):
        v = torch.randn(d, dtype=torch.float32)
        out[k] = (v / v.norm()).numpy()
    return out


def gaussian_directions_batched(d: int, n: int = N_RANDOM, seed: int = SPAN_SEED + 1, device: str = "cpu",
                                batch: int = 32) -> np.ndarray:
    """Unit Gaussian directions from a seeded ``torch.Generator`` in column batches of 32 (D.5).

    The stream depends on the generator device; the paper's run used ``device="cuda"``.
    """
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    out = []
    for start in range(0, n, batch):
        w = min(batch, n - start)
        m = torch.randn(d, w, generator=gen, device=device)
        out.append((m / m.norm(dim=0, keepdim=True).clamp_min(1e-12)).T.cpu().numpy())
    return np.concatenate(out)


def _upper_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


# ---------------------------------------------------------------------------
# D.2 gold-donor source/style control
# ---------------------------------------------------------------------------
def style_classifier_auroc(pairs: list[dict]) -> dict:
    """Held-out AUROC of an answer-only char_wb TF-IDF (2-5 grams) logistic regression separating
    true from false answer strings; the pair train/held-out split is preserved."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    def split(name: str) -> tuple[list[str], list[int]]:
        text, y = [], []
        for p in pairs:
            if p.get("split") == name:
                text += [p["true_answer"], p["false_answer"]]
                y += [1, 0]
        return text, y

    (tr_x, tr_y), (te_x, te_y) = split("train"), split("test")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="liblinear")
    clf.fit(vec.fit_transform(tr_x), tr_y)
    prob = clf.predict_proba(vec.transform(te_x))[:, 1]
    return {"n_train_pairs": len(tr_y) // 2, "n_test_pairs": len(te_y) // 2,
            "test_auroc": float(roc_auc_score(te_y, prob)), "test_accuracy": float(accuracy_score(te_y, prob >= 0.5))}


def heldout_auroc(h_plus: np.ndarray, h_minus: np.ndarray, direction: np.ndarray) -> float:
    """AUROC of correct vs incorrect projections on a fixed direction (rows already held-out)."""
    return _auroc_pos_neg(h_plus @ direction, h_minus @ direction)


def gold_donor_control(direction: np.ndarray, original: tuple[np.ndarray, np.ndarray],
                       control: tuple[np.ndarray, np.ndarray]) -> dict:
    """Table 22 hidden-state transfer: the original TriviaQA MD direction (no refit) scored on the
    held-out original pairs and on held-out gold-donor pairs (each a (h_plus, h_minus) tuple)."""
    return {"original_auroc": heldout_auroc(*original, direction),
            "gold_donor_transfer_auroc": heldout_auroc(*control, direction)}


def semantic_audit(path: Path = REPO_ROOT / "artifacts/camera_ready/v1/review_split_donor_sample.csv") -> dict:
    import csv

    rows = list(csv.DictReader(Path(path).open(newline="")))
    return {"audit_sample_size": len(rows),
            "verified_false": sum(r["semantic_false_verified"].strip().casefold() == "yes" for r in rows)}


# ---------------------------------------------------------------------------
# D.3 / D.4 self-generated answer and pre-answer readouts
# ---------------------------------------------------------------------------
def self_readout(h: np.ndarray, labels: np.ndarray, ids: np.ndarray, split: np.ndarray, direction: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = 42) -> dict:
    """D.3: AUROC of answer-token projections (correct=1) on held-out questions with a
    question-clustered bootstrap CI."""
    mask = np.asarray(split) == "test"
    s = project(h, direction)[mask]
    y, q = np.asarray(labels)[mask], np.asarray(ids)[mask]
    return {"n": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
            "auroc": _auc(y, s), "auroc_ci": bootstrap_ci_python(q, lambda i: _auc(y[i], s[i]), n_boot, seed)}


def pre_answer_readout(h: np.ndarray, p_correct_values: Sequence, ids: np.ndarray, split: np.ndarray,
                       direction: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> dict:
    """D.4: Spearman between pre-answer projections and per-question sampled accuracy (held-out
    questions with at least one retained answer), with a question-clustered bootstrap CI."""
    valid = np.asarray([p is not None for p in p_correct_values]) & (np.asarray(split) == "test")
    x = project(h, direction)[valid]
    y = np.asarray([float(p) for p, v in zip(p_correct_values, valid) if v])
    q = np.asarray(ids)[valid]
    return {"n": int(len(y)), "spearman": spearman(x, y),
            "spearman_ci": bootstrap_ci_python(q, lambda i: spearman(x[i], y[i]), n_boot, seed)}


def triviaqa_intervals(post_scores, labels, post_keys, pre_scores, outcomes, pre_keys, n_boot=N_BOOT):
    """Shared question multiplicities for Tables 23 and 24, conditional on the direction."""
    from sklearn.metrics import roc_auc_score

    keys, pre_group = np.unique(pre_keys, return_inverse=True)
    group = {key: i for i, key in enumerate(keys)}
    post_group = np.asarray([group[key] for key in post_keys])
    rng = np.random.default_rng(42)
    aucs, correlations = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(len(keys), size=len(keys)), minlength=len(keys))
        weights = counts[post_group]
        if weights[labels == 1].sum() and weights[labels == 0].sum():
            aucs.append(roc_auc_score(labels, post_scores, sample_weight=weights))
        idx = np.repeat(np.arange(len(outcomes)), counts[pre_group])
        idx = idx[np.isfinite(outcomes[idx])]
        correlations.append(spearman(pre_scores[idx], outcomes[idx]))
    return _percentile_ci(aucs), _percentile_ci(correlations)


def project_many(h: np.ndarray, directions: np.ndarray, device: str = "cpu") -> np.ndarray:
    """float32 projections [n_rows, n_directions] (torch on ``device`` when installed)."""
    try:
        import torch
    except ImportError:
        return np.asarray(h, np.float32) @ np.asarray(directions, np.float32).T
    ht = torch.as_tensor(np.asarray(h, np.float32), device=device)
    dt = torch.as_tensor(np.asarray(directions, np.float32), device=device)
    return (ht @ dt.T).cpu().numpy()


def random_direction_pvalues(direction: np.ndarray, answer: dict | None = None, pre: dict | None = None,
                             n: int = N_RANDOM, seed: int = RANDOM_SEED) -> dict:
    """One-sided Monte-Carlo p-values of the MD readouts against unit Gaussian directions
    (signed metrics; p = (1 + #{random >= MD}) / (n + 1)). ``answer`` = {h, labels, split},
    ``pre`` = {h, p_correct, split}; held-out rows only."""
    v = np.asarray(direction, np.float32)
    v = v / np.float32(np.linalg.norm(v))
    dirs = gaussian_directions_sequential(len(v), n, seed)
    out: dict[str, Any] = {"n_random_directions": n, "seed": seed}
    if answer is not None:
        m = np.asarray(answer["split"]) == "test"
        h, y = np.asarray(answer["h"], np.float32)[m], np.asarray(answer["labels"])[m]
        obs = _auc(y, project(h, v))
        null = np.asarray([_auc(y, c) for c in project_many(h, dirs).T])
        out["answer_auroc"] = {"md": obs, "p": _upper_p(obs, null), "exceedances": int(np.sum(null >= obs)),
                               "random_values": null.tolist()}
    if pre is not None:
        m = (np.asarray(pre["split"]) == "test") & np.asarray([p is not None for p in pre["p_correct"]])
        h = np.asarray(pre["h"], np.float32)[m]
        y = np.asarray([float(p) for p, k in zip(pre["p_correct"], m) if k])
        obs = spearman(project(h, v), y)
        null = np.asarray([spearman(c, y) for c in project_many(h, dirs).T])
        out["pre_answer_spearman"] = {"md": obs, "p": _upper_p(obs, null), "exceedances": int(np.sum(null >= obs)),
                                      "random_values": null.tolist()}
    return out


# ---------------------------------------------------------------------------
# D.5 answer-span aggregation
# ---------------------------------------------------------------------------
def span_readout(h: np.ndarray, labels: np.ndarray, ids: np.ndarray, direction: np.ndarray,
                 n_boot: int = N_BOOT, seed: int = SPAN_SEED, n_random: int = N_RANDOM,
                 random_device: str = "cpu") -> dict:
    """D.5: AUROC of one answer-position readout (last token or span mean) on held-out
    self-generated answers, a question-clustered CI, and an orientation-invariant random-direction
    p-value (max(AUC, 1 - AUC) for random directions)."""
    s = project(h, direction)
    labels, ids = np.asarray(labels), np.asarray(ids)
    obs = _auc(labels, s)
    dirs = gaussian_directions_batched(h.shape[1], n_random, seed + 1, random_device)
    null = np.asarray([max(a, 1 - a) for a in (_auc(labels, c) for c in project_many(h, dirs, random_device).T)])
    return {"n_answers": int(len(labels)), "n_exact_correct": int(labels.sum()), "n_incorrect": int((labels == 0).sum()),
            "auroc": obs, "auroc_ci": bootstrap_auc_ci_numpy(ids, labels, s, n_boot, seed),
            "random_p": _upper_p(obs, null), "random_auroc_mean": float(null.mean())}


# ---------------------------------------------------------------------------
# D.6 direct answer-generation steering
# ---------------------------------------------------------------------------
def paired_accuracy(baseline: Sequence[bool | None], condition: Sequence[bool | None]) -> dict:
    """Paired exact-match accuracy change with a two-sided exact McNemar (binomial) test;
    questions with an empty answer in either condition are dropped."""
    pairs = [(a, b) for a, b in zip(baseline, condition) if a is not None and b is not None]
    n = len(pairs)
    improve = sum((not a) and b for a, b in pairs)
    worsen = sum(a and (not b) for a, b in pairs)
    base, cond = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    disc = improve + worsen
    p = float(binomtest(improve, disc, 0.5, alternative="two-sided").pvalue) if disc else 1.0
    return {"n_scored": n, "baseline_correct": base, "condition_correct": cond,
            "baseline_accuracy": base / n if n else None, "condition_accuracy": cond / n if n else None,
            "accuracy_delta": (cond - base) / n if n else None,
            "improve_count": improve, "worsen_count": worsen, "mcnemar_exact_two_sided_p": p}


def _exact(answer: str, aliases: Sequence[str]) -> bool | None:
    pred = normalize_answer(answer)
    if not pred:
        return None
    return any(pred == normalize_answer(a) for a in aliases if normalize_answer(a))


def direct_generation(model, tokenizer, model_key: str, pairs: list[dict], direction: np.ndarray,
                      layer: int = DIRECT_LAYER, alpha: float = DIRECT_ALPHA,
                      max_new_tokens: int = OWNGEN_MAX_NEW_TOKENS) -> list[dict]:
    """Greedy short answers without steering and with +alpha*v / -alpha*v added to the layer
    output at every position, during prefill and every cached decoding step (App. D.6)."""
    import contextlib

    import torch

    from . import prompts
    from .models import input_device
    from .steering import steer

    device = input_device(model)

    def gen(prompt: str, a: float | None) -> str:
        ctx = steer(model, layer, direction, a, prefill_only=False) if a is not None else contextlib.nullcontext()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with ctx, torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)[0]
        return parse_short_answer(tokenizer.decode(ids[inputs["input_ids"].shape[1]:], skip_special_tokens=True),
                                  direct=True)

    records = []
    for pair in pairs:
        prompt = prompts.render(tokenizer, model_key,
                                prompts.short_answer_messages(pair["question"], int(pair.get("template_id", 0)), model_key),
                                add_generation_prompt=True)
        aliases = pair.get("ground_truth_aliases") or [pair.get("true_answer", "")]
        rec = {"id": pair["id"], "question": pair["question"], "aliases": aliases}
        for name, a in (("baseline", None), ("plus_v", alpha), ("minus_v", -alpha)):
            ans = gen(prompt, a)
            rec[name] = {"answer": ans, "correct": _exact(ans, aliases)}
        records.append(rec)
    return records


def summarize_direct_generation(records: list[dict]) -> dict:
    base = [r["baseline"]["correct"] for r in records]
    return {c: paired_accuracy(base, [r[c]["correct"] for r in records]) for c in ("plus_v", "minus_v")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_direction(args) -> np.ndarray:
    """``--direction`` (.npy) or the MD direction fitted on the FP16 cache's training split."""
    if args.direction:
        return np.load(args.direction)
    from .io import HiddenCache, cache_path
    from .probes import md_direction

    cache = HiddenCache.load(cache_path(args.model, args.dataset, "fp16"))
    return md_direction(*cache.layer(_layer(args), "train"))


def _layer(args) -> int:
    from .config import Cell

    return args.layer if args.layer is not None else Cell(args.model, args.dataset).layer


def _cmd_style(args) -> None:
    from .io import HiddenCache, cache_path

    orig = json.loads(pairs_path("triviaqa").read_text())
    ctrl = json.loads((GOLD_DONOR_DIR / "all.json").read_text())
    out: dict[str, Any] = {"style_classifier": {"original": style_classifier_auroc(orig),
                                                "gold_donor": style_classifier_auroc(ctrl)},
                           "donor_audit": json.loads((GOLD_DONOR_DIR / "source_matching_audit.json").read_text())}
    for model in args.models:
        args.model, args.dataset = model, "triviaqa"
        layer = _layer(args)
        v = _load_direction(args)
        o = HiddenCache.load(cache_path(model, "triviaqa", "fp16")).layer(layer, "test")
        c = HiddenCache.load(cache_path(model, "triviaqa_gold_donor", "fp16")).layer(layer, "test")
        out[model] = {"layer": layer, **gold_donor_control(v, o, c)}
    write_json(results_dir() / "controls" / "d2_gold_donor.json", out)


def _cmd_generate(args) -> None:
    from .models import load_model

    model, tok = load_model(args.model)
    pairs = json.loads(Path(args.pairs).read_text())
    if isinstance(pairs, dict):
        pairs = pairs["pairs"]
    recs = generate_answers(model, tok, args.model, pairs, args.n_samples, args.seed,
                            allow_span_match=args.allow_span_match)
    write_json(Path(args.output), {"metadata": vars(args) | {"func": None}, "records": recs})


def _cmd_readout(args) -> None:
    """Extract own-generation states (GPU) and compute D.3/D.4 (and D.5 when --span)."""
    from .models import load_model

    records = load_records(Path(args.answers))
    if args.dataset == "triviaqa" and not args.span:
        records = [r for r in records if r["split"] == "test"]
    layer = _layer(args)
    v = _load_direction(args)
    model, tok = load_model(args.model)
    positions = ("last", "mean") if args.span else ("last",)
    st = extract_owngen_states(model, tok, args.model, records, layer, args.context, positions,
                               pre_answer=not args.span)
    del model
    a = st["answer"]
    out: dict[str, Any] = {"layer": layer, "context": args.context, "match_types": st["match_types"]}
    if args.span:
        for p in positions:
            out[p] = span_readout(a["h"][p], np.asarray(a["labels"]), np.asarray(a["ids"]), v)
    else:
        seed = BOOT_SEED.get(args.dataset, 42)
        if args.dataset == "triviaqa":
            from .data import question_key
            lookup = {rec["id"]: question_key(rec["question"]) for rec in records}
            mask = np.asarray(a["split"]) == "test"
            labels = np.asarray(a["labels"])[mask]
            post = np.asarray(a["h"]["last"])[mask] @ v.astype(np.float64)
            post_keys = [lookup[k] for k in np.asarray(a["ids"])[mask]]
            pre = st["pre"]
            mask = np.asarray(pre["split"]) == "test"
            scores = np.asarray(pre["h"])[mask] @ v.astype(np.float64)
            outcomes = np.asarray(pre["p_correct"], dtype=float)[mask]
            keys = [lookup[k] for k in np.asarray(pre["ids"])[mask]]
            auc_ci, rho_ci = triviaqa_intervals(post, labels, post_keys, scores, outcomes, keys)
            valid = np.isfinite(outcomes)
            out["answer"] = dict(n=len(labels), n_positive=int(labels.sum()), n_negative=int((labels == 0).sum()),
                                 auroc=_auc(labels, post), auroc_ci=auc_ci)
            out["pre_answer"] = dict(n=int(valid.sum()), spearman=spearman(scores[valid], outcomes[valid]),
                                     spearman_ci=rho_ci)
        else:
            out["answer"] = self_readout(a["h"]["last"], np.asarray(a["labels"]), np.asarray(a["ids"]),
                                         np.asarray(a["split"]), v, seed=seed)
            pre = st["pre"]
            out["pre_answer"] = pre_answer_readout(pre["h"], pre["p_correct"], np.asarray(pre["ids"]),
                                                   np.asarray(pre["split"]), v, seed=seed)
        if args.random_control:
            out["random_directions"] = random_direction_pvalues(
                v, {"h": a["h"]["last"], "labels": a["labels"], "split": a["split"]},
                {"h": pre["h"], "p_correct": pre["p_correct"], "split": pre["split"]})
    write_json(Path(args.output), out)


def _cmd_direct(args) -> None:
    from .models import load_model

    pairs = json.loads(pairs_path("triviaqa", "test").read_text())
    v = _load_direction(args)
    model, tok = load_model(args.model)
    initial = select_direct_generation_questions(pairs, *DIRECT_RUNS["initial"])
    confirm = select_direct_generation_questions(pairs, *DIRECT_RUNS["confirmation"],
                                                 exclude_ids=[p["id"] for p in initial])
    out = {}
    for name, sel in (("initial", initial), ("confirmation", confirm)):
        recs = direct_generation(model, tok, args.model, sel, v, DIRECT_LAYER, DIRECT_ALPHA)
        out[name] = {"selected_ids": [p["id"] for p in sel], "summary": summarize_direct_generation(recs),
                     "records": recs}
    write_json(results_dir() / "controls" / "d6_direct_generation.json", out)


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("controls", help="validity controls (App. D.2-D.6)")
    sub = p.add_subparsers(dest="control", required=True)

    def common(q, model_default="qwen2.5-7b", dataset_default="triviaqa"):
        q.add_argument("--model", default=model_default)
        q.add_argument("--dataset", default=dataset_default)
        q.add_argument("--layer", type=int, default=None, help="default: selected layer of the cell")
        q.add_argument("--direction", default=None, help=".npy direction; default: train-split MD of the FP16 cache")

    q = sub.add_parser("gold-donor", help="D.2 style classifier and gold-donor transfer (CPU)")
    q.add_argument("--models", nargs="+", default=["llama-3.1-8b", "qwen2.5-7b", "mistral-7b"])
    q.add_argument("--layer", type=int, default=None)
    q.add_argument("--direction", default=None)
    q.set_defaults(func=_cmd_style)

    q = sub.add_parser("generate", help="sample own-generated answers (GPU)")
    common(q)
    q.add_argument("--pairs", required=True, help="pair/question JSON (list or {'pairs': [...]})")
    q.add_argument("--n-samples", type=int, default=16)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--allow-span-match", action="store_true")
    q.add_argument("--output", required=True)
    q.set_defaults(func=_cmd_generate)

    q = sub.add_parser("readout", help="D.3/D.4 (or D.5 with --span) readouts of own-generated answers (GPU)")
    common(q)
    q.add_argument("--answers", required=True, help="answers JSON from `controls generate` or data/owngen/")
    q.add_argument("--context", choices=("assistant", "external_user"), default="assistant")
    q.add_argument("--span", action="store_true", help="D.5: last-token vs mean-pooled answer span")
    q.add_argument("--random-control", action="store_true", help="D.3/D.4 Gaussian-direction p-values")
    q.add_argument("--output", required=True)
    q.set_defaults(func=_cmd_readout)

    q = sub.add_parser("direct-generation", help="D.6 direct answer-generation steering (GPU)")
    common(q)
    q.set_defaults(func=_cmd_direct, layer=DIRECT_LAYER)
