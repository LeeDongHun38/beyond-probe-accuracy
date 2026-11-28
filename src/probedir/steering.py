"""Activation steering and the constrained correctness-judgment readout (Sec. 3.3-3.4; App. B.3).

Intervention (Eq. 4 / Eq. 12). At the selected transformer block, a forward hook adds
``alpha * v`` (v unit-normalized; the sign s is folded into alpha) to the block output at
every non-padding prefill position. The judgment endpoint uses a single prefill forward
pass, so nothing is applied during decoding unless ``prefill_only=False``.

Bidirectional protocol (Eq. 13). The false side (incorrect proposed answer a-) is moved
along +v, the true side (correct answer a+) along -v; the expected signs are
Delta P_F > 0 and Delta P_T < 0.

Readout (Eqs. 14-17). For the next-token logits z at the last prefill position,
P(Yes) is the full-vocabulary softmax probability of the highest-logit token in the Yes
candidate set, and the margin m is the max Yes logit minus the max No logit.

Residual-relative magnitude (Eq. 7). r = alpha / median_i ||h_{l, t*_i}||, where the
median is taken over the baseline (unsteered) block outputs at the final non-padding
prefill position, pooled over seeds and both sides.

Effects are averaged over the 200 held-out pairs within each seed, then over seeds
(App. B.4). Held-out pairs for seed s are ``random.Random(s).sample(heldout, 200)``
over the held-out pairs in file order.
"""
from __future__ import annotations

import argparse
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .config import Cell, load_config, pairs_path, results_dir
from .io import HiddenCache, cache_path, write_json

YES_CANDIDATES = ("Yes", "yes", " Yes", " yes", "YES")
NO_CANDIDATES = ("No", "no", " No", " no", "NO")
# Qwen2.5-14B uses the expanded readout for all new TriviaQA conditions
# and for the fixed-alpha runs on the other datasets.
EXTENDED_YES = ("Yes", "yes", " Yes", " yes", "YES", "True", "true", " True", "Correct", "correct")
EXTENDED_NO = ("No", "no", " No", " no", "NO", "False", "false", " False", "Incorrect", "incorrect")

# Whether the judgment prompt is rendered with the assistant generation prefix.
ADD_GENERATION_PROMPT = {
    "llama-3.1-8b": True,
    "qwen2.5-7b": True,
    "mistral-7b": False,
    "qwen2.5-14b": True,
    "ministral-3-14b": False,
}

SIDES = {"false": ("false_answer", +1.0), "true": ("true_answer", -1.0)}


# --------------------------------------------------------------------------- tokens / prompts

def candidate_token_ids(tokenizer, model_key: str, extended: bool = False) -> tuple[list[int], list[int]]:
    """Token ids of the Yes and No candidate sets.

    Standard models: the first token of each candidate string (``encode`` without special
    tokens). Ministral-3-14B: candidates that encode to a single token, sorted.
    """
    yes_c, no_c = (EXTENDED_YES, EXTENDED_NO) if extended else (YES_CANDIDATES, NO_CANDIDATES)
    if model_key == "ministral-3-14b":
        def singles(cands):
            out = set()
            for text in cands:
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                ids = list(ids) if isinstance(ids, (list, tuple)) else ids.tolist()
                if len(ids) == 1:
                    out.add(int(ids[0]))
            return sorted(out)
        return singles(yes_c), singles(no_c)

    def first(cands):
        ids = []
        for text in cands:
            enc = tokenizer.encode(text, add_special_tokens=False)
            if not enc:
                raise ValueError(f"candidate {text!r} has no tokens")
            ids.append(int(enc[0]))
        return ids

    if extended:   # the original extended run de-duplicated through a Python set
        return list({*first(yes_c)}), list({*first(no_c)})
    return first(yes_c), first(no_c)


def encode_judgment(tokenizer, model_key: str, question: str, answer: str,
                    system: bool = True, add_generation_prompt: bool | None = None) -> dict:
    """Token ids (batch of 1) of the Yes/No correctness-judgment prompt (App. A.2)."""
    from .prompts import judgment_messages

    messages = judgment_messages(question, answer)
    if not system:
        messages = [m for m in messages if m["role"] != "system"]
    if model_key == "ministral-3-14b":
        enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                            continue_final_message=False, return_tensors="pt",
                                            return_dict=True)
        return {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    agp = ADD_GENERATION_PROMPT[model_key] if add_generation_prompt is None else add_generation_prompt
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=agp)
    return dict(tokenizer(text, return_tensors="pt"))


# --------------------------------------------------------------------------- hooks

def _blocks(model) -> list:
    try:
        from .models import blocks
        return list(blocks(model))
    except ImportError:
        inner = getattr(model, "model", model)
        inner = getattr(inner, "language_model", inner)
        return list(inner.layers)


def _input_device(model):
    try:
        from .models import input_device
        return input_device(model)
    except ImportError:
        return model.get_input_embeddings().weight.device


class _State:
    mask = None          # attention mask of the current forward pass
    prefill = True       # whether the current forward pass is the prefill
    norms: list | None = None


@contextmanager
def steer(model, layer: int, direction, alpha: float, prefill_only: bool = True,
          record_norms: list | None = None) -> Iterator[None]:
    """Add ``alpha * direction`` to the output of block ``layer`` at non-padding positions.

    The direction is cast to the block's dtype/device inside the hook and then scaled,
    as in the original runs. ``prefill_only`` restricts the intervention to the prefill
    forward pass (cached decoding steps are left unchanged). If ``record_norms`` is a
    list, the L2 norm of the (unsteered) block output at the final non-padding position
    of each sequence is appended to it.
    """
    import torch

    vec = None if direction is None else torch.as_tensor(np.asarray(direction, dtype=np.float32))
    state = _State()

    def pre_hook(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        emb = kwargs.get("inputs_embeds")
        length = (ids if ids is not None else emb).shape[1]
        state.mask = kwargs.get("attention_mask")
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            state.prefill = int(cache_position.reshape(-1)[0]) == 0
        else:
            past = kwargs.get("past_key_values")
            empty = past is None or (hasattr(past, "get_seq_length") and past.get_seq_length() == 0)
            state.prefill = empty and length >= 1
        if state.mask is not None and state.mask.shape[1] != length:   # cached decoding step
            state.mask = state.mask[:, -length:]

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        mask = state.mask
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        mask = mask.to(hidden.device).bool()
        if record_norms is not None and state.prefill:
            last = mask.long().cumsum(1).argmax(1)       # final non-padding position
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            record_norms.extend(hidden[rows, last].float().norm(dim=-1).tolist())
        if vec is None or alpha == 0.0 or (prefill_only and not state.prefill):
            return None
        delta = alpha * vec.to(device=hidden.device, dtype=hidden.dtype)
        changed = hidden + mask.unsqueeze(-1) * delta.view(1, 1, -1)
        return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

    handles = [model.register_forward_pre_hook(pre_hook, with_kwargs=True),
               _blocks(model)[layer].register_forward_hook(hook)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- readout

def judge(model, tokenizer, model_key: str, questions: Sequence[str], answers: Sequence[str],
          batch_size: int = 1, system: bool = True, extended_candidates: bool = False,
          add_generation_prompt: bool | None = None) -> dict[str, np.ndarray]:
    """P(Yes) and the Yes-minus-No logit margin for each (question, answer).

    ``batch_size=1`` (no padding) reproduces the paper's numbers exactly; larger batches
    left-pad and pass explicit position ids, which is numerically close but not identical.
    """
    import torch

    yes_ids, no_ids = candidate_token_ids(tokenizer, model_key, extended_candidates)
    device = _input_device(model)
    p_yes, margin = [], []
    encoded = [encode_judgment(tokenizer, model_key, q, a, system, add_generation_prompt)
               for q, a in zip(questions, answers)]
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        if len(chunk) == 1:
            inputs = {k: v.to(device) for k, v in chunk[0].items()}
        else:
            inputs = _left_pad(chunk, tokenizer, device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite judgment logits")
        y = logits[:, torch.tensor(yes_ids, device=logits.device)]
        n = logits[:, torch.tensor(no_ids, device=logits.device)]
        y_best = y.argmax(1)
        probs = torch.softmax(logits, dim=-1)
        rows = torch.arange(len(logits), device=logits.device)
        chosen = torch.tensor(yes_ids, device=logits.device)[y_best]
        p_yes.extend(probs[rows, chosen].tolist())
        margin.extend((y.max(1).values.double() - n.max(1).values.double()).tolist())
    return {"p_yes": np.asarray(p_yes), "margin": np.asarray(margin)}


def _left_pad(chunk: list[dict], tokenizer, device) -> dict:
    import torch

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(int(c["input_ids"].shape[1]) for c in chunk)
    ids = torch.full((len(chunk), width), pad, dtype=torch.long)
    mask = torch.zeros((len(chunk), width), dtype=torch.long)
    for i, c in enumerate(chunk):
        n = int(c["input_ids"].shape[1])
        ids[i, width - n:] = c["input_ids"][0]
        mask[i, width - n:] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "position_ids": pos.to(device)}


# --------------------------------------------------------------------------- magnitude

def median_residual_norm(norms: Sequence[float]) -> float:
    """Median baseline residual norm at the final prefill position (Eq. 7 denominator)."""
    return float(np.median(np.asarray(norms, dtype=float)))


def alpha_for_r(r: float, median_norm: float) -> float:
    """Coefficient giving residual-relative magnitude r (Eq. 7 with ||v|| = 1)."""
    return float(r) * float(median_norm)


def r_for_alpha(alpha: float, median_norm: float) -> float:
    return float(alpha) / float(median_norm)


# --------------------------------------------------------------------------- experiment

def heldout_sample(dataset: str, seed: int, n: int | None = None) -> list[dict]:
    """The seed's held-out evaluation pairs: ``random.Random(seed).sample(heldout, n)``."""
    import json

    n = n or int(load_config()["intervention"]["heldout_pairs_per_seed"])
    pairs = [p for p in json.loads(pairs_path(dataset, "all").read_text()) if p.get("split") == "test"]
    return random.Random(seed).sample(pairs, n)


def seeds_for(cell: Cell) -> list[int]:
    cfg = load_config()["intervention"]
    return list(cfg["seeds_14b"] if cell.is_14b else cfg["seeds_7b_8b"])


def side_effects(model, tokenizer, model_key: str, layer: int, pairs: Sequence[dict],
                 direction, alpha: float, baseline: dict[str, dict[str, np.ndarray]] | None = None,
                 **judge_kwargs) -> dict[str, Any]:
    """Baseline and steered readouts for both sides of one seed's pairs.

    False side: +alpha v on (q, a-). True side: -alpha v on (q, a+).
    Returns per-side per-sample arrays and the seed-mean Delta P / Delta m.
    """
    qs = [p["question"] for p in pairs]
    out: dict[str, Any] = {"ids": [p["id"] for p in pairs], "baseline_norms": []}
    baseline = baseline or {}
    for side, (key, sign) in SIDES.items():
        answers = [p[key] for p in pairs]
        if side not in baseline:
            norms: list[float] = []
            with steer(model, layer, None, 0.0, record_norms=norms):
                baseline[side] = judge(model, tokenizer, model_key, qs, answers, **judge_kwargs)
            baseline[side]["norms"] = np.asarray(norms)
        with steer(model, layer, direction, sign * alpha):
            steered = judge(model, tokenizer, model_key, qs, answers, **judge_kwargs)
        out[side] = {
            "baseline_p_yes": baseline[side]["p_yes"], "steered_p_yes": steered["p_yes"],
            "baseline_margin": baseline[side]["margin"], "steered_margin": steered["margin"],
            "delta_p": float(np.mean(steered["p_yes"] - baseline[side]["p_yes"])),
            "delta_m": float(np.mean(steered["margin"] - baseline[side]["margin"])),
        }
        out["baseline_norms"].extend(baseline[side]["norms"].tolist())
    out["baseline"] = baseline
    return out


def run_cell(model_key: str, dataset: str, alpha: float | None = None, r: float | None = None,
             random_null: bool = False, sensitivity_null: bool = False, seeds: Sequence[int] | None = None,
             system: bool = True, extended_candidates: bool | None = None, tag: str | None = None,
             precision: str = "fp16") -> dict[str, Any]:
    """Steer one cell with its MD direction (and optionally the 127 random directions).

    Exactly one of ``alpha`` or ``r`` is given; with ``r`` the coefficient is
    ``r * median baseline residual norm`` measured on the same seeds' baselines.
    ``system=False`` reproduces the earlier no-system-prompt Mistral-7B TriviaQA run
    from the review-version experiments.
    """
    from .models import load_model
    from .nulls import cell_random_directions
    from .probes import md_direction

    cell = Cell(model_key, dataset)
    seeds = list(seeds or seeds_for(cell))
    if extended_candidates is None:
        extended_candidates = model_key == "qwen2.5-14b" and (dataset == "triviaqa" or r is None)
    cache = HiddenCache.load(cache_path(model_key, dataset, "fp16"))
    v = md_direction(*cache.layer(cell.layer, "train")).astype(np.float32)
    model, tokenizer = load_model(model_key, precision)
    kw = {"system": system, "extended_candidates": extended_candidates}
    if not system:   # the no-system run also rendered the generation prefix
        kw["add_generation_prompt"] = True

    baselines, per_seed = {}, {}
    for s in seeds:
        pairs = heldout_sample(dataset, s)
        baselines[s] = {"pairs": pairs, "readout": {}}
        norms: list[float] = []
        for side, (key, _) in SIDES.items():
            with steer(model, cell.layer, None, 0.0, record_norms=norms):
                res = judge(model, tokenizer, model_key, [p["question"] for p in pairs],
                            [p[key] for p in pairs], **kw)
            baselines[s]["readout"][side] = res
        baselines[s]["norms"] = norms
    median_norm = median_residual_norm([x for s in seeds for x in baselines[s]["norms"]])
    if (alpha is None) == (r is None):
        raise ValueError("give exactly one of alpha or r")
    coef = alpha if alpha is not None else alpha_for_r(r, median_norm)

    directions = {"probe": v}
    if random_null or sensitivity_null:
        rand = cell_random_directions(model_key, dataset, v.shape[0], sensitivity=sensitivity_null or (dataset == "triviaqa" and r == 0.05))
        directions.update({f"R{i:03d}": rand[i] for i in range(len(rand))})

    seed_effects = []
    for name, vec in directions.items():
        for s in seeds:
            res = side_effects(model, tokenizer, model_key, cell.layer, baselines[s]["pairs"], vec, coef,
                               baseline=baselines[s]["readout"], **kw)
            for side in SIDES:
                seed_effects.append({"model": model_key, "dataset": dataset, "side": side, "direction": name,
                                     "seed": s, "delta_p": res[side]["delta_p"], "delta_m": res[side]["delta_m"]})
            if name == "probe":
                per_seed[s] = {side: {k: res[side][k] for k in ("delta_p", "delta_m")} | {
                    "baseline_p": float(np.mean(res[side]["baseline_p_yes"]))} for side in SIDES}

    probe_rows = [e for e in seed_effects if e["direction"] == "probe"]
    summary = {
        "model": model_key, "dataset": dataset, "layer": cell.layer, "precision": precision,
        "alpha": coef, "r_target": r, "r": r_for_alpha(coef, median_norm), "median_residual_norm": median_norm,
        "seeds": seeds, "system_prompt": system, "extended_candidates": extended_candidates,
        "delta_p_false": float(np.mean([e["delta_p"] for e in probe_rows if e["side"] == "false"])),
        "delta_p_true": float(np.mean([e["delta_p"] for e in probe_rows if e["side"] == "true"])),
        "delta_m_false": float(np.mean([e["delta_m"] for e in probe_rows if e["side"] == "false"])),
        "delta_m_true": float(np.mean([e["delta_m"] for e in probe_rows if e["side"] == "true"])),
        "per_seed": per_seed, "seed_effects": seed_effects,
    }
    label = tag or (f"alpha{coef:g}" if alpha is not None else f"r{r:g}")
    write_json(results_dir() / "steering" / model_key / dataset / label / "summary.json", summary)
    return summary


def _cmd_steer(args: argparse.Namespace) -> None:
    alpha = None if args.r is not None else args.alpha
    tag = "alpha4" if alpha == 4.0 and args.system and not args.sensitivity_null else None
    s = run_cell(args.model, args.dataset, alpha=alpha, r=args.r, random_null=args.random_null,
                 sensitivity_null=args.sensitivity_null, seeds=args.seeds, system=args.system, tag=tag)
    print(f"{args.model}/{args.dataset}: alpha={s['alpha']:.4f} r={s['r']:.4f} "
          f"dP_F={s['delta_p_false']:+.4f} dP_T={s['delta_p_true']:+.4f}")


def add_cli(subparsers) -> None:
    p = subparsers.add_parser("steer", help="bidirectional MD steering with the P(Yes) readout (Sec. 3.3)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--alpha", type=float, default=float(load_config()["intervention"]["default_alpha"]))
    p.add_argument("--r", type=float, default=None, help="residual-relative magnitude (overrides --alpha)")
    p.add_argument("--random-null", action="store_true", help="also run the 127 random directions")
    p.add_argument("--sensitivity-null", action="store_true",
                   help="use the separate TriviaQA r~.05 random sets (Table 18)")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--no-system", dest="system", action="store_false",
                   help="omit the system message (earlier Mistral-7B TriviaQA run, Tables 15-16)")
    p.set_defaults(func=_cmd_steer)
