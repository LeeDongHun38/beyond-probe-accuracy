"""Answer-final residual representations (Sec. 3.1, Eq. 1; App. B.1).

For every pair, x+ = (q, a+) and x- = (q, a-) are run separately (batch size 1,
no padding) and the output of each transformer block is read at the last token
of the proposed answer.

Locating that token:

* HF tokenizers (Qwen, Mistral): the chat prompt is rendered to text, the
  answer's character span is the text after the last ``"A: "`` marker, and the
  target is the last token whose character offsets overlap that span. (Matching
  the standalone answer tokenization instead can miss the leading-space boundary
  of byte-level BPE vocabularies, which is why Qwen2.5 uses this alignment.)
* Llama-3.1-8B: the released representations used the earlier token-matching
  rule (last occurrence of the standalone answer tokens in the prompt). It agrees
  with the rendered-span rule on 42,290 of 42,316 answers; in the other 26 it
  reads an earlier prompt token. ``ALIGNMENT`` keeps that rule so the published
  numbers are reproduced; pass ``alignment="rendered_span"`` to use the other rule.
* Ministral-3-14B (mistral-common tokenizer, no offsets): the prompt is tokenized
  with and without the answer; the answer span is what lies between their longest
  common prefix and suffix, it must decode to the answer, and the target is its
  last non-special token.

Quantized conditions reuse the FP16 example order and target positions (App. B.1).
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from . import data, prompts
from .config import Cell, load_config
from .io import HiddenCache, cache_path
from .models import blocks, input_device, is_ministral, load_model

# Answer-token alignment rule per model (see module docstring); default "rendered_span".
ALIGNMENT = {"llama-3.1-8b": "token_match"}


# --------------------------------------------------------------------------- target position

def answer_token_index(tokenizer: Any, model_key: str, prompt_text: str, answer: str) -> int:
    """Index of the last token of ``answer`` inside the rendered ``prompt_text`` (HF tokenizers)."""
    if prompts.is_mistral_common(tokenizer):
        raise ValueError("use ministral_answer_span() for the mistral-common tokenizer")
    if getattr(tokenizer, "is_fast", False):
        marker = prompt_text.rfind(f"A: {answer}")
        start = marker + 3 if marker >= 0 else prompt_text.rfind(answer)
        if start >= 0:
            end = start + len(answer)
            offsets = tokenizer(prompt_text, return_offsets_mapping=True)["offset_mapping"]
            hits = [i for i, (a, b) in enumerate(offsets) if b > a and b > start and a < end]
            if hits:
                return hits[-1]
    return _token_match_index(tokenizer(prompt_text)["input_ids"], tokenizer, answer)


def _token_match_index(input_ids: list[int], tokenizer: Any, answer: str) -> int:
    """Fallback for slow tokenizers: last occurrence of the standalone answer tokens."""
    ans = tokenizer.encode(answer, add_special_tokens=False)
    ids = list(input_ids)
    if not ans:
        return len(ids) - 1
    for s in range(len(ids) - len(ans), -1, -1):
        if ids[s:s + len(ans)] == ans:
            return s + len(ans) - 1
    for k in range(min(3, len(ans)), 0, -1):
        tail = ans[-k:]
        for s in range(len(ids) - k, -1, -1):
            if ids[s:s + k] == tail:
                return s + k - 1
    if tokenizer.eos_token_id and ids[-1] == tokenizer.eos_token_id:
        return len(ids) - 2
    return len(ids) - 1


def ministral_answer_span(tokenizer: Any, question: str, answer: str, template_id: int) -> dict[str, Any]:
    """Answer span by longest common prefix/suffix of prompts with and without the answer (App. B.1)."""
    full = prompts.encode(tokenizer, "ministral-3-14b", prompts.pair_messages(question, answer, template_id))
    prefix = prompts.encode(tokenizer, "ministral-3-14b", prompts.pair_messages(question, "", template_id))
    lcp = 0
    while lcp < min(len(prefix), len(full)) and prefix[lcp] == full[lcp]:
        lcp += 1
    lcs = 0
    while lcs < len(prefix) - lcp and lcs < len(full) - lcp and prefix[-1 - lcs] == full[-1 - lcs]:
        lcs += 1
    stop = len(full) - lcs if lcs else len(full)
    candidate = full[lcp:stop]
    specials = set(getattr(tokenizer, "all_special_ids", []))
    usable = [i for i, t in enumerate(candidate, start=lcp) if t not in specials]
    decoded = tokenizer.decode(candidate, skip_special_tokens=True)
    if not usable or decoded.strip() != answer.strip():
        raise RuntimeError(f"answer span decode mismatch: {decoded!r} vs {answer!r}")
    return {"input_ids": full, "target_pos": usable[-1], "candidate_start": lcp, "candidate_stop": stop}


def locate(tokenizer: Any, model_key: str, question: str, answer: str, template_id: int,
           alignment: str | None = None) -> tuple[list[int], int]:
    """(input ids, target position) of the pair input x = (q, a)."""
    if is_ministral(model_key):
        span = ministral_answer_span(tokenizer, question, answer, template_id)
        return span["input_ids"], span["target_pos"]
    text = prompts.render(tokenizer, model_key, prompts.pair_messages(question, answer, template_id))
    ids = list(tokenizer(text)["input_ids"])
    if (alignment or ALIGNMENT.get(model_key, "rendered_span")) == "token_match":
        return ids, _token_match_index(ids, tokenizer, answer)
    return ids, answer_token_index(tokenizer, model_key, text, answer)


# --------------------------------------------------------------------------- forward capture

def forward_capture(model: Any, input_ids: list[int], position: int, layers: list[int],
                    attention_mask: bool = True, dtype: str = "float16") -> np.ndarray:
    """Block outputs at one position for one unpadded sequence -> [len(layers), d].

    Stored as float16 (the hidden-state caches) or float32 (the 14B template
    control, which scored unrounded block outputs). ``attention_mask=False`` runs the
    forward pass with input ids only (Ministral-3-14B, App. B.1).
    """
    import torch

    captured: dict[int, Any] = {}
    mods = blocks(model)
    handles = []
    for layer in layers:
        def hook(_m, _i, out, layer=layer):
            h = out[0] if isinstance(out, tuple) else out
            captured[layer] = h[0, position, :].detach().to("cpu", getattr(torch, dtype))
        handles.append(mods[layer].register_forward_hook(hook))
    ids = torch.tensor([input_ids], dtype=torch.long, device=input_device(model))
    try:
        with torch.no_grad():
            if attention_mask:
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
            else:
                model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([captured[layer] for layer in layers]).numpy()


# --------------------------------------------------------------------------- extraction

def native_rtn(model_key: str, dataset: str) -> bool:
    """Whether the simulated weight quantizer uses float16 statistics (7-8B TriviaQA conditions)."""
    return dataset == "triviaqa" and not Cell(model_key, dataset).is_14b


def extract(model_key: str, dataset: str, precision: str = "fp16", layers: list[int] | None = None,
            pairs: list[dict] | None = None, model: Any = None, tokenizer: Any = None,
            reference: HiddenCache | None = None) -> HiddenCache:
    """Answer-final block outputs for every pair of ``dataset``.

    ``layers`` defaults to all blocks for FP16 and to the FP16-selected layer for
    quantized conditions. For quantized conditions the target positions are taken
    from ``reference`` (default: the saved FP16 cache, when present).
    """
    pairs = pairs if pairs is not None else data.load_pairs(dataset)
    if model is None:
        model, tokenizer = load_model(model_key, precision, upcast_rtn=not native_rtn(model_key, dataset))
    n_blocks = len(blocks(model))
    if layers is None:
        layers = list(range(n_blocks)) if precision == "fp16" else [Cell(model_key, dataset).layer]
    if precision != "fp16" and reference is None and cache_path(model_key, dataset).exists():
        reference = HiddenCache.load(cache_path(model_key, dataset))
    ref_pos = None
    if reference is not None:
        if reference.ids != [p["id"] for p in pairs]:
            raise RuntimeError("reference cache order does not match the pairs")
        ref_pos = (reference.meta["target_pos_plus"], reference.meta["target_pos_minus"])

    hp, hm = [], []
    meta: dict[str, list] = {"target_pos_plus": [], "target_pos_minus": [], "input_len_plus": [],
                             "input_len_minus": [], "template_id": []}
    for i, pair in enumerate(pairs):
        t = int(pair.get("template_id", 0))
        for side, answer, store in (("plus", pair["true_answer"], hp), ("minus", pair["false_answer"], hm)):
            ids, pos = locate(tokenizer, model_key, pair["question"], answer, t)
            if ref_pos is not None:
                pos = int(ref_pos[0 if side == "plus" else 1][i])
            store.append(forward_capture(model, ids, pos, layers, attention_mask=not is_ministral(model_key)))
            meta[f"target_pos_{side}"].append(pos)
            meta[f"input_len_{side}"].append(len(ids))
        meta["template_id"].append(t)
    cfg = load_config()["models"][model_key]
    return HiddenCache(
        h_plus=np.stack(hp, axis=1), h_minus=np.stack(hm, axis=1),
        ids=[p["id"] for p in pairs], split=[p["split"] for p in pairs], layers=list(layers),
        meta={"model": model_key, "hf_id": cfg["hf_id"], "dataset": dataset, "precision": precision,
              "readout": "transformer-block output at the last answer token", **meta})


# --------------------------------------------------------------------------- 14B template control (App. D.1)

def extract_template_control(model_key: str, dataset: str, pair_ids: list[str] | None = None,
                             layer: int | None = None, model: Any = None, tokenizer: Any = None,
                             templates: tuple[int, ...] = tuple(range(len(prompts.TEMPLATE_SYSTEM_MESSAGES)))
                             ) -> dict[int, HiddenCache]:
    """Selected-layer FP16 representations of one fixed pair subset under every template.

    14B design of Table 21: the same held-out pairs (default
    ``probes.template_control_subset`` of the test split: 200 pairs, seed 20260713)
    are rendered under each of the five system messages in
    ``prompts.TEMPLATE_SYSTEM_MESSAGES``. Block outputs are kept in float32, as
    scored in the original runs. Returns ``{template_id: HiddenCache}`` with one
    layer; ``meta`` holds per-row input lengths and target positions.
    """
    pairs = data.load_pairs(dataset, "test")
    if pair_ids is None:
        from .probes import template_control_subset
        subset = template_control_subset(pairs)
    else:
        by_id = {p["id"]: p for p in pairs}
        subset = [by_id[i] for i in pair_ids]
    layer = Cell(model_key, dataset).layer if layer is None else int(layer)
    if model is None:
        model, tokenizer = load_model(model_key, "fp16")
    out = {}
    for t in templates:
        hp, hm = [], []
        meta: dict[str, list] = {"target_pos_plus": [], "target_pos_minus": [],
                                 "input_len_plus": [], "input_len_minus": []}
        for pair in subset:
            for side, answer, store in (("plus", pair["true_answer"], hp), ("minus", pair["false_answer"], hm)):
                ids, pos = locate(tokenizer, model_key, pair["question"], answer, t)
                store.append(forward_capture(model, ids, pos, [layer], attention_mask=not is_ministral(model_key),
                                             dtype="float32"))
                meta[f"target_pos_{side}"].append(pos)
                meta[f"input_len_{side}"].append(len(ids))
        out[t] = HiddenCache(h_plus=np.stack(hp, axis=1), h_minus=np.stack(hm, axis=1),
                             ids=[p["id"] for p in subset], split=["test"] * len(subset), layers=[layer],
                             meta={"model": model_key, "dataset": dataset, "precision": "fp16",
                                   "template_id": t, "system_message": prompts.TEMPLATE_SYSTEM_MESSAGES[t], **meta})
    return out


# --------------------------------------------------------------------------- CLI

def add_cli(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract", help="extract answer-final hidden states (Sec. 3.1)")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--precision", default="fp16")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.set_defaults(func=_cli)


def _cli(args: argparse.Namespace) -> None:
    cache = extract(args.model, args.dataset, args.precision, args.layers)
    out = cache_path(args.model, args.dataset, args.precision)
    cache.save(out)
    print(f"wrote {out}: layers={len(cache.layers)} pairs={len(cache.ids)}")
