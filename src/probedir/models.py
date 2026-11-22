"""Model loading for the FP16 reference and the five quantized conditions.

Conditions (Sec. 4, App. E):

* ``fp16``: reference model in float16 (Ministral-3-14B: BF16 checkpoint cast to FP16).
* ``awq`` / ``gptq``: public pre-quantized W4A16 checkpoints (:data:`QUANTIZED_CHECKPOINTS`).
  For Ministral-3-14B the packed public weights are dequantized explicitly into the
  FP16 base model, because the stock loaders do not map its nested language-model
  keys: AWQ restores all 7 attention/MLP projections per block (group size 32);
  GPTQ restores the MLP gate/up/down projections only (group size 128, static
  activation order), as the public checkpoint quantizes only that scope.
* ``rtn``: simulated W4A16 -- symmetric per-group (128) absmax round-to-nearest on
  every Linear weight inside the transformer blocks.
* ``w8a8`` / ``w4a8``: simulated INT8 / INT4 weights as above plus per-token
  symmetric INT8 quantize-dequantize of every transformer-block output.

Only residual-stream states are read from quantized models, so whether the final
vocabulary projection is quantized does not affect any reported quantity.
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, load_config

GROUP_SIZE = 128
SIMULATED = {"rtn": (4, False), "w8a8": (8, True), "w4a8": (4, True)}


# --------------------------------------------------------------------------- paths

# Public pre-quantized W4A16 checkpoints (environment/MODELS.md); the local directory
# name, when present under $PROBEDIR_MODELS (default ./models), takes precedence.
QUANTIZED_CHECKPOINTS = {
    "llama-3.1-8b": {"awq": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                     "gptq": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"},
    "qwen2.5-7b": {"awq": "Qwen/Qwen2.5-7B-Instruct-AWQ", "gptq": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"},
    "mistral-7b": {"awq": "solidrust/Mistral-7B-Instruct-v0.3-AWQ",
                   "gptq": "thesven/Mistral-7B-Instruct-v0.3-GPTQ"},
    "qwen2.5-14b": {"awq": "Qwen/Qwen2.5-14B-Instruct-AWQ", "gptq": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"},
    "ministral-3-14b": {"awq": "cyankiwi/Ministral-3-14B-Instruct-2512-AWQ-4bit",
                        "gptq": "inference-optimization/Ministral-3-14B-Instruct-2512.w4a16"},
}
# Local directory names used for the 14B quantized checkpoints.
LOCAL_DIRS = {
    "inference-optimization/Ministral-3-14B-Instruct-2512.w4a16": "Ministral-3-14B-Instruct-2512-GPTQ-W4A16",
}


def checkpoint(model_key: str, precision: str = "fp16") -> str:
    """Local directory (``$PROBEDIR_MODELS/<name>``, default ``./models``) or Hugging Face id."""
    ref = (load_config()["models"][model_key]["hf_id"] if precision in ("fp16", *SIMULATED)
           else QUANTIZED_CHECKPOINTS[model_key][precision])
    name = LOCAL_DIRS.get(ref, ref.split("/")[-1])
    local = Path(os.environ.get("PROBEDIR_MODELS", REPO_ROOT / "models")) / name
    return str(local) if local.exists() else ref


def is_ministral(model_key: str) -> bool:
    return model_key.startswith("ministral")


# --------------------------------------------------------------------------- structure

def blocks(model: Any) -> list[Any]:
    """Transformer blocks (zero-based), whose outputs are the residual stream h_l."""
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is not None and hasattr(lm, "layers"):          # Mistral3ForConditionalGeneration
        return list(lm.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise RuntimeError(f"cannot locate transformer blocks in {type(model).__name__}")


def input_device(model: Any) -> Any:
    lm = getattr(getattr(model, "model", None), "language_model", None)
    emb = (lm or model).get_input_embeddings()
    return emb.weight.device


# --------------------------------------------------------------------------- simulated quantization

def rtn_quantize_(weight: Any, n_bits: int, group_size: int = GROUP_SIZE, rows_per_chunk: int = 32) -> None:
    """In-place symmetric per-group absmax round-to-nearest quantize-dequantize (float32 math).

    Each output row is split into groups of ``group_size`` input columns (zero
    padded); scale s = max|w| / (2^(b-1) - 1), clamped at 1e-10;
    w <- clamp(round(w / s), +-(2^(b-1) - 1)) * s. Processed in row chunks to bound memory.
    """
    import torch

    if weight.ndim != 2:
        return
    out_f, in_f = weight.shape
    pad = (group_size - in_f % group_size) % group_size
    limit = 2 ** (n_bits - 1) - 1
    with torch.no_grad():
        for start in range(0, out_f, rows_per_chunk):
            stop = min(out_f, start + rows_per_chunk)
            block = weight.data[start:stop]
            work = torch.nn.functional.pad(block, (0, pad)) if pad else block
            grouped = work.reshape(stop - start, -1, group_size).float()
            scale = (grouped.abs().amax(dim=-1, keepdim=True) / limit).clamp_min(1e-10)
            restored = ((grouped / scale).round().clamp(-limit, limit) * scale).reshape(stop - start, -1)
            block.copy_(restored[:, :in_f].to(block.dtype))


def rtn_quantize_native(weight: Any, n_bits: int, group_size: int = GROUP_SIZE) -> Any:
    """Same quantizer with group statistics computed in the weight dtype (float16).

    The 7-8B TriviaQA quantized representations were produced this way; all
    other conditions used :func:`rtn_quantize_`. Returns the new weight tensor.
    """
    import torch

    out_f, in_f = weight.shape
    w = weight
    if in_f % group_size:
        w = torch.nn.functional.pad(w, (0, group_size - in_f % group_size))
    g = w.reshape(out_f, -1, group_size)
    q_max = 2 ** (n_bits - 1) - 1
    scale = (g.abs().amax(dim=-1, keepdim=True) / q_max).clamp(min=1e-10)
    return ((g / scale).round().clamp(-q_max, q_max) * scale).reshape(out_f, -1)[:, :in_f].to(weight.dtype)


def apply_rtn(model: Any, n_bits: int, upcast: bool = True) -> int:
    """Quantize every Linear weight inside the transformer blocks; returns the count.

    ``upcast=False`` selects :func:`rtn_quantize_native` (7-8B TriviaQA conditions).
    """
    import torch

    count = 0
    for block in blocks(model):
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                if upcast:
                    rtn_quantize_(module.weight, n_bits)
                else:
                    with torch.no_grad():
                        module.weight.data = rtn_quantize_native(module.weight.data, n_bits)
                count += 1
    return count


class ActivationQuantizer:
    """Per-token symmetric INT8 quantize-dequantize of every block output (W*A8)."""

    def __init__(self, model: Any):
        self.handles = [b.register_forward_hook(self._hook) for b in blocks(model)]

    @staticmethod
    def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
        state = output[0] if isinstance(output, tuple) else output
        fp = state.float()
        scale = (fp.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-10)
        q = ((fp / scale).round().clamp(-128, 127) * scale).to(state.dtype)
        return (q,) + tuple(output[1:]) if isinstance(output, tuple) else q

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- loading

def load_model(model_key: str, precision: str = "fp16", device_map: Any = "auto",
               upcast_rtn: bool = True) -> tuple[Any, Any]:
    """Load (model, tokenizer) for one model and precision condition.

    The tokenizer always comes from the FP16 reference checkpoint. Activation
    quantization hooks (W8A8/W4A8) are attached to the returned model as
    ``model._probedir_act_quant`` and stay active for its lifetime.
    """
    import torch

    if is_ministral(model_key):
        model, tok = _load_ministral(model_key, precision, device_map)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(checkpoint(model_key))
        if precision == "awq":
            from awq import AutoAWQForCausalLM

            wrapper = AutoAWQForCausalLM.from_quantized(checkpoint(model_key, "awq"), fuse_layers=False,
                                                        device_map=device_map)
            model = getattr(wrapper, "model", wrapper)
        else:
            path = checkpoint(model_key, "gptq" if precision == "gptq" else "fp16")
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map=device_map)
    if precision in SIMULATED:
        bits, act = SIMULATED[precision]
        apply_rtn(model, bits, upcast=upcast_rtn)
        if act:
            model._probedir_act_quant = ActivationQuantizer(model)
    model.eval()
    return model, tok


def _load_ministral(model_key: str, precision: str, device_map: Any) -> tuple[Any, Any]:
    import torch
    import transformers.modeling_utils as modeling_utils
    from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

    # Load-time allocator warm-up can over-reserve on 24 GB cards; it is only an optimization.
    modeling_utils.caching_allocator_warmup = lambda *_a, **_k: None
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    base = checkpoint(model_key, "fp16")
    tok = MistralCommonBackend.from_pretrained(base)
    model = Mistral3ForConditionalGeneration.from_pretrained(base, dtype=torch.float16, device_map=device_map)
    if precision == "awq":
        _restore_ministral_awq(model, Path(checkpoint(model_key, "awq")))
    elif precision == "gptq":
        _restore_ministral_gptq(model, Path(checkpoint(model_key, "gptq")))
    return model, tok


_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _restore_ministral_awq(model: Any, path: Path) -> None:
    """Dequantize the public compressed-tensors AWQ weights (4-bit, symmetric, group 32)."""
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    from compressed_tensors.quantization.quant_args import (QuantizationArgs, QuantizationStrategy,
                                                            QuantizationType)
    from safetensors import safe_open

    args = QuantizationArgs(num_bits=4, type=QuantizationType.INT, symmetric=True, group_size=32,
                            strategy=QuantizationStrategy.GROUP)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    with ExitStack() as stack:
        files = {n: stack.enter_context(safe_open(path / n, framework="pt", device="cpu"))
                 for n in sorted(set(weight_map.values()))}
        get = lambda key: files[weight_map[key]].get_tensor(key)  # noqa: E731
        for i, block in enumerate(blocks(model)):
            for name in _PROJ:
                prefix = f"language_model.model.layers.{i}.{name}"
                target = block.get_submodule(name)
                shape = tuple(int(x) for x in get(prefix + ".weight_shape").tolist())
                w = dequantize(unpack_from_int32(get(prefix + ".weight_packed"), 4, torch.Size(shape), packed_dim=1),
                               get(prefix + ".weight_scale"), args=args, dtype=torch.float16)
                with torch.no_grad():
                    target.weight.copy_(w.to(device=target.weight.device, dtype=target.weight.dtype))


def _restore_ministral_gptq(model: Any, path: Path) -> None:
    """Dequantize the public raw GPTQ checkpoint (MLP only, 4-bit, group 128, static order)."""
    import torch
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from compressed_tensors.quantization.lifecycle.forward import dequantize
    from compressed_tensors.quantization.quant_args import (ActivationOrdering, QuantizationArgs,
                                                            QuantizationStrategy, QuantizationType)
    from safetensors import safe_open

    args = QuantizationArgs(num_bits=4, type=QuantizationType.INT, symmetric=True, group_size=128,
                            strategy=QuantizationStrategy.GROUP, actorder=ActivationOrdering.STATIC)
    with safe_open(path / "consolidated.safetensors", framework="pt", device="cpu") as fh:
        for i, block in enumerate(blocks(model)):
            for src, dst in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                prefix = f"layers.{i}.feed_forward.{src}"
                target = getattr(block.mlp, dst)
                shape = tuple(int(x) for x in fh.get_tensor(prefix + ".weight_shape").tolist())
                w = dequantize(unpack_from_int32(fh.get_tensor(prefix + ".weight_packed"), 4, torch.Size(shape)),
                               fh.get_tensor(prefix + ".qscale_weight"), args=args, dtype=torch.float16)
                with torch.no_grad():
                    target.weight.copy_(w.to(device=target.weight.device, dtype=target.weight.dtype))
