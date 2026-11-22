"""Hidden-state cache format.

A cache stores answer-final residual representations for one
(model, dataset, precision) condition::

    results/hidden/<model>/<dataset>/<precision>.safetensors
        h_plus   float16 [L, n, d]   correct-answer inputs  (x+ = (q, a+))
        h_minus  float16 [L, n, d]   incorrect-answer inputs (x- = (q, a-))
    results/hidden/<model>/<dataset>/<precision>.json
        ids, split per row, layers stored, model id/revision, extraction settings

Rows follow the order of the dataset's ``data/pairs/<dataset>/all.json``. ``layers`` may be
all transformer blocks or a subset (e.g. only the FP16-selected layer for
quantized conditions).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

from .config import results_dir


def cache_path(model: str, dataset: str, precision: str = "fp16") -> Path:
    return results_dir() / "hidden" / model / dataset / f"{precision}.safetensors"


@dataclass
class HiddenCache:
    h_plus: np.ndarray            # [L, n, d]
    h_minus: np.ndarray           # [L, n, d]
    ids: list[str]
    split: list[str]              # "train" | "test" per row
    layers: list[int]             # transformer-block index of axis 0
    meta: dict[str, Any] = field(default_factory=dict)

    def layer(self, layer: int, split: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (h_plus, h_minus) as float64 [n, d] for one layer, optionally one split."""
        i = self.layers.index(layer)
        hp, hm = self.h_plus[i].astype(np.float64), self.h_minus[i].astype(np.float64)
        if split is None:
            return hp, hm
        mask = np.asarray(self.split) == split
        return hp[mask], hm[mask]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"h_plus": np.ascontiguousarray(self.h_plus, dtype=np.float16),
                   "h_minus": np.ascontiguousarray(self.h_minus, dtype=np.float16)}, str(path))
        side = {"ids": self.ids, "split": self.split, "layers": self.layers, **self.meta}
        path.with_suffix(".json").write_text(json.dumps(side, indent=1))

    @classmethod
    def load(cls, path: Path) -> "HiddenCache":
        t = load_file(str(path))
        side = json.loads(path.with_suffix(".json").read_text())
        ids, split, layers = side.pop("ids"), side.pop("split"), side.pop("layers")
        return cls(t["h_plus"], t["h_minus"], ids, split, layers, side)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
