"""Experiment configuration.

All fixed experimental choices (models, selected layers, seeds, intervention
settings) live in ``configs/experiment.yaml``. Code reads them through
:func:`load_config` instead of hard-coding them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

def _repo_root() -> Path:
    """``$PROBEDIR_ROOT``, else the source checkout (editable install), else the working directory."""
    if "PROBEDIR_ROOT" in os.environ:
        return Path(os.environ["PROBEDIR_ROOT"])
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "configs" / "experiment.yaml").exists() else Path.cwd()


REPO_ROOT = _repo_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment.yaml"

MODELS = ("llama-3.1-8b", "qwen2.5-7b", "mistral-7b", "qwen2.5-14b", "ministral-3-14b")
DATASETS = ("triviaqa", "arc_easy", "commonsenseqa")
PRECISIONS = ("fp16", "awq", "gptq", "rtn", "w8a8", "w4a8")


def results_dir() -> Path:
    """Directory for run outputs (hidden-state caches, steering results, ...)."""
    return Path(os.environ.get("PROBEDIR_RESULTS", REPO_ROOT / "results"))


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Cell:
    """One model-dataset cell of the analysis."""

    model: str
    dataset: str

    @property
    def layer(self) -> int:
        return int(load_config()["selected_layers"][self.model][self.dataset])

    @property
    def model_cfg(self) -> dict[str, Any]:
        return load_config()["models"][self.model]

    @property
    def is_14b(self) -> bool:
        return self.model in ("qwen2.5-14b", "ministral-3-14b")

    def __str__(self) -> str:
        return f"{self.model}/{self.dataset}"


def all_cells() -> list[Cell]:
    return [Cell(m, d) for m in MODELS for d in DATASETS]


def pairs_path(dataset: str, split: str = "all") -> Path:
    """Path of the released pair file; ``split`` is ``all``, ``train`` or ``test`` (held-out)."""
    rel = load_config()["datasets"][dataset]["pairs_dir"]
    return REPO_ROOT / rel / f"{split}.json"
