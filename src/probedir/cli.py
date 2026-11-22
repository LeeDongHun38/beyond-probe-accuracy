"""Command-line entry point: ``probedir <command> ...`` (or ``python -m probedir``).

Commands, in pipeline order:

  pairs             build contrastive pairs / gold-donor control       Sec. 4, App. A.1, D.2
  extract           answer-final hidden states (FP16 or quantized)     Sec. 3.1, App. B.1
  probes            layer selection, MD direction, split-half          Sec. 3.2-3.3, Tables 10, 12
  estimator-sweep   MD vs. L2-LR directions at the selected layer      Table 1, Tables 13-14
  steer             bidirectional steering, P(Yes) readout             Sec. 3.3-3.4, Tables 15-16
  null-family       random-direction specificity tests + BH            App. C.4, Tables 17-19
  transfer          fixed FP16 direction on quantized models           Table 2, App. E.1, E.3
  geometry          quantization-error geometry + matched null         Sec. 3.5, Fig. 4, App. E.2
  geometry-bh       BH-FDR over the geometry test families             Tables 26-28
  template-control  prompt-template and input-length control           App. D.1, Table 20
  controls          validity controls                                  App. D.2-D.6, Tables 21-24
"""
from __future__ import annotations

import argparse
import importlib
import sys

# Modules are imported lazily so CPU-only commands do not require torch/transformers.
MODULES = ("data", "extract", "probes", "steering", "nulls", "transfer", "geometry", "controls")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probedir", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in MODULES:
        module = importlib.import_module(f"probedir.{name}")
        module.add_cli(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
