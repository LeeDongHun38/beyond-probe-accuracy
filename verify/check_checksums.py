#!/usr/bin/env python3
"""Verify release file hashes and run the numerical checks for the paper."""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest() -> list[str]:
    errors = []
    with open(ROOT / "RELEASE_MANIFEST.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        p = ROOT / r["release_path"]
        if not p.is_file():
            errors.append(f"missing: {r['release_path']}")
        elif sha256(p) != r["sha256"]:
            errors.append(f"sha256 mismatch: {r['release_path']}")
    print(f"RELEASE_MANIFEST.csv: {len(rows)} files, {len(errors)} problems")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stats", action="store_true", help="check file hashes only")
    a = ap.parse_args()
    errors = check_manifest()
    if not a.skip_stats:
        result = subprocess.run([sys.executable, str(ROOT / "verify/verify_paper_numbers.py")])
        if result.returncode:
            errors.append("Paper numerical verification failed")
    for e in errors:
        print("  " + e)
    print("PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
