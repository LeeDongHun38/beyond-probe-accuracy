#!/usr/bin/env python3
"""Integrity checks for the release.

1. Every file listed in RELEASE_MANIFEST.csv exists and matches its sha256.
2. The portable statistics package (artifacts/stats_package) rebuilds all of
   its BH-family CSVs byte-for-byte from its raw null draws
   (portable_recalculate.py --verify-reference), into a temporary directory.

Usage (repository root):  python verify/check_checksums.py [--skip-stats]
Exit status is non-zero on any failure.  Requires NumPy and SciPy for step 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
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


def check_stats_package() -> list[str]:
    pkg = ROOT / "artifacts/stats_package"
    with tempfile.TemporaryDirectory() as tmp:
        record = Path(tmp) / "verification.json"
        proc = subprocess.run([sys.executable, str(pkg / "portable_recalculate.py"), "--output-dir",
                               str(Path(tmp) / "rebuild"), "--verify-reference", str(pkg),
                               "--verification-record", str(record)], capture_output=True, text=True)
        if proc.returncode != 0 or not record.exists():
            return [f"stats package recalculation failed: {proc.stderr.strip()[-500:]}"]
        rec = json.loads(record.read_text())
    bad = [f["file"] for f in rec.get("files", []) if not f.get("byte_identical")]
    print(f"stats_package: status={rec.get('status')}, {len(rec.get('files', []))} CSVs rebuilt, "
          f"{len(bad)} differ")
    return [f"stats package CSV differs: {b}" for b in bad] + ([] if rec.get("status") == "PASS" else ["stats package status != PASS"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stats", action="store_true", help="skip the NumPy/SciPy recalculation")
    a = ap.parse_args()
    errors = check_manifest()
    if not a.skip_stats:
        errors += check_stats_package()
    for e in errors:
        print("  " + e)
    print("PASS" if not errors else "FAIL")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
