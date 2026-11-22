#!/usr/bin/env python3
"""Check every regenerated paper table and key in-text number against the PDF.

1. Regenerates all tables with paper/make_tables.py into a temporary directory.
2. Compares them cell by cell with verify/expected/tableNN.csv (values
   transcribed from EACL_2026.pdf, sha256 b85d714962864d9051ce5940e021a114d02cf4e64b9c41dcef620a87418d50ad)
   at the printed precision.
3. Recomputes the in-text claims listed in verify/expected/in_text_claims.csv.
4. Classifies each difference as explained (listed, with the exact regenerated
   value, in verify/known_discrepancies.csv) or unexplained.

Writes verify/VERIFY_REPORT.md and verify/verify_report.json and exits
non-zero if any unexplained mismatch remains.  Expected values are never
edited to make a check pass.

Usage (repository root):  python verify/verify_paper_numbers.py
Dependencies: Python 3.9+ standard library only.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "verify"
EXP = VER / "expected"
PDF_SHA256 = "b85d714962864d9051ce5940e021a114d02cf4e64b9c41dcef620a87418d50ad"
KEY_COLS = {"estimator", "model", "dataset", "dir", "analysis", "scope", "quantization"}


def norm(v: str) -> str:
    v = (v or "").strip().replace("−", "-").replace(",", "")
    if v.startswith("+"):
        v = v[1:]
    return "-" if v in ("", "-", "–", "NA") else v


def read(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_known() -> dict:
    known = {}
    for r in read(VER / "known_discrepancies.csv"):
        known[(r["table"], r["row_key"], r["column"])] = r
    return known


def compare_tables(gen_dir: Path, known: dict) -> tuple[list, list]:
    results, mismatches = [], []
    for exp_path in sorted(EXP.glob("table*.csv")):
        tid = exp_path.stem  # tableNN
        gen = sorted(gen_dir.glob(f"{tid}_*.csv"))
        if not gen:
            results.append({"table": tid, "status": "MISSING_GENERATED", "cells": 0, "mismatches": 0})
            mismatches.append({"table": tid, "row_key": "*", "column": "*", "expected": "", "generated": "",
                               "explained": False, "note": "no generated table"})
            continue
        exp_rows, gen_rows = read(exp_path), read(gen[0])
        cols = list(exp_rows[0].keys())
        keys = [c for c in cols if c in KEY_COLS]
        index = {tuple(norm(r[k]) for k in keys): r for r in gen_rows}
        n_cells = n_bad = n_explained = 0
        for er in exp_rows:
            rk = tuple(norm(er[k]) for k in keys)
            gr = index.get(rk)
            row_key = "|".join(er[k] for k in keys)
            for c in cols:
                if c in keys:
                    continue
                n_cells += 1
                ev = norm(er[c])
                gv = norm(gr.get(c, "")) if gr else "<row missing>"
                if ev == gv:
                    continue
                k = known.get((tid, row_key, c))
                explained = bool(k) and norm(k["generated"]) == gv and norm(k["expected"]) == ev
                n_bad += 1
                n_explained += explained
                mismatches.append({"table": tid, "row_key": row_key, "column": c, "expected": er[c],
                                   "generated": gr.get(c, "") if gr else "<row missing>", "explained": explained,
                                   "note": k["explanation"] if k else ""})
        status = "MATCH" if n_bad == 0 else ("KNOWN_DISCREPANCY" if n_bad == n_explained else "MISMATCH")
        results.append({"table": tid, "generated_file": str(gen[0].name), "status": status, "cells": n_cells,
                        "mismatches": n_bad, "explained": n_explained})
    return results, mismatches


def check_claims(known: dict) -> tuple[list, list]:
    spec = importlib.util.spec_from_file_location("claims", VER / "in_text_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results, mismatches = [], []
    for r in read(EXP / "in_text_claims.csv"):
        fn = getattr(mod, r["check"], None)
        if fn is None:
            results.append({**r, "computed": "", "status": "NOT_CHECKED"})
            continue
        got = str(fn())
        ok = norm(got) == norm(r["expected"])
        k = known.get(("in_text", r["claim_id"], "value"))
        explained = (not ok) and bool(k) and norm(k["generated"]) == norm(got)
        status = "MATCH" if ok else ("KNOWN_DISCREPANCY" if explained else "MISMATCH")
        results.append({**r, "computed": got, "status": status})
        if not ok:
            mismatches.append({"table": "in_text", "row_key": r["claim_id"], "column": "value",
                               "expected": r["expected"], "generated": got, "explained": explained,
                               "note": k["explanation"] if k else ""})
    return results, mismatches


def main() -> int:
    known = load_known()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "paper/make_tables.py"), "--out-dir", tmp], check=True,
                       stdout=subprocess.DEVNULL)
        tables, mm_t = compare_tables(Path(tmp), known)
    claims, mm_c = check_claims(known)
    mismatches = mm_t + mm_c
    unexplained = [m for m in mismatches if not m["explained"]]
    report = {"paper_pdf_sha256": PDF_SHA256, "tables": tables, "in_text_claims": claims,
              "mismatches": mismatches, "unexplained_mismatches": len(unexplained),
              "status": "PASS" if not unexplained else "FAIL"}
    (VER / "verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = ["# Paper-number verification report", "",
             f"Paper: `EACL_2026.pdf` (sha256 `{PDF_SHA256}`)", "",
             f"**Status: {report['status']}** — {sum(t['cells'] for t in tables)} table cells and "
             f"{len(claims)} in-text claims checked; {len(mismatches)} differences, "
             f"{len(mismatches) - len(unexplained)} explained in `known_discrepancies.csv`, "
             f"{len(unexplained)} unexplained.", "",
             "Regenerated by `python verify/verify_paper_numbers.py`; tables come from "
             "`paper/make_tables.py`, which reads only `artifacts/` and `data/`.", "",
             "## Tables", "", "| Table | Status | Cells | Differences (explained) |", "|---|---|---|---|"]
    for t in tables:
        lines.append(f"| {t['table']} | {t['status']} | {t['cells']} | {t['mismatches']} ({t.get('explained', 0)}) |")
    lines += ["", "## In-text claims", "", "| Claim | Location | Expected | Computed | Status |", "|---|---|---|---|---|"]
    for c in claims:
        lines.append(f"| {c['claim_id']} | {c['location']} | {c['expected']} | {c['computed']} | {c['status']} |")
    if mismatches:
        lines += ["", "## Differences", "", "| Where | Row | Column | Paper | Regenerated | Explained | Note |",
                  "|---|---|---|---|---|---|---|"]
        for m in mismatches:
            lines.append(f"| {m['table']} | {m['row_key']} | {m['column']} | {m['expected']} | {m['generated']} | "
                         f"{'yes' if m['explained'] else '**NO**'} | {m['note']} |")
    (VER / "VERIFY_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"{report['status']}: {len(mismatches)} differences, {len(unexplained)} unexplained "
          f"(see verify/VERIFY_REPORT.md)")
    return 0 if not unexplained else 1


if __name__ == "__main__":
    sys.exit(main())
