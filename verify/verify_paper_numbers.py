"""Rebuild the final numerical tables, compare with the PDF, and audit the archived runs."""
import csv
import hashlib
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))
from make_tables import ART, MODELS, build, fixed_null, geometry_panel

PDF_HASH = "207a6cce60fa0efef236cb32d5c9fa4f7b61b7fb28213b809a6b0797b6918c15"
# Printed identifiers uniquely select one row in both the reference and rebuilt tables.
KEYS = set("model model_name dataset estimator precision side condition family scaling lambda_value quantization statistic".split())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row, columns):
    return tuple(float(row[c]) if c == "lambda_value" else row[c] for c in columns)


def compare_tables(dest):
    sources = sorted((ROOT / "verify/expected").glob("table*.csv"))
    assert len(sources) == len(list(dest.glob("table*.csv"))) == 22
    notes_path = ROOT / "verify/known_discrepancies.csv"
    notes = {(r["table"], r["model"], r["dataset"], r["column"]): r for r in rows(notes_path)}
    seen = set()
    count = 0
    problems = []
    for source in sources:
        actual_file, = dest.glob(source.stem + "_*.csv")
        assert actual_file.read_bytes() == (ROOT / "paper/tables" / actual_file.name).read_bytes(), actual_file.name
        expected = rows(source)
        actual = rows(actual_file)
        cols = [c for c in expected[0] if c in KEYS or (source.stem == "table12" and c == "layer")]
        index = {key(r, cols): r for r in actual}
        assert len(index) == len(expected) == len(actual)
        assert set(index) == {key(r, cols) for r in expected}
        for row in expected:
            result = index[key(row, cols)]
            for field, printed in row.items():
                if field in cols or not printed:
                    continue
                count += 1
                name = field.removeprefix("abs_").removeprefix("pct_")
                value = float(result[name])
                if field.startswith("abs_"):
                    value = abs(value)
                if field.startswith("pct_"):
                    value *= 100
                if printed.startswith("<"):
                    matches = 0 <= value < float(printed[1:])
                else:
                    precision = len(printed.split(".")[1]) if "." in printed else 0
                    tolerance = 0.5 * 10**(-precision) if precision else 0
                    matches = abs(value - float(printed)) <= tolerance + 1e-11
                if not matches:
                    k = (source.stem, row.get("model"), row.get("dataset"), field)
                    note = notes.get(k)
                    if note and note["printed"] == printed and np.isclose(value, float(note["source"]), rtol=0, atol=1e-11):
                        seen.add(k)
                    else:
                        problems.append((source.stem, key(row, cols), field, printed, value))
    if problems:
        raise ValueError(f"Unexplained paper differences: {problems}")
    assert seen == set(notes), "Stale discrepancy entries"
    print(f"{len(sources)} numeric tables, {count} printed values, {len(seen)} documented rounding differences")


def check_runs():
    split = pd.read_csv(ART / "v2/split_manifest.csv")
    assert split.row_id.is_unique
    for side, n in (("train", 2508), ("test", 2491)):
        subset = split[split.split == side]
        pairs = json.loads((ROOT / f"data/pairs/triviaqa/{side}.json").read_text())
        assert subset.row_id.tolist() == [r["id"] for r in pairs]
        assert len(pairs) == n and subset.question_key.nunique() == 2203
    assert split.question_key.nunique() == 4406
    sweep = pd.read_csv(ART / "v2/table1_comparison.csv")
    transfer = pd.read_csv(ART / "v3/A/results.csv")
    for model in MODELS:
        profile = pd.read_csv(ART / f"v2/models/{model}/layer_train_auroc.csv")
        layer = profile.loc[profile.train_auroc.idxmax(), "layer"]
        est = sweep[sweep.model_name == MODELS[model].replace("Ministral-3", "Ministral")]
        assert len(est) == 4 and (est.layer == layer).all()
        partitions = pd.read_csv(ART / f"v3/manifests/threshold_{model}.csv")
        evaluation = partitions.assignment == "evaluation"
        assert partitions.row_id.tolist() == split.loc[split.split == "test", "row_id"].tolist()
        assert partitions.groupby("question_key").assignment.nunique().max() == 1
        with np.load(ART / f"v2/models/{model}/test_scores.npz") as scores:
            np.testing.assert_array_equal(scores["row_id"], partitions.row_id)
            for r in est.itertuples():
                np.testing.assert_allclose(auc(scores[r.estimator+"_true"], scores[r.estimator+"_false"]), r.auc)
        with np.load(ART / f"v3/A/{model}_fp16_scores.npz") as fp:
            cal = ~evaluation
            p, m = fp["true"], fp["false"]
            cal_p, cal_m = p[cal], m[cal]

            def threshold_key(t):
                return ((np.mean(cal_p >= t) + np.mean(cal_m < t)) / 2, t)

            candidates = np.r_[np.unique(np.r_[p[cal], m[cal]]), np.inf]
            # The run evaluated all distinct FP16 calibration scores, breaking ties upward.
            t = max(candidates, key=threshold_key)
            refs = np.r_[p[evaluation], m[evaluation]] >= t
        for r in transfer[transfer.model == model].itertuples():
            with np.load(ART / f"v3/A/{model}_{r.precision}_scores.npz") as s:
                np.testing.assert_array_equal(s["row_id"], partitions.row_id)
                p, m = s["true"], s["false"]
            np.testing.assert_allclose(auc(p, m), r.full_test_auroc)
            p, m = p[evaluation], m[evaluation]
            decisions = np.r_[p, m] >= t
            np.testing.assert_allclose([t, auc(p, m), (m >= t).mean(), (p < t).mean()],
                                       [r.threshold, r.evaluation_auroc, r.fpr, r.fnr])
            assert (decisions != refs).sum() == r.flip_numerator and len(refs) == r.flip_denominator
    scale = pd.read_csv(ART / "v2/table14_results.csv")
    assert len(scale) == 66 and (scale.lr_auc > scale.md_auc).all() and (scale.lr_stab < scale.md_stab).all()
    assert (scale.ci_low > 0).sum() == 64
    by_name = {v: k for k, v in MODELS.items()}
    for row in scale.itertuples():
        name = by_name[row.model_name]
        folder = ART / "v2/table14" / name / f"L{row.layer}"
        with np.load(folder / f"{row.scaling}_lambda{row.lambda_value:g}_scores.npz") as fit:
            np.testing.assert_allclose(auc(fit["true"], fit["false"]), row.lr_auc)
            np.testing.assert_allclose(np.quantile(fit["delta_bootstrap"], [.025, .975]), [row.ci_low, row.ci_high])
    generation = pd.read_csv(ART / "v4/E15/results.csv")
    for row in generation.itertuples():
        folder = ART / "v4/E15" / row.model
        with np.load(folder / "scores.npz") as scores:
            observed_auc = roc_auc_score(scores["labels"], scores["post"])
            observed_rho = spearmanr(scores["pre"], scores["p_correct"]).statistic
        random = pd.read_csv(folder / "random_effects.csv")
        assert len(random) == 1000
        np.testing.assert_allclose([observed_auc, observed_rho], [row.auc, row.rho])
        np.testing.assert_allclose([(1 + (random.auc >= observed_auc).sum()) / 1001,
                                    (1 + (random.rho >= observed_rho).sum()) / 1001],
                                   [row.auc_random_p, row.rho_random_p])

    null = fixed_null()
    assert len(null) == 30 and (null.n == 127).all() and (null.q < .05).all()
    exception = null.loc[null.exceedance > 0, ["model", "dataset", "side", "exceedance"]]
    assert exception.values.tolist() == [["mistral", "triviaqa", "false", 1]]
    geo = geometry_panel().set_index(["model", "dataset", "quantization", "statistic"]).sort_index()
    archived = pd.read_csv(ART / "v3/B/results_75_tests.csv").set_index(geo.index.names).sort_index()
    assert geo.index.equals(archived.index)
    fields = ["observed", "null_mean", "p_upper", "q_bh_family"]
    np.testing.assert_allclose(geo[fields], archived[fields], atol=1e-14, rtol=0)
    print("Verified question partitions, projection AUROCs, FP16 thresholds, decision changes, steering and geometry families")


def auc(pos, neg):
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


if __name__ == "__main__":
    assert hashlib.sha256((ROOT / "paper/Beyond Probe Accuracy.pdf").read_bytes()).hexdigest() == PDF_HASH
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        compare_tables(Path(tmp))
    check_runs()
