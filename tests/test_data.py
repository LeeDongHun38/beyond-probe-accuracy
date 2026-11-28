"""Tests for probedir.data (pair construction and the gold-donor control)."""
from types import SimpleNamespace

from probedir import data


def _records(n):
    return [{"id": f"q{i}", "question": f"question {i}?", "correct_answer": f"right {i}",
             "wrong_answers": [f"wrong {i}a", f"wrong {i}b", f"wrong {i}c"], "source_dataset": "arc_easy"}
            for i in range(n)]


def test_mc_pairs_deterministic_and_balanced():
    a = data.build_mc_pairs(_records(101))
    b = data.build_mc_pairs(_records(101))
    assert a == b
    assert sum(p["split"] == "train" for p in a) == 50
    assert {p["template_id"] for p in a} <= {0, 1, 2}
    for p, r in zip(a, _records(101)):
        assert p["false_answer"] in r["wrong_answers"] and p["true_answer"] == r["correct_answer"]


def test_triviaqa_pairs_drop_missing_distractors():
    recs = [{"id": f"t{i}", "question": f"q{i}", "ground_truth": f"A{i}", "ground_truth_aliases": [f"A{i}"]}
            for i in range(10)]
    distractors = {f"t{i}": (None if i == 3 else f"B{i}") for i in range(10)}
    pairs = data.build_triviaqa_pairs(recs, distractors)
    assert len(pairs) == 9 and all(p["id"] != "t3" for p in pairs)
    assert data.split_counts(pairs) == {"train": 4, "test": 5}


def test_distractor_generation_retries_on_alias():
    replies = iter(["Paris", "paris", "Lyon"])

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])))

    rec = {"id": "x", "question": "Capital of France?", "ground_truth": "Paris", "ground_truth_aliases": ["Paris"]}
    assert data.generate_triviaqa_distractors([rec], Client(), sleep=lambda s: None) == {"x": "Lyon"}


def test_answer_profile_and_cost():
    assert data.answer_profile("1066")["number_kind"] == "year"
    assert data.answer_profile("PULSAR")["case_kind"] == "upper"
    p, q = data.answer_profile("John Smith"), data.answer_profile("Mary Jones")
    assert data.match_cost(p, q) == 0.0
    assert data.match_cost(p, data.answer_profile("1999")) > 60


def test_gold_donor_control_properties():
    pairs = [{"id": f"t{i}", "question": f"Who is number {i}?", "true_answer": f"Person {chr(65 + i)}",
              "false_answer": "x", "ground_truth_aliases": [f"Person {chr(65 + i)}"],
              "split": "train" if i % 2 else "test"} for i in range(20)]
    out, assignments, review = data.build_gold_donor_control(pairs, review_size=5)
    by_id = {p["id"]: p for p in pairs}
    assert len(out) == 20 and len(review) == 5
    for a in assignments:
        assert a["target_id"] != a["donor_id"]
        assert by_id[a["target_id"]]["split"] == by_id[a["donor_id"]]["split"]
    assert len({a["donor_id"] for a in assignments}) == 20  # donors used once
    assert out == data.build_gold_donor_control(pairs, review_size=5)[0]


def test_triviaqa_partition_keeps_repeated_normalized_questions_together():
    from collections import defaultdict

    partitions = defaultdict(set)
    for row in data.load_pairs("triviaqa"):
        partitions[data.question_key(row["question"])].add(row["split"])
    assert len(partitions) == 4406
    assert all(len(sides) == 1 for sides in partitions.values())
    assert sum(sides == {"train"} for sides in partitions.values()) == 2203
