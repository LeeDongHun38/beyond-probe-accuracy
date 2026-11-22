import numpy as np
import pytest

from probedir import nulls


def test_random_directions_unit_and_reproducible():
    pytest.importorskip("torch")
    a = nulls.random_directions(32, n=7, seed=123)
    b = nulls.random_directions(32, n=7, seed=123)
    assert a.shape == (7, 32) and a.dtype == np.float32
    assert np.array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert not np.array_equal(a, nulls.random_directions(32, n=7, seed=124))


def test_seed_registry_covers_all_cells():
    from probedir.config import all_cells

    for cell in all_cells():
        assert (cell.model, cell.dataset) in nulls.RANDOM_DIRECTION_SEEDS
    assert len(set(nulls.RANDOM_DIRECTION_SEEDS.values())) == 15


def test_specificity_test_counts_exceedances():
    probe = [0.40, 0.44]                                   # two seeds, mean .42
    randoms = np.array([[0.1, 0.1], [-0.5, -0.4], [0.2, 0.3]])  # T = .1, .45, .25
    out = nulls.specificity_test(probe, randoms)
    assert out["T_probe"] == pytest.approx(0.42)
    assert out["exceedance"] == 1                          # the opposite-sign .45 counts
    assert out["p_mc"] == pytest.approx(2 / 4)
    assert out["null_max"] == pytest.approx(0.45)


def test_family_bh_and_long_format():
    rows = []
    for side, probe in (("false", 0.5), ("true", -0.5)):
        rows.append({"model": "m", "dataset": "d", "side": side, "direction": "probe", "seed": 1, "delta_p": probe})
        for j in range(127):
            rows.append({"model": "m", "dataset": "d", "side": side, "direction": f"R{j:03d}", "seed": 1,
                         "delta_p": 0.01 * ((j % 5) - 2)})
    tests = nulls.null_from_effects(rows)
    assert len(tests) == 2
    for t in tests:
        assert t["exceedance"] == 0 and t["p_mc"] == pytest.approx(1 / 128)
        assert t["q_bh"] == pytest.approx(1 / 128)
