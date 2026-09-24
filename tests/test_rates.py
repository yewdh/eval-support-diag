import numpy as np
from evalsupport.rates import rates, f1_at_prev


def test_rates_basic():
    t = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    p = np.array([1, 1, 1, 2, 2, 2, 2, 1])
    r = rates(t, p, 1)
    assert r["n_pos"] == 4
    assert 0 <= r["tpr"] <= 1
    assert 0 <= r["fpr"] <= 1
    assert 0 <= r["f1"] <= 1


def test_f1_at_prev_consistency():
    tpr, fpr = 0.8, 0.05
    _, f1 = f1_at_prev(tpr, fpr, 0.2)
    assert 0 <= f1 <= 1
