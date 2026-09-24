from evalsupport.simulation import decomp_states, shapley


def test_shapley_identity():
    D = decomp_states(0.5, 0.6, 0.55, 0.65, w=0.4)
    s = shapley(D)
    total = s["weight"] + s["base"] + s["resid"]
    assert abs(total - s["total"]) < 1e-12


def test_shapley_order_average():
    D = decomp_states(0.5, 0.6, 0.55, 0.65, w=0.4)
    s = shapley(D)
    o1w = D["S0"] - D["SW"]
    o2w = D["SP"] - D["S2"]
    assert abs(s["weight"] - (o1w + o2w) / 2) < 1e-12
