# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from evalsupport.simulation import build_scenarios, run_scenario


def _toy_g():
    rows = []
    for L, n, nt, nm in [("A", 1000, 100, 200), ("B", 1000, 120, 220),
                         ("C", 1000, 300, 400), ("D", 1000, 90, 180),
                         ("E", 800, 52, 549)]:
        rows.append(dict(label=L, cls="tobacco", tpr=0.7, fpr=0.05,
                         n=n, n_pos=nt, f1=0.5))
        rows.append(dict(label=L, cls="maize", tpr=0.8, fpr=0.05,
                         n=n, n_pos=nm, f1=0.6))
    return pd.DataFrame(rows)


def test_scenarios_run():
    g = _toy_g()
    rng = np.random.RandomState(0)
    for name, (e_t, e_m, comp) in build_scenarios(g).items():
        truth, est, _, _ = run_scenario(g, e_t, e_m, comp,
                                        n_mc=200, rng=rng)
        for k in ["total", "weight", "base", "resid"]:
            assert np.isfinite(truth[k])
            assert np.all(np.isfinite(est[k]))


def test_toy_csv_pipeline():
    p = os.path.join("data", "toy", "pr_rates_by_region_toy.csv")
    if not os.path.exists(p):
        return  # toy 数据不存在则跳过

    d = pd.read_csv(p)
    d = d[d.setting == "ood"]
    g = (d.groupby(["label", "cls"])
           .agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"),
                n_pos=("n_pos", "first"), n=("n", "first"),
                f1=("f1", "mean"))
           .reset_index())

    rng = np.random.RandomState(0)
    for name, (e_t, e_m, comp) in build_scenarios(g).items():
        truth, est, _, _ = run_scenario(g, e_t, e_m, comp,
                                        n_mc=100, rng=rng)

        # 1) 真值分解恒等式
        recon = truth["weight"] + truth["base"] + truth["resid"]
        assert abs(truth["total"] - recon) < 1e-9, name

        # 2) 估计值三项均值之和约等于 total 均值
        est_total = est["weight"].mean() + est["base"].mean() + est["resid"].mean()
        assert abs(est["total"].mean() - est_total) < 1e-6, name

        # 3) 所有估计值有限
        for k in ["total", "weight", "base", "resid"]:
            assert np.all(np.isfinite(est[k])), name