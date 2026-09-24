#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Confusion-matrix simulation: Table 7 alignment + scenarios.
import argparse
import os
import sys

import numpy as np
import pandas as pd

from evalsupport.simulation import (build_scenarios, run_scenario,
                                    f1_from_counts, f1_at_prev)


def align_table7(d, mech_csv, n_mc, rng, target_n):
    rows = []
    t = d[(d.cls == "tobacco") & (d.label != "E")]
    mech = pd.read_csv(mech_csv) if mech_csv and os.path.exists(mech_csv) else None
    for _, r in t.iterrows():
        n_neg = int(r.n - r.n_pos)
        TP = rng.binomial(target_n, r.tpr, n_mc)
        FP = rng.binomial(n_neg, r.fpr, n_mc)
        f1 = f1_from_counts(TP, FP, target_n - TP)
        pi = target_n / (target_n + n_neg)
        emp = np.nan
        if mech is not None:
            m = mech[(mech.scale == r.scale) & (mech.label == r.label)]
            if len(m):
                emp = float(m.f1_sub.iloc[0])
        rows.append(dict(
            scale=r.scale, label=r.label, f1_full=r.f1,
            beta_pred=r.f1,
            cm_analytic=float(f1_at_prev(r.tpr, r.fpr, pi)),
            cm_median=float(np.median(f1)),
            cm_lo=float(np.percentile(f1, 2.5)),
            cm_hi=float(np.percentile(f1, 97.5)),
            empirical=emp,
        ))
    return pd.DataFrame(rows).sort_values(["scale", "label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True)
    ap.add_argument("--mech", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="ood")
    ap.add_argument("--n-mc", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-n", type=int, default=52)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    d = pd.read_csv(args.rates)
    d = d[d.setting == args.setting]

    g = (d.groupby(["label", "cls"])
           .agg(tpr=("tpr", "mean"), fpr=("fpr", "mean"),
                n_pos=("n_pos", "first"), n=("n", "first"),
                f1=("f1", "mean"))
           .reset_index())

    a = align_table7(d, args.mech, args.n_mc, rng, args.target_n)
    a_path = os.path.join(args.out, "sim_partA_table7.csv")
    a.to_csv(a_path, index=False, encoding="utf-8-sig")
    print("[OK] %s" % a_path)

    rows = []
    for name, (e_t, e_m, comp) in build_scenarios(g).items():
        truth, est, e_ft, e_true = run_scenario(g, e_t, e_m, comp,
                                                args.n_mc, rng)
        row = dict(scenario=name, n_tob=comp["n_t"], n=comp["n"],
                   E_tob_f1_true=e_true,
                   E_tob_f1_mean=float(np.mean(e_ft)),
                   E_tob_f1_lo=float(np.percentile(e_ft, 2.5)),
                   E_tob_f1_hi=float(np.percentile(e_ft, 97.5)))
        for k in ["total", "weight", "base", "resid"]:
            row[k + "_true"] = float(truth[k])
            row[k + "_mean"] = float(np.mean(est[k]))
            row[k + "_lo"] = float(np.percentile(est[k], 2.5))
            row[k + "_hi"] = float(np.percentile(est[k], 97.5))
        rows.append(row)
    b_path = os.path.join(args.out, "sim_partB_scenarios.csv")
    pd.DataFrame(rows).to_csv(b_path, index=False, encoding="utf-8-sig")
    print("[OK] %s" % b_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
