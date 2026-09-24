#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Shapley decomposition + parcel-level cluster bootstrap.
import argparse
import os
import sys

import pandas as pd

from evalsupport.config import load_config
from evalsupport.decomposition import decompose_block
from evalsupport.bootstrap import (bootstrap_shares, summarize_by_scale,
                                   summarize_cross_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_dir = args.base_dir or cfg["paths"]["base_dir"]
    data_dir = args.data_dir or cfg["paths"]["data_dir"]
    out_dir = args.out or base_dir
    os.makedirs(out_dir, exist_ok=True)

    rates_csv = os.path.join(out_dir, "pr_rates_by_region.csv")
    if not os.path.exists(rates_csv):
        print("Missing %s; run run_pr_decomposition.py first." % rates_csv)
        return 1

    d = pd.read_csv(rates_csv)
    pos_tmpl = cfg["file_templates"]["pos"]
    neg_tmpl = cfg["file_templates"]["neg"]
    exp_name = cfg["exp_name"]
    scales = cfg["scales"]
    n_boot = args.n_boot or cfg["bootstrap"]["n_boot"]
    seed = args.seed if args.seed is not None else cfg["bootstrap"]["seed"]
    min_gap = cfg["bootstrap"]["min_total_gap"]
    lo_pct = cfg["bootstrap"]["ci_low"]
    hi_pct = cfg["bootstrap"]["ci_high"]

    raw_rows = []
    for setting in ["ood", "in"]:
        for scale in scales:
            b = d[(d.setting == setting) & (d.scale == scale)]
            if b.empty or "E" not in set(b.label):
                continue
            r = decompose_block(b)
            r.update(setting=setting, scale=scale,
                     e_region=int(b[b.label == "E"].region.iloc[0]))
            raw_rows.append(r)
    R = pd.DataFrame(raw_rows)
    if R.empty:
        print("No decomposition computed.")
        return 1
    R.to_csv(os.path.join(out_dir, "decomp_order.csv"),
             index=False, encoding="utf-8-sig")
    print("[OK] %s" % os.path.join(out_dir, "decomp_order.csv"))

    allb = []
    for _, r in R.iterrows():
        b = d[(d.setting == r.setting) & (d.scale == r.scale)]
        bs = bootstrap_shares(
            b, base_dir, data_dir, r.scale, r.setting, int(r.e_region),
            exp_name, pos_tmpl, neg_tmpl,
            n_boot=n_boot, seed=seed, min_total_gap=min_gap,
        )
        if bs is None or len(bs) < 100:
            print("[%s %s] skipped (insufficient data)" % (r.setting, r.scale))
            continue
        bs["setting"] = r.setting
        bs["scale"] = r.scale
        allb.append(bs)
        print("[%s %s] valid bootstraps: %d" % (r.setting, r.scale, len(bs)))

    if not allb:
        print("No bootstrap results.")
        return 1

    allb_df = pd.concat(allb, ignore_index=True)
    allb_df.to_csv(os.path.join(out_dir, "decomp_order_bootstrap.csv"),
                   index=False, encoding="utf-8-sig")

    ss = summarize_by_scale(allb_df, lo_pct=lo_pct, hi_pct=hi_pct)
    ss.to_csv(os.path.join(out_dir, "decomp_order_bootstrap_by_scale.csv"),
              index=False, encoding="utf-8-sig")
    print("[OK] %s" % os.path.join(out_dir, "decomp_order_bootstrap_by_scale.csv"))

    summary = summarize_cross_scale(allb_df, scales=scales,
                                    lo_pct=lo_pct, hi_pct=hi_pct)
    summary.to_csv(os.path.join(out_dir, "decomp_order_bootstrap_summary.csv"),
                   index=False, encoding="utf-8-sig")
    print("[OK] %s" % os.path.join(out_dir, "decomp_order_bootstrap_summary.csv"))

    if not summary.empty:
        for _, r in summary.iterrows():
            print("")
            print("[%s] n_rep=%d" % (r["setting"].upper(), int(r["n_rep"])))
            print("  weight: %.1f%% [%.1f, %.1f]" % (
                r["weight_mean"] * 100,
                r["weight_ci_low"] * 100,
                r["weight_ci_high"] * 100))
            print("  base:   %.1f%% [%.1f, %.1f]" % (
                r["base_mean"] * 100,
                r["base_ci_low"] * 100,
                r["base_ci_high"] * 100))
            print("  resid:  %.1f%% [%.1f, %.1f]" % (
                r["resid_mean"] * 100,
                r["resid_ci_low"] * 100,
                r["resid_ci_high"] * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
