#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Per-region per-class rates and three-step decomposition.
import argparse
import os
import sys

import numpy as np
import pandas as pd

from evalsupport.config import load_config
from evalsupport.io import load_loro, load_indomain, region_letters
from evalsupport.rates import rates, mean_over
from evalsupport.decomposition import decompose_three_step


def collect_ood(scale, letters, base_dir):
    R = load_loro(base_dir, scale)
    if R is None:
        return []
    out = []
    for reg, res in R.get("region_results", {}).items():
        preds = res.get("ood_preds")
        if not preds:
            continue
        L = letters.get(int(reg), "R%d" % int(reg))
        for cls, cname in [(1, "tobacco"), (2, "maize")]:
            per_seed = [rates(np.asarray(pr["y_true"]),
                              np.asarray(pr["y_pred"]), cls) for pr in preds]
            rec = {k: mean_over(per_seed, k)
                   for k in ("prev", "tpr", "fpr", "prec", "f1")}
            rec.update(scale=scale, region=int(reg), label=L, cls=cname,
                       setting="ood",
                       n_pos=per_seed[0]["n_pos"], n=per_seed[0]["n"],
                       n_seeds=len(per_seed))
            out.append(rec)
    return out


def collect_indomain(scale, letters, base_dir, exp_name):
    ind = load_indomain(base_dir, scale, exp_name)
    R = load_loro(base_dir, scale)
    if ind is None or R is None or R.get("region_labels") is None:
        return []
    vidx, trues, preds = ind
    reg_all = np.asarray(R["region_labels"])
    n = min(len(vidx), len(trues), len(preds))
    vidx, trues, preds = vidx[:n], trues[:n], preds[:n]
    sreg = reg_all[vidx]
    out = []
    for reg in sorted(np.unique(sreg)):
        m = sreg == reg
        L = letters.get(int(reg), "R%d" % int(reg))
        for cls, cname in [(1, "tobacco"), (2, "maize")]:
            r = rates(trues[m], preds[m], cls)
            r.update(scale=scale, region=int(reg), label=L, cls=cname,
                     setting="in", n_seeds=1)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--base-dir", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_dir = args.base_dir or cfg["paths"]["base_dir"]
    data_dir = args.data_dir or cfg["paths"]["data_dir"]
    out_dir = args.out or base_dir
    os.makedirs(out_dir, exist_ok=True)

    scales = cfg["scales"]
    exp_name = cfg["exp_name"]

    all_rows = []
    for scale in scales:
        letters = region_letters(base_dir, scale)
        if letters is None:
            print("[%s] no region-letter map, skipping" % scale)
            continue
        a = collect_ood(scale, letters, base_dir)
        b = collect_indomain(scale, letters, base_dir, exp_name)
        print("[%s] ood regions: %d, in regions: %d" % (scale, len(a)//2, len(b)//2))
        all_rows += a + b

    if not all_rows:
        print("No predictions available.")
        return 1

    d = pd.DataFrame(all_rows)
    rates_path = os.path.join(out_dir, "pr_rates_by_region.csv")
    d.to_csv(rates_path, index=False, encoding="utf-8-sig")
    print("[OK] %s" % rates_path)

    decs = []
    for setting in ["ood", "in"]:
        dec = decompose_three_step(d, setting, scales)
        if not dec.empty:
            decs.append(dec)
    if decs:
        dec_path = os.path.join(out_dir, "pr_decomposition.csv")
        pd.concat(decs).to_csv(dec_path, index=False, encoding="utf-8-sig")
        print("[OK] %s" % dec_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
