import numpy as np
import pandas as pd

from .io import load_loro, load_indomain, load_samples
from .rates import rates_multi
from .decomposition import decompose_block


TOBACCO = 1
MAIZE = 2


def get_e_predictions(base_dir, data_dir, scale, setting, e_region,
                      exp_name, pos_tmpl, neg_tmpl):
    R = load_loro(base_dir, scale)
    if R is None or R.get("region_labels") is None:
        return None
    reg_all = np.asarray(R["region_labels"])
    df = load_samples(data_dir, scale, pos_tmpl, neg_tmpl)
    if df is None:
        return None
    locs_all = df.locid.values

    if setting == "ood":
        res = (R["region_results"].get(e_region)
               or R["region_results"].get(str(e_region)))
        if not res or not res.get("ood_preds"):
            return None
        L = locs_all[reg_all == e_region]
        ts = [np.asarray(p["y_true"]) for p in res["ood_preds"]]
        ps = [np.asarray(p["y_pred"]) for p in res["ood_preds"]]
        m = min(len(L), *[len(x) for x in ts])
        return [x[:m] for x in ts], [x[:m] for x in ps], L[:m]

    ind = load_indomain(base_dir, scale, exp_name)
    if ind is None:
        return None
    vidx, tr, pr = ind
    n = min(len(vidx), len(tr), len(pr))
    vidx, tr, pr = vidx[:n], tr[:n], pr[:n]
    msk = reg_all[vidx] == e_region
    return [tr[msk]], [pr[msk]], locs_all[vidx][msk]


def bootstrap_shares(block, base_dir, data_dir, scale, setting, e_region,
                     exp_name, pos_tmpl, neg_tmpl,
                     n_boot=2000, seed=42, min_total_gap=0.05):
    got = get_e_predictions(base_dir, data_dir, scale, setting, e_region,
                            exp_name, pos_tmpl, neg_tmpl)
    if got is None:
        return None
    ts, ps, locs = got
    uniq = pd.unique(locs)
    idx = {L: np.where(locs == L)[0] for L in uniq}
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_boot):
        sel = uniq[rng.randint(0, len(uniq), len(uniq))]
        s_ = np.concatenate([idx[L] for L in sel])
        bt = [t[s_] for t in ts]
        bp = [p[s_] for p in ps]
        if (bt[0] == TOBACCO).sum() < 3:
            continue
        eo = {"tobacco": rates_multi(bt, bp, TOBACCO),
              "maize": rates_multi(bt, bp, MAIZE)}
        rows.append(decompose_block(block, e_override=eo))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df.total > min_total_gap].copy()
    if df.empty:
        return df
    df["weight_share"] = df["sh_weight"] / df["total"]
    df["base_share"] = df["sh_base"] / df["total"]
    df["resid_share"] = df["resid"] / df["total"]
    df["base_minus_resid"] = df["base_share"] - df["resid_share"]
    df["share_sum"] = df["weight_share"] + df["base_share"] + df["resid_share"]
    return df


def percentile_ci(x, lo_pct=2.5, hi_pct=97.5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.percentile(x, lo_pct)), float(np.percentile(x, hi_pct))


def summarize_by_scale(allb, lo_pct=2.5, hi_pct=97.5):
    rows = []
    for (setting, scale), s in allb.groupby(["setting", "scale"]):
        for col, name in [("weight_share", "weight"),
                          ("base_share", "base"),
                          ("resid_share", "resid"),
                          ("base_minus_resid", "base_minus_resid")]:
            vals = s[col].values
            lo, hi = percentile_ci(vals, lo_pct, hi_pct)
            rows.append(dict(setting=setting, scale=scale, component=name,
                             mean=float(np.mean(vals)), ci_low=lo, ci_high=hi,
                             n_boot=len(vals)))
    return pd.DataFrame(rows)


def summarize_cross_scale(allb, scales=("1km", "2km", "5km"),
                          lo_pct=2.5, hi_pct=97.5):
    if allb.empty:
        return pd.DataFrame()
    summary_rows = []
    for setting in ["ood", "in"]:
        s = allb[allb.setting == setting].copy()
        if s.empty:
            continue
        s["rep"] = s.groupby("scale").cumcount()
        counts = s.groupby("rep")["scale"].nunique()
        valid = counts[counts == len(scales)].index
        s = s[s.rep.isin(valid)].copy()
        if s.empty:
            continue
        cross = (s.groupby("rep")[
                    ["weight_share", "base_share",
                     "resid_share", "base_minus_resid"]]
                 .mean().sort_index())
        row = dict(setting=setting, n_rep=len(cross), n_scale=len(scales))
        for col, prefix in [("weight_share", "weight"),
                            ("base_share", "base"),
                            ("resid_share", "resid"),
                            ("base_minus_resid", "base_minus_resid")]:
            vals = cross[col].values
            lo, hi = percentile_ci(vals, lo_pct, hi_pct)
            row[prefix + "_mean"] = float(np.mean(vals))
            row[prefix + "_ci_low"] = lo
            row[prefix + "_ci_high"] = hi
        row["share_sum_mean"] = (row["weight_mean"] + row["base_mean"]
                                 + row["resid_mean"])
        row["share_identity_error"] = row["share_sum_mean"] - 1.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)
