import os
import glob
import pickle
import numpy as np
import pandas as pd


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def load_loro(base_dir, scale):
    d = os.path.join(base_dir, scale)
    p = first_existing(
        [os.path.join(d, "loro_result_" + scale + ".pkl")]
        + glob.glob(os.path.join(d, "loro_result_*.pkl"))
    )
    if p is None:
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def load_indomain(base_dir, scale, exp_name):
    d = os.path.join(base_dir, scale)
    p = first_existing(
        [os.path.join(d, "ablation_checkpoint_" + scale + ".pkl")]
        + glob.glob(os.path.join(d, "ablation_checkpoint_*.pkl"))
    )
    if p is None:
        return None
    with open(p, "rb") as f:
        R = pickle.load(f)
    G = R.get("GLOBAL_RESULTS", R) if isinstance(R, dict) else {}
    for k in (exp_name, "Model_E_Ultimate_V59", "Model_A_Baseline_V59"):
        if k in G and isinstance(G[k], dict) and all(
            x in G[k] for x in ("preds", "trues", "val_indices")
        ):
            r = G[k]
            return (np.asarray(r["val_indices"]),
                    np.asarray(r["trues"]),
                    np.asarray(r["preds"]))
    return None


def load_samples(data_dir, scale, pos_tmpl, neg_tmpl):
    pos = first_existing([os.path.join(data_dir, pos_tmpl.format(sc=scale))])
    neg = first_existing([os.path.join(data_dir, neg_tmpl.format(sc=scale))])
    if pos is None or neg is None:
        return None
    df = pd.concat([pd.read_csv(pos), pd.read_csv(neg)], ignore_index=True)
    lat = next(c for c in ["latitude", "lat", "y"] if c in df.columns)
    lon = next(c for c in ["longitude", "lon", "x"] if c in df.columns)
    df["locid"] = (df[lon].round(6).astype(str)
                   + "_" + df[lat].round(6).astype(str))
    return df


def region_letters(base_dir, scale):
    d = os.path.join(base_dir, scale)
    p = first_existing(
        [os.path.join(d, scale + "_LORO_per_region.csv")]
        + glob.glob(os.path.join(d, "*_LORO_per_region.csv"))
    )
    if p is None:
        return None
    t = pd.read_csv(p)
    col = next((c for c in t.columns
                if c.startswith("drift_NDVI_peak_t") and c.endswith("_smd")),
               None)
    if col is None:
        return None
    order = t.sort_values(col)["region"].astype(int).tolist()
    return {r: "ABCDE"[i] for i, r in enumerate(order)}
