import numpy as np


def rates(t, p, cls):
    pos = (t == cls)
    pred = (p == cls)
    TP = int((pos & pred).sum()); FN = int((pos & ~pred).sum())
    FP = int((~pos & pred).sum()); TN = int((~pos & ~pred).sum())
    n = len(t)
    tpr = TP / (TP + FN) if TP + FN else np.nan
    fpr = FP / (FP + TN) if FP + TN else np.nan
    prec = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return dict(TP=TP, FN=FN, FP=FP, TN=TN,
                n_pos=TP + FN, n=n, prev=(TP + FN) / n,
                tpr=tpr, fpr=fpr, prec=prec, f1=f1)


def f1_at_prev(tpr, fpr, pi):
    if not np.isfinite(tpr) or not np.isfinite(fpr):
        return np.nan, np.nan
    den = tpr * pi + fpr * (1 - pi)
    prec = tpr * pi / den if den > 0 else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0
    return prec, f1


def mean_over(rows, key):
    v = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(v)) if v else np.nan


def rates_multi(ts, ps, cls):
    out = []
    for t, p in zip(ts, ps):
        r = rates(t, p, cls)
        out.append((r["tpr"], r["fpr"], r["f1"], r["n_pos"], r["n"]))
    a = np.array(out, dtype=float)
    return dict(tpr=np.nanmean(a[:, 0]),
                fpr=np.nanmean(a[:, 1]),
                f1=np.mean(a[:, 2]),
                n_pos=a[0, 3],
                n=a[0, 4])
