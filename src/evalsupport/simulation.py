import numpy as np


def f1_from_counts(TP, FP, FN):
    TP = np.asarray(TP, float); FP = np.asarray(FP, float); FN = np.asarray(FN, float)
    prec = np.where(TP + FP > 0, TP / np.maximum(TP + FP, 1), 0.0)
    rec = np.where(TP + FN > 0, TP / np.maximum(TP + FN, 1), 0.0)
    return np.where(prec + rec > 0,
                    2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)


def f1_at_prev(tpr, fpr, pi):
    tpr = np.asarray(tpr, float); fpr = np.asarray(fpr, float)
    den = tpr * pi + fpr * (1 - pi)
    prec = np.where(den > 0, tpr * pi / np.maximum(den, 1e-12), 0.0)
    return np.where(prec + tpr > 0,
                    2 * prec * tpr / np.maximum(prec + tpr, 1e-12), 0.0)


def decomp_states(ft_own, fm_own, ft_ref, fm_ref, w):
    return dict(
        S0=(ft_own + fm_own) / 2,
        SW=w * ft_own + (1 - w) * fm_own,
        SP=(ft_ref + fm_ref) / 2,
        S2=w * ft_ref + (1 - w) * fm_ref,
    )


def shapley(D):
    o1w = D["S0"] - D["SW"]; o1b = D["SW"] - D["S2"]
    o2b = D["S0"] - D["SP"]; o2w = D["SP"] - D["S2"]
    return dict(total=D["S0"],
                weight=(o1w + o2w) / 2,
                base=(o1b + o2b) / 2,
                resid=D["S2"])


def build_scenarios(g):
    ref = g[g.label != "E"]
    rt = ref[ref.cls == "tobacco"]
    rm = ref[ref.cls == "maize"]
    mean_rt = dict(tpr=rt.tpr.mean(), fpr=rt.fpr.mean())
    mean_rm = dict(tpr=rm.tpr.mean(), fpr=rm.fpr.mean())
    et = g[(g.label == "E") & (g.cls == "tobacco")].iloc[0]
    em = g[(g.label == "E") & (g.cls == "maize")].iloc[0]
    nE = int(et.n)
    p_t = rt.n_pos.sum() / rt.n.sum()
    p_m = rm.n_pos.sum() / rm.n.sum()
    normal = dict(n=nE, n_t=int(round(p_t * nE)), n_m=int(round(p_m * nE)))
    actual = dict(n=nE, n_t=int(et.n_pos), n_m=int(em.n_pos))
    big = dict(n=nE * 20, n_t=int(et.n_pos) * 20, n_m=int(em.n_pos) * 20)
    low_recall_t = dict(tpr=et.tpr, fpr=mean_rt["fpr"])
    return {
        "S1_reference":       (mean_rt, mean_rm, normal),
        "S2_base_rate_only":  (mean_rt, mean_rm, actual),
        "S3_base_rate_x20":   (mean_rt, mean_rm, big),
        "S4_classifier_only": (low_recall_t, mean_rm, normal),
        "S5_empirical":       (dict(tpr=et.tpr, fpr=et.fpr),
                               dict(tpr=em.tpr, fpr=em.fpr), actual),
    }


def run_scenario(g, e_t, e_m, e_comp, n_mc, rng):
    ref = g[g.label != "E"]
    rt = ref[ref.cls == "tobacco"]
    rm = ref[ref.cls == "maize"]
    pi_t = rt.n_pos.sum() / rt.n.sum()
    pi_m = rm.n_pos.sum() / rm.n.sum()
    w = rt.n_pos.sum() / (rt.n_pos.sum() + rm.n_pos.sum())
    labels = sorted(set(ref.label))

    def truth_states(tpr_t, fpr_t, tpr_m, fpr_m, n, n_t, n_m):
        return decomp_states(
            f1_at_prev(tpr_t, fpr_t, n_t / n),
            f1_at_prev(tpr_m, fpr_m, n_m / n),
            f1_at_prev(tpr_t, fpr_t, pi_t),
            f1_at_prev(tpr_m, fpr_m, pi_m),
            w)

    def sim_states(tpr_t, fpr_t, tpr_m, fpr_m, n, n_t, n_m):
        TPt = rng.binomial(n_t, tpr_t, n_mc)
        FPt = rng.binomial(n - n_t, fpr_t, n_mc)
        TPm = rng.binomial(n_m, tpr_m, n_mc)
        FPm = rng.binomial(n - n_m, fpr_m, n_mc)
        ft = f1_from_counts(TPt, FPt, n_t - TPt)
        fm = f1_from_counts(TPm, FPm, n_m - TPm)
        et_tpr = TPt / n_t; et_fpr = FPt / (n - n_t)
        em_tpr = TPm / n_m; em_fpr = FPm / (n - n_m)
        return decomp_states(ft, fm,
                             f1_at_prev(et_tpr, et_fpr, pi_t),
                             f1_at_prev(em_tpr, em_fpr, pi_m),
                             w), ft

    tru_o, sim_o = [], []
    for L in labels:
        a = g[(g.label == L) & (g.cls == "tobacco")].iloc[0]
        b = g[(g.label == L) & (g.cls == "maize")].iloc[0]
        args = (a.tpr, a.fpr, b.tpr, b.fpr, int(a.n), int(a.n_pos), int(b.n_pos))
        tru_o.append(truth_states(*args))
        sim_o.append(sim_states(*args)[0])
    tru_om = {k: np.mean([o[k] for o in tru_o]) for k in tru_o[0]}
    sim_om = {k: np.mean([o[k] for o in sim_o], axis=0) for k in sim_o[0]}

    eargs = (e_t["tpr"], e_t["fpr"], e_m["tpr"], e_m["fpr"],
             e_comp["n"], e_comp["n_t"], e_comp["n_m"])
    tru_e = truth_states(*eargs)
    sim_e, e_ft = sim_states(*eargs)

    truth = shapley({k: tru_om[k] - tru_e[k] for k in tru_e})
    est = shapley({k: sim_om[k] - sim_e[k] for k in sim_e})
    e_true_f1 = float(f1_at_prev(e_t["tpr"], e_t["fpr"],
                                 e_comp["n_t"] / e_comp["n"]))
    return truth, est, e_ft, e_true_f1
