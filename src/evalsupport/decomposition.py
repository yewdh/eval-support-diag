import numpy as np
import pandas as pd

from .rates import f1_at_prev


def region_states(t, m, pi_t, pi_m, w):
    ft_own = t["f1"]; fm_own = m["f1"]
    _, ft_ref = f1_at_prev(t["tpr"], t["fpr"], pi_t)
    _, fm_ref = f1_at_prev(m["tpr"], m["fpr"], pi_m)
    return dict(
        S0=(ft_own + fm_own) / 2,
        SW=w * ft_own + (1 - w) * fm_own,
        SP=(ft_ref + fm_ref) / 2,
        S2=w * ft_ref + (1 - w) * fm_ref,
    )


def decompose_block(block, e_override=None):
    ref = block[block.label != "E"]
    rt = ref[ref.cls == "tobacco"]
    rm = ref[ref.cls == "maize"]
    pi_t = rt.n_pos.sum() / rt.n.sum()
    pi_m = rm.n_pos.sum() / rm.n.sum()
    w = rt.n_pos.sum() / (rt.n_pos.sum() + rm.n_pos.sum())

    def rec(label, cls):
        x = block[(block.label == label) & (block.cls == cls)]
        return x.iloc[0].to_dict()

    others = [region_states(rec(L, "tobacco"), rec(L, "maize"), pi_t, pi_m, w)
              for L in sorted(set(ref.label))]
    om = {k: np.mean([o[k] for o in others]) for k in others[0]}

    if e_override is None:
        e = region_states(rec("E", "tobacco"), rec("E", "maize"), pi_t, pi_m, w)
    else:
        e = region_states(e_override["tobacco"], e_override["maize"],
                          pi_t, pi_m, w)

    D = {k: om[k] - e[k] for k in om}

    o1_weight = D["S0"] - D["SW"]
    o1_base = D["SW"] - D["S2"]
    o2_base = D["S0"] - D["SP"]
    o2_weight = D["SP"] - D["S2"]

    sh_weight = (o1_weight + o2_weight) / 2
    sh_base = (o1_base + o2_base) / 2
    resid = D["S2"]
    total = D["S0"]

    reconstruction = sh_weight + sh_base + resid
    if not np.isclose(total, reconstruction, atol=1e-10):
        raise RuntimeError("Shapley identity failed: %s vs %s" % (total, reconstruction))

    return dict(total=total, resid=resid,
                o1_weight=o1_weight, o1_base=o1_base,
                o2_weight=o2_weight, o2_base=o2_base,
                sh_weight=sh_weight, sh_base=sh_base)


def decompose_three_step(df, setting, scales):
    sub = df[df.setting == setting]
    rows = []
    for scale in scales:
        s = sub[sub.scale == scale]
        if s.empty or "E" not in set(s.label):
            continue
        ref = s[s.label != "E"]
        rt = ref[ref.cls == "tobacco"]
        rm = ref[ref.cls == "maize"]
        pi_t = rt.n_pos.sum() / rt.n.sum()
        pi_m = rm.n_pos.sum() / rm.n.sum()
        wt = rt.n_pos.sum() / (rt.n_pos.sum() + rm.n_pos.sum())

        def region_scores(label):
            t = s[(s.label == label) & (s.cls == "tobacco")].iloc[0]
            m = s[(s.label == label) & (s.cls == "maize")].iloc[0]
            macro = (t.f1 + m.f1) / 2
            common_w = wt * t.f1 + (1 - wt) * m.f1
            _, ft = f1_at_prev(t.tpr, t.fpr, pi_t)
            _, fm = f1_at_prev(m.tpr, m.fpr, pi_m)
            common_prev = wt * ft + (1 - wt) * fm
            return dict(macro=macro, common_w=common_w, common_prev=common_prev,
                        tob_f1=t.f1, tob_f1_adj=ft, tob_tpr=t.tpr,
                        tob_prec=t.prec, tob_fpr=t.fpr, tob_prev=t.prev)

        e = region_scores("E")
        others = [region_scores(L) for L in sorted(set(ref.label))]
        om = {k: float(np.mean([o[k] for o in others])) for k in e}
        rows.append(dict(scale=scale, setting=setting,
                         pi_ref_tob=pi_t, pi_ref_mai=pi_m, w_ref_tob=wt,
                         **{("E_" + k): v for k, v in e.items()},
                         **{("O_" + k): v for k, v in om.items()},
                         tpr_min_others=float(np.min([o["tob_tpr"] for o in others])),
                         tpr_max_others=float(np.max([o["tob_tpr"] for o in others]))))
    return pd.DataFrame(rows)
