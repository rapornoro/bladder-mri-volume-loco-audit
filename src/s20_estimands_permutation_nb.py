"""Additional estimands, permutation test and net benefit.
(a) Center-stratified (pair-weighted) dAUC vs volume: stratified AUC = within-center concordant pairs / all
    within-center case-control pairs; patient bootstrap stratified by center. Every center contributes (no SE dropping).
(b) Hartung-Knapp-Sidik-Jonkman (HKSJ) random-effects CI for within-center dAUC (k = 4).
(c) Permutation null for 'epoch chosen on the held-out center' optimism (3D CNN, seed-mean per-epoch predictions).
(d) Net benefit at thresholds 0.10, 0.15, 0.20, 0.30 for volume, primary model and treat-all; dNB with bootstrap CI.
(e) Part B under both estimands: pooled-prediction AUC inflation and mean per-test-set AUC (size-matched, case-mix matched).
Outputs: results/posthoc_*.csv"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, PREDS, CENTERS, load_index, load_preds, to_patient, auc, net_benefit, net_benefit_all  # noqa: E402
from s07_stats import RES  # noqa: E402

df = load_index()
B = 2000


def strat_auc(y, p, cen):
    num = den = 0.0
    for c in CENTERS:
        m = cen == c
        pos, neg = p[m & (y == 1)], p[m & (y == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        den += len(pos) * len(neg)
    return num / den


def paired_patient(arm, ref="A_volume"):
    a, r = to_patient(load_preds(arm)), to_patient(load_preds(ref))
    return a.merge(r, on=["patient_id", "center", "y"], suffixes=("", "_r"))


def strat_dauc(arm, seed=0):
    m = paired_patient(arm)
    y, p, pr, cen = m.y.values, m.p.values, m.p_r.values, m.center.values
    est = strat_auc(y, p, cen) - strat_auc(y, pr, cen)
    rng = np.random.default_rng(seed)
    idx_c = [np.where(cen == c)[0] for c in CENTERS]
    bs = []
    for _ in range(B):
        i = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_c])
        bs.append(strat_auc(y[i], p[i], cen[i]) - strat_auc(y[i], pr[i], cen[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, strat_auc(y, p, cen)


def hksj(arm, seed=0):
    m = paired_patient(arm)
    rng = np.random.default_rng(seed)
    d, v = [], []
    for c in CENTERS:
        s = m[m.center == c].reset_index(drop=True)
        d.append(auc(s.y, s.p) - auc(s.y, s.p_r))
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(s), len(s))
            bs.append(auc(s.y[i], s.p[i]) - auc(s.y[i], s.p_r[i]))
        v.append(max(np.nanvar(bs), 1e-8))  # floor: a center with ~0 variance still contributes
    d, v = np.array(d), np.array(v)
    w = 1 / v
    fe = np.sum(w * d) / w.sum()
    q = np.sum(w * (d - fe) ** 2)
    k = len(d)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (v + tau2)
    mu = np.sum(ws * d) / ws.sum()
    se_hk = np.sqrt(np.sum(ws * (d - mu) ** 2) / ((k - 1) * ws.sum()))
    t = stats.t.ppf(0.975, k - 1)
    return mu, mu - t * se_hk, mu + t * se_hk


def epoch_permutation(n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = df.y.values
    ep_by_c, te_by_c = {}, {}
    for c in CENTERS:
        z = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
        ep_by_c[c], te_by_c[c] = z["epochs"].mean(0), z["te"]

    def optimism(yy):
        p_best, p_fin = np.zeros(len(df)), np.zeros(len(df))
        for c in CENTERS:
            te, ep = te_by_c[c], ep_by_c[c]
            a = [roc_auc_score(yy[te], e) for e in ep]
            p_best[te], p_fin[te] = ep[int(np.argmax(a))], ep[-1]
        return roc_auc_score(yy, p_best) - roc_auc_score(yy, p_fin)

    obs = optimism(y)
    null = []
    for _ in range(n_perm):
        yy = y.copy()
        for c in CENTERS:
            te = te_by_c[c]
            yy[te] = rng.permutation(yy[te])
        null.append(optimism(yy))
    null = np.array(null)
    return dict(observed_optimism=obs, null_mean=null.mean(), null_p95=np.percentile(null, 95),
                p_value=(1 + np.sum(null >= obs)) / (1 + n_perm))


def net_benefit_table(arms, th=(0.10, 0.15, 0.20, 0.30), seed=0):
    rows = []
    pts = {a: to_patient(load_preds(a)).set_index("patient_id") for a in arms}
    base = pts[arms[0]]
    for a in arms:
        nb = net_benefit(pts[a].y.values, pts[a].p.values, th)
        rows.append(dict(model=a, **{f"nb_{t:.2f}": v for t, v in zip(th, nb)}))
    rows.append(dict(model="treat_all", **{f"nb_{t:.2f}": v for t, v in zip(th, net_benefit_all(base.y.values, th))}))
    rng = np.random.default_rng(seed)
    ids_c = [base.index[base.center == c].values for c in CENTERS]
    for a in arms[1:] + ["treat_all"]:
        bs = []
        for _ in range(B):
            ids = np.concatenate([rng.choice(ix, len(ix)) for ix in ids_c])
            r = pts[arms[0]].loc[ids]
            nb_ref = net_benefit(r.y.values, r.p.values, th)
            if a == "treat_all":
                nb_a = net_benefit_all(r.y.values, th)
            else:
                q = pts[a].loc[ids]
                nb_a = net_benefit(q.y.values, q.p.values, th)
            bs.append(nb_a - nb_ref)
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        rows.append(dict(model=f"dNB {a} - {arms[0]}", **{f"nb_{t:.2f}": f"{m:+.3f} ({l:+.3f}, {h:+.3f})"
                                                           for t, m, l, h in zip(th, bs.mean(0), lo, hi)}))
    return pd.DataFrame(rows)


def part_b_both():
    t6 = pd.read_csv(os.path.join(RES, "table6_inflation.csv"))
    rows = []
    for m in ["Volume (logistic)", "2D ResNet-18", "3D ResNet-18"]:
        for des in ["lesion_cv", "patient_cv"]:
            r = t6[(t6.model == m) & (t6.design == des)]
            if len(r):
                r = r.iloc[0]
                rows.append(dict(model=m, estimand="pooled out-of-fold predictions (same lesions)", design=des,
                                 inflation_vs_loco=r.inflation, lo=r.infl_lo, hi=r.infl_hi))
    key = {"volume": "Volume (logistic)", "cnn2d": "2D ResNet-18", "cnn3d": "3D ResNet-18"}
    pb = pd.read_csv(os.path.join(RES, "posthoc_partB_sizematched_singleseed.csv"))
    for k, m in key.items():
        lo_ = pb[(pb.model == k) & (pb.design == "LOCO mean fold AUC")]
        sm = pb[(pb.model == k) & (pb.design.str.startswith("size-matched"))]
        if len(lo_) and len(sm):
            rows.append(dict(model=m, estimand="mean per-test-set AUC", design="size-matched center-mixed",
                             inflation_vs_loco=sm.iloc[0].inflation_vs_loco, lo=sm.iloc[0].lo - lo_.iloc[0].lesion_auc,
                             hi=sm.iloc[0].hi - lo_.iloc[0].lesion_auc))
        f = os.path.join(WORK, "sizematched", f"{k}_volstrat.csv")
        if os.path.exists(f) and len(lo_):
            s = pd.read_csv(f).merge(df[["lesion_id", "y"]], on="lesion_id")
            fa = [auc(g.y, g.p) for _, g in s.groupby(["rep", "fold"])]
            rows.append(dict(model=m, estimand="mean per-test-set AUC", design="case-mix-matched center-mixed",
                             inflation_vs_loco=np.mean(fa) - lo_.iloc[0].lesion_auc,
                             lo=np.min(fa) - lo_.iloc[0].lesion_auc, hi=np.max(fa) - lo_.iloc[0].lesion_auc,
                             mean_auc=np.mean(fa)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    if "a" in what or "b" in what:
        arms = ["Selected_vol", "Selected_vol_patientsel", "A_diameter", "Aplus_clinvol", "B_radiomics", "B_radiomics_vol",
                "B_texture", "B_texture_vol", "C_cnn", "C_cnn_vol", "C2_mednet", "C2_mednet_vol", "D_dino", "D_dino_vol",
                "E_medsam", "E_medsam_vol"]
        rows = []
        for i, a in enumerate(arms):
            if not os.path.exists(os.path.join(PREDS, f"{a}.csv")):
                continue
            e, lo, hi, sa = strat_dauc(a, seed=i)
            mu, hlo, hhi = hksj(a, seed=100 + i)
            rows.append(dict(arm=a, strat_auc=sa, strat_dauc=e, strat_lo=lo, strat_hi=hi, hksj_dauc=mu, hksj_lo=hlo, hksj_hi=hhi))
        t = pd.DataFrame(rows)
        t["decision_strat"] = np.where(t.strat_hi < 0.05, "no meaningful increment", np.where(t.strat_lo > 0.05, "meaningful", "inconclusive"))
        t.to_csv(os.path.join(RES, "posthoc_within_center_estimands.csv"), index=False)
        print(t.round(3).to_string(index=False))
    if "c" in what:
        r = epoch_permutation()
        pd.DataFrame([r]).to_csv(os.path.join(RES, "posthoc_epoch_permutation.csv"), index=False)
        print({k: round(v, 4) for k, v in r.items()})
    if "d" in what:
        nb = net_benefit_table(["A_volume", "Selected_vol"])
        nb.to_csv(os.path.join(RES, "posthoc_net_benefit.csv"), index=False)
        print(nb.to_string(index=False))
    if "e" in what:
        pb = part_b_both()
        pb.to_csv(os.path.join(RES, "posthoc_partB_two_estimands.csv"), index=False)
        print(pb.round(3).to_string(index=False))
