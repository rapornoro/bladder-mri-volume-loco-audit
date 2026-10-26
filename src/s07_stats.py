"""Part A statistics + figures (primary analysis per SAP).
1) builds CNN arms from work/cnn/loco_c*.npz: C_cnn (3-seed mean), C_cnn_vol (stacked with log-volume on inner OOF),
   C_cnn_res (CNN logit residualized on log-volume, fit on inner OOF);
2) primary arm 'Selected_vol' = per outer fold, the +volume arm with the highest inner-CV AUC (training centers only);
3) tables: discrimination, dAUC vs volume, calibration (per center + random-effects), NRI/IDI, subgroups, DCA;
4) figures to study/results/."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import (WORK, CENTERS, load_index, save_preds, load_preds, to_patient, auc, logit, boot_metric,  # noqa: E402
                    calibration, dl_meta, auc_se_hanley, net_benefit, net_benefit_all)

RES = os.path.abspath(os.path.join(WORK, "..", "results"))
os.makedirs(RES, exist_ok=True)
B = 2000
COL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "ink": "#0b0b0b",
       "ink2": "#52514e", "grid": "#e4e3df"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["ink2"], "axes.labelcolor": COL["ink"],
                     "xtick.color": COL["ink2"], "ytick.color": COL["ink2"], "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

LABEL = {"A_volume": "Volume (reference)", "Aplus_clinvol": "Volume + age, sex, lesion count",
         "B_radiomics": "Radiomics", "B_radiomics_vol": "Radiomics + volume", "B_radiomics_res": "Radiomics, volume-residualized",
         "C_cnn": "3D CNN", "C_cnn_vol": "3D CNN + volume", "C_cnn_res": "3D CNN, volume-residualized",
         "D_dino": "DINOv2", "D_dino_vol": "DINOv2 + volume", "D_dino_res": "DINOv2, volume-residualized",
         "Selected_vol": "Pre-selected image + volume (primary)"}


def build_cnn(df):
    lv, y = df.log_vol.values, df.y.values
    p, pv, pr = (np.full(len(df), np.nan) for _ in range(3))
    sel = []
    for c in CENTERS:
        f = os.path.join(WORK, "cnn", f"loco_c{c}.npz")
        if not os.path.exists(f):
            return None
        d = np.load(f)
        te, tr, oof = d["te"], d["tr"], d["inner_oof"]
        pt = d["final"].mean(0)
        p[te] = pt
        S = np.c_[logit(oof), lv[tr]]
        stk = LogisticRegression(C=1e6, max_iter=5000).fit(S, y[tr])
        pv[te] = stk.predict_proba(np.c_[logit(pt), lv[te]])[:, 1]
        beta = np.polyfit(lv[tr], logit(oof), 1)
        pr[te] = 1 / (1 + np.exp(-(logit(pt) - np.polyval(beta, lv[te]))))
        # inner AUC of stacker for selection (5-fold on inner OOF features, patient-grouped)
        from sklearn.model_selection import StratifiedGroupKFold
        so = np.zeros(len(tr))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(S, y[tr], df.patient_id.values[tr]):
            so[b] = LogisticRegression(C=1e6, max_iter=5000).fit(S[a], y[tr][a]).predict_proba(S[b])[:, 1]
        sel.append(dict(center=c, arm="C_cnn_vol", inner_auc=roc_auc_score(y[tr], so), inner_auc_img=roc_auc_score(y[tr], oof)))
    save_preds("C_cnn", df, p)
    save_preds("C_cnn_vol", df, pv)
    save_preds("C_cnn_res", df, pr)
    return pd.DataFrame(sel)


def build_selected(df, sel):
    cand = sel[sel.arm.isin(["B_radiomics_vol", "C_cnn_vol", "D_dino_vol"])]
    choice = cand.loc[cand.groupby("center").inner_auc.idxmax()][["center", "arm", "inner_auc"]]
    p = np.full(len(df), np.nan)
    for _, r in choice.iterrows():
        pr = load_preds(r.arm)
        m = pr.center.values == r.center
        p[m] = pr.p.values[m]
    save_preds("Selected_vol", df, p)
    return choice


def disc_table(arms):
    ref = load_preds("A_volume")
    rows = []
    for a in arms:
        pr = load_preds(a)
        pt = to_patient(pr)
        fr = {"a": pr, "ref": ref}
        pa, lo, hi, _ = boot_metric({"a": pr}, lambda f: auc(f["a"].y, f["a"].p), "patient", B)
        d, dlo, dhi, ds = boot_metric(fr, lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p), "patient", B, seed=1)
        la, llo, lhi, _ = boot_metric({"a": pr}, lambda f: auc(f["a"].y, f["a"].p), "lesion", B, seed=2)
        ld, ldlo, ldhi, _ = boot_metric(fr, lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p), "lesion", B, seed=3)
        # per-center patient AUCs + random-effects pooled (logit scale)
        ce, se = [], []
        for c in CENTERS:
            s = pt[pt.center == c]
            a_c = auc(s.y, s.p)
            ce.append(a_c)
            se.append(auc_se_hanley(s.y.values, s.p.values))
        lg = np.log(np.array(ce) / (1 - np.array(ce)))
        lgse = np.array(se) / (np.array(ce) * (1 - np.array(ce)))
        pm, pmlo, pmhi, tau2, i2 = dl_meta(lg, lgse)
        inv = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
        rows.append(dict(arm=a, label=LABEL.get(a, a),
                         pt_auc=pa, pt_lo=lo, pt_hi=hi, pt_dauc=d, pt_dlo=dlo, pt_dhi=dhi,
                         p_boot_dauc_le0=float(np.mean(ds <= 0)),
                         les_auc=la, les_lo=llo, les_hi=lhi, les_dauc=ld, les_dlo=ldlo, les_dhi=ldhi,
                         **{f"auc_c{c}": v for c, v in zip(CENTERS, ce)},
                         re_auc=inv(pm), re_lo=inv(pmlo), re_hi=inv(pmhi), re_I2=i2))
    return pd.DataFrame(rows)


def calib_table(arms):
    rows = []
    for a in arms:
        pt = to_patient(load_preds(a))
        per = []
        for c in CENTERS:
            s = pt[pt.center == c]
            cal = calibration(s.y.values, s.p.values)
            cal.update(arm=a, center=c, n=len(s), events=int(s.y.sum()))
            per.append(cal)
            rows.append(cal)
        per = pd.DataFrame(per)
        for k in ["cal_intercept", "cal_slope"]:
            pm, lo, hi, tau2, i2 = dl_meta(per[k], per[f"{k}_se"])
            rows.append({"arm": a, "center": "RE-pooled", k: pm, f"{k}_lo": lo, f"{k}_hi": hi, f"{k}_I2": i2})
        allc = calibration(pt.y.values, pt.p.values)
        allc.update(arm=a, center="pooled-data")
        rows.append(allc)
    t = pd.DataFrame(rows)
    return t.groupby(["arm", "center"], sort=False).first().reset_index()


def nri_idi(arm, ref="A_volume", thr=0.30):
    def f(fr):
        a, r = fr["a"], fr["ref"]
        y = a.y.values
        up = (a.p.values >= thr) & (r.p.values < thr)
        dn = (a.p.values < thr) & (r.p.values >= thr)
        e, ne = y == 1, y == 0
        nri_e = (up[e].sum() - dn[e].sum()) / e.sum()
        nri_ne = (dn[ne].sum() - up[ne].sum()) / ne.sum()
        idi = (a.p.values[e].mean() - a.p.values[ne].mean()) - (r.p.values[e].mean() - r.p.values[ne].mean())
        return np.array([nri_e, nri_ne, nri_e + nri_ne, idi])
    return boot_metric({"a": load_preds(arm), "ref": load_preds(ref)}, f, "patient", B, seed=4)


def subgroups(df, arms):
    rows = []
    for a in arms:
        pr = load_preds(a).merge(df[["lesion_id", "vol_ml", "t_stage"]], on="lesion_id")
        for name, m in [("lesions < 3 mL", pr.vol_ml < 3), ("pT1 vs pT2 lesions", pr.t_stage.isin(["T1", "T2"]))]:
            s = pr[m]
            pa, lo, hi, _ = boot_metric({"a": s}, lambda f: auc(f["a"].y, f["a"].p), "lesion", B, seed=5)
            rows.append(dict(arm=a, subgroup=name, n=len(s), events=int(s.y.sum()), auc=pa, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def local_recal(pr):
    """Intercept-only update per held-out center (logit offset), using that center's own outcomes.
    In-sample by construction (calibration-in-the-large becomes 0); shows what local recalibration can recover."""
    import statsmodels.api as sm
    out = pr.copy()
    for c in CENTERS:
        m = (out.center == c).values
        lp = logit(out.p.values[m])
        a = sm.GLM(out.y.values[m].astype(float), np.ones(m.sum()), family=sm.families.Binomial(), offset=lp).fit().params[0]
        out.loc[m, "p"] = 1 / (1 + np.exp(-(lp + a)))
    return out


def sensitivity(df, arms):
    """SAP sensitivity analyses: (1) lesion level (in table 2); (2) excluding multifocal patients;
    (3) local intercept recalibration per held-out center (patient level: Brier, calibration slope, net benefit)."""
    rows = []
    uni = set(df.patient_id[df.n_lesions_patient == 1])
    ref = load_preds("A_volume")
    for a in arms:
        pr = load_preds(a)
        # (2) unifocal patients only, dAUC vs volume
        s, r = pr[pr.patient_id.isin(uni)], ref[ref.patient_id.isin(uni)]
        pa, lo, hi, _ = boot_metric({"a": s}, lambda f: auc(f["a"].y, f["a"].p), "patient", B, seed=6)
        d, dlo, dhi, _ = boot_metric({"a": s, "ref": r}, lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p), "patient", B, seed=7)
        rows.append(dict(arm=a, analysis="unifocal patients only", n=s.patient_id.nunique(), events=int(to_patient(s).y.sum()),
                         auc=pa, lo=lo, hi=hi, dauc=d, dlo=dlo, dhi=dhi))
        # (3) local recalibration
        for name, p in [("as transported", pr), ("local intercept recalibration", local_recal(pr))]:
            pt = to_patient(p)
            cal = calibration(pt.y.values, pt.p.values)
            nb = net_benefit(pt.y.values, pt.p.values, [0.2, 0.3])
            per_slope = [calibration(pt[pt.center == c].y.values, pt[pt.center == c].p.values)["cal_slope"] for c in CENTERS]
            per_oe = [calibration(pt[pt.center == c].y.values, pt[pt.center == c].p.values)["oe"] for c in CENTERS]
            rows.append(dict(arm=a, analysis=name, n=len(pt), events=int(pt.y.sum()), brier=cal["brier"],
                             oe_min=min(per_oe), oe_max=max(per_oe), slope_min=min(per_slope), slope_max=max(per_slope),
                             nb_020=nb[0], nb_030=nb[1]))
    return pd.DataFrame(rows)


# ---------------- figures -----------------
def fig_forest(t):
    t = t[t.arm != "A_volume"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(t) + 0.9))
    yy = np.arange(len(t))
    ax.axvspan(-0.05, 0.05, color=COL["grid"], alpha=0.6, lw=0)
    ax.axvline(0, color=COL["ink2"], lw=1)
    for i, (_, r) in enumerate(t.iterrows()):
        c = COL["orange"] if r.arm == "Selected_vol" else COL["blue"]
        ax.plot([r.pt_dlo, r.pt_dhi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.pt_dauc, i, "o", ms=6, color=c, mec="white", mew=1.2)
        ax.text(1.02, i, f"{r.pt_dauc:+.3f} ({r.pt_dlo:+.3f}, {r.pt_dhi:+.3f})", va="center", fontsize=8, color=COL["ink"],
                transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.02, len(t) - 0.4, "ΔAUC (95% CI)", fontsize=8, color=COL["ink2"], transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_yticks(yy, t.label)
    ax.set_xlim(-0.45, 0.2)
    ax.set_xlabel("ΔAUC vs volume-only model (patient level, LOCO), 95% cluster-bootstrap CI")
    ax.text(0, len(t) - 0.35, "±0.05 = pre-specified\nclinically meaningful margin", fontsize=7, color=COL["ink2"], ha="center", va="bottom")
    fig.savefig(os.path.join(RES, "fig_forest_pooled.png"))
    fig.savefig(os.path.join(RES, "fig_forest_pooled.pdf"))
    plt.close(fig)


def fig_calibration(arms):
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, c in zip(axs, CENTERS):
        ax.plot([0, 1], [0, 1], color=COL["ink2"], lw=1, ls="--")
        for a, col in zip(arms, [COL["blue"], COL["orange"]]):
            pt = to_patient(load_preds(a))
            s = pt[pt.center == c].sort_values("p")
            q = pd.qcut(s.p, min(5, s.y.sum()), duplicates="drop")
            g = s.groupby(q, observed=True).agg(p=("p", "mean"), y=("y", "mean"))
            ax.plot(g.p, g.y, "-o", color=col, lw=2, ms=5, mec="white", label=LABEL[a])
        n, e = len(s), int(s.y.sum())
        ax.set_title(f"Held-out center {c}\n{n} patients, {e} MIBC", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted risk")
    axs[0].set_ylabel("Observed MIBC proportion")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    fig.savefig(os.path.join(RES, "fig_calibration.png"))
    fig.savefig(os.path.join(RES, "fig_calibration.pdf"))
    plt.close(fig)


def fig_dca(arms):
    th = np.linspace(0.05, 0.6, 56)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    pt0 = to_patient(load_preds(arms[0]))
    ax.plot(th, net_benefit_all(pt0.y.values, th), color=COL["ink2"], lw=1.2, ls="--")
    ax.text(0.07, net_benefit_all(pt0.y.values, [0.07])[0], "Treat all", fontsize=7, color=COL["ink2"], va="bottom")
    ax.axhline(0, color=COL["ink2"], lw=1)
    ax.text(0.58, 0.005, "Treat none", fontsize=7, color=COL["ink2"], ha="right", va="bottom")
    for a, col in zip(arms, [COL["blue"], COL["orange"], COL["aqua"]]):
        pt = to_patient(load_preds(a))
        nb = net_benefit(pt.y.values, pt.p.values, th)
        ax.plot(th, nb, color=col, lw=2, label=LABEL[a])
    ax.set_ylim(-0.05, max(0.4, pt0.y.mean() + 0.02))
    ax.set_xlabel("Threshold probability of MIBC")
    ax.set_ylabel("Net benefit (patient level)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(RES, "fig_dca.png"))
    fig.savefig(os.path.join(RES, "fig_dca.pdf"))
    plt.close(fig)


def main():
    df = load_index()
    sel = pd.read_csv(os.path.join(WORK, "inner_selection_tabular.csv"))
    cnn_sel = build_cnn(df)
    arms = ["A_volume", "Aplus_clinvol", "B_radiomics", "B_radiomics_vol", "B_radiomics_res",
            "D_dino", "D_dino_vol", "D_dino_res"]
    if cnn_sel is not None:
        sel = pd.concat([sel, cnn_sel], ignore_index=True)
        arms += ["C_cnn", "C_cnn_vol", "C_cnn_res"]
    else:
        print("WARNING: CNN LOCO not finished; CNN arms excluded")
    choice = build_selected(df, sel)
    arms += ["Selected_vol"]
    print("Selected per fold:\n", choice.to_string(index=False))
    choice.to_csv(os.path.join(RES, "primary_arm_selection.csv"), index=False)

    t = disc_table(arms)
    t.to_csv(os.path.join(RES, "table2_discrimination.csv"), index=False)
    cols = ["label", "pt_auc", "pt_lo", "pt_hi", "pt_dauc", "pt_dlo", "pt_dhi", "les_auc", "les_dauc", "les_dlo", "les_dhi",
            "auc_c1", "auc_c2", "auc_c3", "auc_c4", "re_auc", "re_I2"]
    print(t[cols].round(3).to_string(index=False))

    cal = calib_table(["A_volume", "Selected_vol"] + [a for a in arms if a.endswith("_vol") and a != "Selected_vol"])
    cal.to_csv(os.path.join(RES, "table3_calibration.csv"), index=False)
    print(cal[["arm", "center", "cal_intercept", "cal_slope", "oe", "brier"]].round(3).to_string(index=False))

    nri = nri_idi("Selected_vol")
    pd.DataFrame([dict(metric=m, est=nri[0][i], lo=nri[1][i], hi=nri[2][i]) for i, m in
                  enumerate(["NRI_events", "NRI_nonevents", "NRI_total", "IDI"])]).to_csv(os.path.join(RES, "table4_nri_idi.csv"), index=False)
    print("NRI/IDI (Selected_vol vs volume, thr 0.30):", np.round(nri[0], 3), np.round(nri[1], 3), np.round(nri[2], 3))

    sg = subgroups(df, ["A_volume", "Selected_vol", "B_radiomics", "C_cnn" if cnn_sel is not None else "D_dino", "D_dino"])
    sg.to_csv(os.path.join(RES, "table5_subgroups.csv"), index=False)
    print(sg.round(3).to_string(index=False))

    sens = sensitivity(df, ["A_volume", "Selected_vol", "D_dino_vol", "B_radiomics_vol"] + (["C_cnn_vol"] if cnn_sel is not None else []))
    sens.to_csv(os.path.join(RES, "table7_sensitivity.csv"), index=False)
    print(sens.round(3).to_string(index=False))

    fig_forest(t)
    fig_calibration(["A_volume", "Selected_vol"])
    fig_dca(["A_volume", "Selected_vol"])
    print("figures written to", RES)


if __name__ == "__main__":
    main()
