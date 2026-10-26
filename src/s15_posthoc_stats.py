"""Post hoc statistics.
- within-center paired patient-level dAUC vs volume for every arm, random-effects pooled (DerSimonian-Laird).
- lesion-level AUC within pooled volume tertiles; subgroup dAUC vs volume (<3 mL, pT1 vs pT2).
- MedicalNet 3D CNN arms built from work/cnn_mednet (C2_mednet, _vol, _res), like s07.build_cnn.
- seed/selection uncertainty: primary 'Selected_vol' rebuilt for 5 inner-CV seeds; per-seed 3D CNN(+vol) dAUC.
- Part B: single-seed LOCO vs single-model CV designs (3D CNN); size-matched pooled splits vs LOCO (mean fold AUC).
Outputs: results/posthoc_*.csv"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from common import (WORK, PREDS, CENTERS, load_index, save_preds, load_preds, to_patient, auc, logit,  # noqa: E402
                    boot_metric, dl_meta)
from s07_stats import RES  # noqa: E402

B = 2000
df = load_index()
Y, LV = df.y.values, df.log_vol.values


def build_cnn_from(folder, prefix):
    p, pv, pr = (np.full(len(df), np.nan) for _ in range(3))
    sel = []
    for c in CENTERS:
        f = os.path.join(WORK, folder, f"loco_c{c}.npz")
        if not os.path.exists(f):
            return None
        d = np.load(f)
        te, tr, oof = d["te"], d["tr"], d["inner_oof"]
        pt = d["final"].mean(0)
        p[te] = pt
        S = np.c_[logit(oof), LV[tr]]
        pv[te] = LogisticRegression(C=1e6, max_iter=5000).fit(S, Y[tr]).predict_proba(np.c_[logit(pt), LV[te]])[:, 1]
        beta = np.polyfit(LV[tr], logit(oof), 1)
        pr[te] = 1 / (1 + np.exp(-(logit(pt) - np.polyval(beta, LV[te]))))
        so = np.zeros(len(tr))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(S, Y[tr], df.patient_id.values[tr]):
            so[b] = LogisticRegression(C=1e6, max_iter=5000).fit(S[a], Y[tr][a]).predict_proba(S[b])[:, 1]
        sel.append(dict(center=c, arm=f"{prefix}_vol", inner_auc=roc_auc_score(Y[tr], so)))
    save_preds(prefix, df, p)
    save_preds(f"{prefix}_vol", df, pv)
    save_preds(f"{prefix}_res", df, pr)
    return pd.DataFrame(sel)


# ---------------- within-center dAUC ----------------
def within_center_dauc(arm, ref="A_volume", seed=0):
    rng = np.random.default_rng(seed)
    a, r = to_patient(load_preds(arm)), to_patient(load_preds(ref))
    m = a.merge(r, on=["patient_id", "center", "y"], suffixes=("", "_r"))
    est, se, per = [], [], {}
    for c in CENTERS:
        s = m[m.center == c].reset_index(drop=True)
        d0 = auc(s.y, s.p) - auc(s.y, s.p_r)
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(s), len(s))
            bs.append(auc(s.y[i], s.p[i]) - auc(s.y[i], s.p_r[i]))
        est.append(d0)
        se.append(np.nanstd(bs))
        per[f"d_c{c}"] = d0
    pm, lo, hi, tau2, i2 = dl_meta(est, se)
    return dict(arm=arm, re_dauc=pm, re_lo=lo, re_hi=hi, tau2=tau2, I2=i2, **per)


def decision(lo, hi, margin=0.05):
    if lo > margin:
        return "meaningful increment"
    if hi < margin:
        return "no meaningful increment"
    return "inconclusive"


# ---------------- strata ----------------
def strata_tables(arms):
    les = df[["lesion_id", "vol_ml", "t_stage"]]
    q = np.quantile(df.vol_ml, [1 / 3, 2 / 3])
    rows, sg = [], []
    ref = load_preds("A_volume").merge(les, on="lesion_id")
    for a in arms:
        pr = load_preds(a).merge(les, on="lesion_id")
        pr["tert"] = np.digitize(pr.vol_ml, q)
        for k, s in pr.groupby("tert"):
            pa, lo, hi, _ = boot_metric({"a": s}, lambda f: auc(f["a"].y, f["a"].p), "lesion", B, seed=11)
            rows.append(dict(arm=a, volume_tertile=int(k) + 1, vol_range=f"{s.vol_ml.min():.2f}-{s.vol_ml.max():.2f} mL",
                             n=len(s), events=int(s.y.sum()), auc=pa, lo=lo, hi=hi))
        for name, mk in [("lesions < 3 mL", lambda d: d.vol_ml < 3), ("pT1 vs pT2", lambda d: d.t_stage.isin(["T1", "T2"]))]:
            s, r = pr[mk(pr)], ref[mk(ref)]
            d, dlo, dhi, _ = boot_metric({"a": s, "ref": r}, lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p),
                                         "lesion", B, seed=12)
            sg.append(dict(arm=a, subgroup=name, n=len(s), events=int(s.y.sum()), dauc=d, lo=dlo, hi=dhi))
    return pd.DataFrame(rows), pd.DataFrame(sg)


# ---------------- seeds ----------------
def seed_uncertainty(cnn_sel):
    rows = []
    ref = load_preds("A_volume")
    for k in range(5):
        f = os.path.join(WORK, f"inner_selection_seed{k}.csv")
        if not os.path.exists(f):
            continue
        sel = pd.read_csv(f)
        sel["arm_base"] = sel.arm
        cand = pd.concat([sel[["center", "arm", "inner_auc"]], cnn_sel], ignore_index=True)
        choice = cand.loc[cand.groupby("center").inner_auc.idxmax()]
        p = np.full(len(df), np.nan)
        for _, r in choice.iterrows():
            name = r.arm if r.arm.startswith("C_") else f"{r.arm}__seed{k}"
            pr = load_preds(name)
            m = pr.center.values == r.center
            p[m] = pr.p.values[m]
        sp = save_preds(f"Selected_vol__seed{k}", df, p)
        for arm in ["Selected_vol", "B_radiomics_vol", "D_dino_vol"]:
            nm = f"{arm}__seed{k}"
            d, lo, hi, _ = boot_metric({"a": load_preds(nm), "ref": ref},
                                       lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p), "patient", 500, seed=k)
            rows.append(dict(source=f"inner-CV seed {k}", arm=arm, choice=";".join(choice.arm.str.replace("_vol", "")), dauc=d, lo=lo, hi=hi))
    # per-seed 3D CNN (+vol)
    for s_ in range(3):
        p, pv = np.full(len(df), np.nan), np.full(len(df), np.nan)
        for c in CENTERS:
            d = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
            te, tr, oof = d["te"], d["tr"], d["inner_oof"]
            p[te] = d["final"][s_]
            S = np.c_[logit(oof), LV[tr]]
            pv[te] = LogisticRegression(C=1e6, max_iter=5000).fit(S, Y[tr]).predict_proba(np.c_[logit(p[te]), LV[te]])[:, 1]
        for arm, pp in [("C_cnn", p), ("C_cnn_vol", pv)]:
            sp = save_preds(f"{arm}__cnnseed{s_}", df, pp)
            dd, lo, hi, _ = boot_metric({"a": sp, "ref": ref},
                                        lambda f: auc(f["a"].y, f["a"].p) - auc(f["ref"].y, f["ref"].p), "patient", 500, seed=s_)
            rows.append(dict(source=f"CNN training seed {s_}", arm=arm, dauc=dd, lo=lo, hi=hi))
    return pd.DataFrame(rows)


# ---------------- Part B ----------------
def part_b_posthoc():
    rows = []
    # (a) single-seed LOCO (3D) vs single-model CV designs
    loco_seed = []
    for s_ in range(3):
        p = np.full(len(df), np.nan)
        for c in CENTERS:
            d = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
            p[d["te"]] = d["final"][s_]
        loco_seed.append(auc(Y, p))
    rows.append(dict(model="3D ResNet-18", design="LOCO, single seed (mean of 3)", lesion_auc=np.mean(loco_seed),
                     lo=np.min(loco_seed), hi=np.max(loco_seed), note="range over seeds"))
    for des in ["patient_cv", "lesion_cv"]:
        rows.append(dict(model="3D ResNet-18", design=f"{des}, single model", lesion_auc=auc(Y, np.load(os.path.join(WORK, "cnn", f"{des}.npz"))["oof"])))
    # (b) size-matched pooled splits vs LOCO: mean per-fold lesion AUC
    loco = {"volume": load_preds("A_volume"), "cnn3d": None, "cnn2d": None}
    p3 = np.full(len(df), np.nan)
    for c in CENTERS:
        d = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
        p3[d["te"]] = d["final"][0]  # single seed, matching size-matched single models
    loco["cnn3d"] = df[["lesion_id", "center", "y"]].assign(p=p3)
    s2 = pd.read_csv(os.path.join(WORK, "cnn2d", "loco.csv")).groupby("lesion_id").p.mean().reset_index()
    loco["cnn2d"] = df[["lesion_id", "center", "y"]].merge(s2, on="lesion_id")
    for which in ["volume", "cnn2d", "cnn3d"]:
        f = os.path.join(WORK, "sizematched", f"{which}.csv")
        L = loco[which]
        loco_folds = [auc(L[L.center == c].y, L[L.center == c].p) for c in CENTERS]
        r = dict(model=which, design="LOCO mean fold AUC", lesion_auc=np.mean(loco_folds), lo=np.min(loco_folds), hi=np.max(loco_folds))
        rows.append(r)
        if os.path.exists(f):
            sm = pd.read_csv(f).merge(df[["lesion_id", "y"]], on="lesion_id")
            fa = [auc(s.y, s.p) for _, s in sm.groupby(["rep", "fold"])]
            rows.append(dict(model=which, design="size-matched pooled split mean fold AUC (3 reps x 4)", lesion_auc=np.mean(fa),
                             lo=np.min(fa), hi=np.max(fa), inflation_vs_loco=np.mean(fa) - np.mean(loco_folds)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sel_cnn = pd.read_csv(os.path.join(RES, "primary_arm_selection.csv"))  # not used for selection; for reference
    from s07_stats import build_cnn
    cnn_sel = build_cnn(df)
    mn_sel = build_cnn_from("cnn_mednet", "C2_mednet")
    arms = ["A_diameter", "Aplus_clinvol", "B_radiomics", "B_radiomics_vol", "B_radiomics_res", "B_radiomics_res_spl",
            "B_texture", "B_texture_vol", "B_texture_res_spl",
            "C_cnn", "C_cnn_vol", "C_cnn_res", "D_dino", "D_dino_vol", "D_dino_res", "D_dino_res_spl", "Selected_vol"]
    if mn_sel is not None:
        arms += ["C2_mednet", "C2_mednet_vol", "C2_mednet_res"]
    if os.path.exists(os.path.join(PREDS, "E_medsam.csv")):
        arms += ["E_medsam", "E_medsam_vol", "E_medsam_res", "E_medsam_res_spl"]
    arms = [a for a in arms if os.path.exists(os.path.join(PREDS, f"{a}.csv"))]

    wc = pd.DataFrame([within_center_dauc(a, seed=i) for i, a in enumerate(arms)])
    wc["decision_vs_0.05"] = [decision(l, h) for l, h in zip(wc.re_lo, wc.re_hi)]
    wc.to_csv(os.path.join(RES, "posthoc_within_center_dauc.csv"), index=False)
    print(wc.round(3).to_string(index=False))

    # pooled patient-level dAUC + AUC for the new arms (same as table 2 format)
    from s07_stats import disc_table
    new = [a for a in arms if a in ("A_diameter", "B_radiomics_res_spl", "B_texture", "B_texture_vol", "B_texture_res_spl",
                                    "D_dino_res_spl", "C2_mednet", "C2_mednet_vol", "C2_mednet_res",
                                    "E_medsam", "E_medsam_vol", "E_medsam_res", "E_medsam_res_spl")]
    t = disc_table(new)
    t.to_csv(os.path.join(RES, "posthoc_discrimination_new_arms.csv"), index=False)
    print(t[["arm", "pt_auc", "pt_lo", "pt_hi", "pt_dauc", "pt_dlo", "pt_dhi", "les_auc", "re_auc", "re_I2"]].round(3).to_string(index=False))

    st, sg = strata_tables(["A_volume", "A_diameter", "Selected_vol", "B_texture_vol", "C_cnn_vol", "D_dino_vol"] +
                           (["C2_mednet_vol"] if "C2_mednet_vol" in arms else []) + (["E_medsam_vol"] if "E_medsam_vol" in arms else []))
    st.to_csv(os.path.join(RES, "posthoc_volume_tertiles.csv"), index=False)
    sg.to_csv(os.path.join(RES, "posthoc_subgroup_dauc.csv"), index=False)
    print(st.round(3).to_string(index=False))
    print(sg.round(3).to_string(index=False))

    su = seed_uncertainty(cnn_sel)  # candidates restricted to the pre-specified B, C, D arms
    su.to_csv(os.path.join(RES, "posthoc_seed_uncertainty.csv"), index=False)
    print(su.round(3).to_string(index=False))

    pb = part_b_posthoc()
    pb.to_csv(os.path.join(RES, "posthoc_partB_sizematched_singleseed.csv"), index=False)
    print(pb.round(3).to_string(index=False))
