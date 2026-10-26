"""Robustness analyses.
(a)  Primary-arm selection repeated with PATIENT-level inner AUC (inner OOF stacked predictions aggregated by max).
(b)  Within-center dAUC robustness: unweighted mean of the 4 center dAUCs, leave-one-center-out range of the
     random-effects estimate, and a center-level bootstrap of the unweighted mean (4 centers resampled).
(c)  Reproducibility manifest: SHA256, rows, columns of results/*.csv; SHA256 of src/*.py, SAP.md, manuscript files.
Outputs: results/posthoc_patient_level_selection.csv, results/posthoc_within_center_robustness.csv, MANIFEST.csv"""
import os
import sys
import glob
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
import s05_tabular as t5  # noqa: E402
from common import WORK, CENTERS, load_index, logit, dl_meta  # noqa: E402
from s07_stats import RES  # noqa: E402

ROOT = os.path.abspath(os.path.join(WORK, ".."))
df = load_index()
y, g, lv = df.y.values, df.patient_id.values, df.log_vol.values


def patient_auc(idx, p):
    s = pd.DataFrame(dict(pid=g[idx], y=y[idx], p=p)).groupby("pid").agg(y=("y", "max"), p=("p", "max"))
    return roc_auc_score(s.y, s.p)


def stacked_inner(est, X, tr):
    oof = t5.inner_oof(est, X[tr], y[tr], g[tr])
    S = np.c_[logit(oof), lv[tr]]
    return t5.inner_oof(LogisticRegression(C=1e6, max_iter=5000), S, y[tr], g[tr])


def selection():
    feats = {"radiomics": df[["lesion_id"]].merge(pd.read_csv(os.path.join(WORK, "radiomics.csv")), how="left").drop(columns="lesion_id").values,
             "dino": df[["lesion_id"]].merge(pd.read_csv(os.path.join(WORK, "dinov2.csv")), how="left").drop(columns="lesion_id").values}
    rows = []
    for c in CENTERS:
        tr = np.where(df.center != c)[0]
        for kind, name in [("radiomics", "B_radiomics_vol"), ("dino", "D_dino_vol")]:
            est, _ = t5.fit_tuned(kind, False, feats[kind][tr], y[tr], g[tr])
            so = stacked_inner(est, feats[kind], tr)
            rows.append(dict(center=c, arm=name, inner_lesion_auc=roc_auc_score(y[tr], so), inner_patient_auc=patient_auc(tr, so)))
        d = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
        S = np.c_[logit(d["inner_oof"]), lv[tr]]
        so = np.zeros(len(tr))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(S, y[tr], g[tr]):
            so[b] = LogisticRegression(C=1e6, max_iter=5000).fit(S[a], y[tr][a]).predict_proba(S[b])[:, 1]
        rows.append(dict(center=c, arm="C_cnn_vol", inner_lesion_auc=roc_auc_score(y[tr], so), inner_patient_auc=patient_auc(tr, so)))
    t = pd.DataFrame(rows)
    t["chosen_lesion_level"] = t.groupby("center").inner_lesion_auc.transform("max") == t.inner_lesion_auc
    t["chosen_patient_level"] = t.groupby("center").inner_patient_auc.transform("max") == t.inner_patient_auc
    t.to_csv(os.path.join(RES, "posthoc_patient_level_selection.csv"), index=False)
    print(t.round(4).to_string(index=False))
    return t


def within_center_robustness(B=5000, seed=0):
    wc = pd.read_csv(os.path.join(RES, "posthoc_within_center_dauc.csv"))
    rng = np.random.default_rng(seed)
    rows = []
    # per-center SEs are not stored; recover them from the DL output by re-running the bootstrap would be costly,
    # so the leave-one-center-out range uses equal weights as well as the stored RE estimate.
    for _, r in wc.iterrows():
        d = np.array([r[f"d_c{c}"] for c in CENTERS])
        boots = [rng.choice(d, 4, replace=True).mean() for _ in range(B)]
        loo = [np.delete(d, k).mean() for k in range(4)]
        rows.append(dict(arm=r.arm, re_dauc=r.re_dauc, re_lo=r.re_lo, re_hi=r.re_hi, unweighted_mean=d.mean(),
                         center_boot_lo=np.percentile(boots, 2.5), center_boot_hi=np.percentile(boots, 97.5),
                         loo_min=min(loo), loo_max=max(loo), max_center_dauc=d.max(), n_centers_above_0_05=int((d > 0.05).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "posthoc_within_center_robustness.csv"), index=False)
    print(t.round(3).to_string(index=False))


def manifest():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.csv"))):
        d = pd.read_csv(f)
        rows.append(dict(file=os.path.relpath(f, ROOT), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(),
                         rows=len(d), columns=";".join(map(str, d.columns))))
    for pat in ["src/*.py", "SAP.md", "manuscript/submission_eurradiol/*.md", "results/*.png", "results/*.pdf"]:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            rows.append(dict(file=os.path.relpath(f, ROOT), sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(), rows="", columns=""))
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "MANIFEST.csv"), index=False)
    print("manifest entries:", len(rows))


if __name__ == "__main__":
    selection()
    within_center_robustness()
    manifest()
