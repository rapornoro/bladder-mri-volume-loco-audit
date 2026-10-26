"""Part B: AUC under evaluation designs; inflation relative to LOCO (same backbone).
Inputs: work/cnn2d/*.csv, work/cnn/{loco_c*,lesion_cv,patient_cv}.npz, lesions.csv.
Outputs: results/table6_inflation.csv, results/fig_inflation_designs.(png|pdf)"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, CENTERS, load_index, to_patient, auc  # noqa: E402
from s07_stats import RES, COL  # noqa: E402

B = 2000
df = load_index()
Y = df.y.values
rng0 = np.random.default_rng(11)


def frame(p):
    f = df[["lesion_id", "patient_id", "center", "y"]].copy()
    f["p"] = p
    return f


def boot_diff(pa, pb, level="lesion", seed=0):
    """Paired patient-cluster bootstrap (stratified by center) of AUC(pa)-AUC(pb); pa/pb are lesion-level arrays."""
    rng = np.random.default_rng(seed)
    fa, fb = frame(pa), frame(pb)
    if level == "patient":
        fa, fb = to_patient(fa), to_patient(fb)
    fa, fb = fa.set_index("patient_id"), fb.set_index("patient_id")
    pats = df.groupby("center").patient_id.unique()
    out = []
    for _ in range(B):
        ids = np.concatenate([rng.choice(pats[c], len(pats[c])) for c in CENTERS])
        a, b = fa.loc[ids], fb.loc[ids]
        out.append(auc(a.y, a.p) - auc(b.y, b.p))
    return np.nanpercentile(out, [2.5, 97.5])


def boot_auc(p, level="lesion", seed=0):
    rng = np.random.default_rng(seed)
    f = frame(p)
    if level == "patient":
        f = to_patient(f)
    f = f.set_index("patient_id")
    pats = df.groupby("center").patient_id.unique()
    out = []
    for _ in range(B):
        ids = np.concatenate([rng.choice(pats[c], len(pats[c])) for c in CENTERS])
        s = f.loc[ids]
        out.append(auc(s.y, s.p))
    return np.nanpercentile(out, [2.5, 97.5])


# ---------- volume baseline under each design ----------
lv = df.log_vol.values[:, None]


def vol_cv(splits):
    p = np.zeros(len(df))
    for tr, te in splits:
        p[te] = LogisticRegression(C=1e6).fit(lv[tr], Y[tr]).predict_proba(lv[te])[:, 1]
    return p


vol = {
    "lesion_cv": vol_cv(StratifiedKFold(5, shuffle=True, random_state=7).split(lv, Y)),
    "patient_cv": vol_cv(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(lv, Y, df.patient_id)),
    "loco": vol_cv([(np.where(df.center != c)[0], np.where(df.center == c)[0]) for c in CENTERS]),
}

# ---------- 2D CNN ----------
d2 = {}
for des in ["lesion_cv", "patient_cv", "loco"]:
    s = pd.read_csv(os.path.join(WORK, "cnn2d", f"{des}.csv"))
    d2[des] = df[["lesion_id"]].merge(s.groupby("lesion_id").p.mean().reset_index(), how="left").p.values
sr = pd.read_csv(os.path.join(WORK, "cnn2d", "slice_random.csv"))
sr["y"] = Y[sr.lesion_idx]
slice_aucs = [roc_auc_score(g.y, g.p) for _, g in sr.groupby("rep")]
lo2 = pd.read_csv(os.path.join(WORK, "cnn2d", "loco.csv"))
loco_slice_auc = roc_auc_score(Y[lo2.lesion_idx], lo2.p)
les_aucs = []
for _, g in sr.groupby("rep"):
    l = g.groupby("lesion_idx").agg(p=("p", "mean"), y=("y", "first"))
    les_aucs.append(roc_auc_score(l.y, l.p))

# ---------- 3D CNN ----------
d3 = {}
have3 = all(os.path.exists(os.path.join(WORK, "cnn", f"loco_c{c}.npz")) for c in CENTERS)
if have3:
    p_loco, p_ada, p_best = (np.zeros(len(df)) for _ in range(3))
    best_ep = {}
    for c in CENTERS:
        z = np.load(os.path.join(WORK, "cnn", f"loco_c{c}.npz"))
        te = z["te"]
        p_loco[te] = z["final"].mean(0)
        p_ada[te] = z["adabn"].mean(0)
        ep = z["epochs"].mean(0)  # epochs x n_te (seed-averaged)
        a = [roc_auc_score(Y[te], e) for e in ep]
        best_ep[c] = int(np.argmax(a)) + 1
        p_best[te] = ep[int(np.argmax(a))]
    d3 = {"loco": p_loco, "loco_adabn": p_ada, "loco_testselected": p_best}
    for des in ["lesion_cv", "patient_cv"]:
        f = os.path.join(WORK, "cnn", f"{des}.npz")
        if os.path.exists(f):
            d3[des] = np.load(f)["oof"]
    print("best epoch on test per center:", best_ep)

rows = []


def add(model, design, p, ref, note=""):
    a_l, a_p = auc(Y, p), auc(to_patient(frame(p)).y, to_patient(frame(p)).p)
    lo, hi = boot_auc(p, seed=len(rows))
    r = dict(model=model, design=design, lesion_auc=a_l, lesion_lo=lo, lesion_hi=hi, patient_auc=a_p, note=note)
    if ref is not None and design != "loco":
        r["inflation"] = a_l - auc(Y, ref)
        r["infl_lo"], r["infl_hi"] = boot_diff(p, ref, seed=len(rows) + 100)
    rows.append(r)


for des in ["lesion_cv", "patient_cv", "loco"]:
    add("Volume (logistic)", des, vol[des], vol["loco"])
rows.append(dict(model="2D ResNet-18", design="slice_random", slice_auc=np.mean(slice_aucs), slice_min=np.min(slice_aucs),
                 slice_max=np.max(slice_aucs), lesion_auc=np.mean(les_aucs), lesion_lo=np.min(les_aucs), lesion_hi=np.max(les_aucs),
                 inflation=np.mean(slice_aucs) - loco_slice_auc, loco_slice_auc=loco_slice_auc,
                 inflation_lesion_agg=np.mean(les_aucs) - auc(Y, d2["loco"]),
                 note="inflation = slice-level AUC minus LOCO slice-level AUC; lesion columns = range over 5 repeats"))
for des in ["lesion_cv", "patient_cv", "loco"]:
    add("2D ResNet-18", des, d2[des], d2["loco"])
for des in ["lesion_cv", "patient_cv", "loco", "loco_adabn", "loco_testselected"]:
    if des in d3:
        add("3D ResNet-18", des, d3[des], d3["loco"])
t = pd.DataFrame(rows)
t.to_csv(os.path.join(RES, "table6_inflation.csv"), index=False)
print(t.round(3).to_string(index=False))

# ---------- figure ----------
ORDER = [("slice_random", "Random slice split"), ("lesion_cv", "Random lesion split (5-fold)"),
         ("patient_cv", "Patient-grouped 5-fold"), ("loco_testselected", "LOCO, epoch chosen on test"),
         ("loco_adabn", "LOCO + test-time AdaBN"), ("loco", "Leave-one-center-out (reference)")]
models = [("Volume (logistic)", COL["ink2"], "s"), ("2D ResNet-18", COL["blue"], "o"), ("3D ResNet-18", COL["orange"], "D")]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for j, (m, col, mk) in enumerate(models):
    for i, (des, _) in enumerate(ORDER):
        r = t[(t.model == m) & (t.design == des)]
        if r.empty:
            continue
        r = r.iloc[0]
        yv = len(ORDER) - 1 - i + (j - 1) * 0.22
        x = r.slice_auc if des == "slice_random" else r.lesion_auc
        lo = r.slice_min if des == "slice_random" else r.lesion_lo
        hi = r.slice_max if des == "slice_random" else r.lesion_hi
        ax.plot([lo, hi], [yv, yv], color=col, lw=2, solid_capstyle="round")
        ax.plot(x, yv, mk, color=col, ms=6, mec="white", mew=1.1, label=m if i == len(ORDER) - 1 or (des == "slice_random") else None)
r2 = t[(t.model == "2D ResNet-18") & (t.design == "slice_random")].iloc[0]
ax.plot(r2.loco_slice_auc, 0 + 0 * 0.22, "o", ms=6, mfc="white", mec=COL["blue"], mew=1.5, label="2D ResNet-18, slice-level AUC")
ax.set_yticks(range(len(ORDER)), [o[1] for o in ORDER][::-1])
ax.set_xlabel("AUC for MIBC (lesion level; slice level for random slice split)")
h, l = ax.get_legend_handles_labels()
uniq = dict(zip(l, h))
ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.set_xlim(0.6, 1.0)
fig.savefig(os.path.join(RES, "fig_inflation_designs.png"))
fig.savefig(os.path.join(RES, "fig_inflation_designs.pdf"))
print("done")
