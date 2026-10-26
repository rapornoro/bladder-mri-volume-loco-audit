"""Post hoc analysis: separate center mixing from training-set size in Part B.
'size_matched_pooled': for each LOCO fold (held-out center c), draw a patient-level random test set with the same
number of patients as center c from ALL centers (stratified by MIBC), train on the rest -> same train/test sizes as
LOCO but centers mixed. 3 repetitions. Single model (seed fixed) for 3D CNN (scratch), 2D CNN and volume.
Output: work/sizematched/{cnn3d,cnn2d,volume}.csv (rep, fold, lesion_id, p)"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, CENTERS, load_index  # noqa: E402

OUT = os.path.join(WORK, "sizematched")
os.makedirs(OUT, exist_ok=True)
df = load_index()
pat = df.groupby("patient_id").agg(center=("center", "first"), y=("y", "max")).reset_index()


STRAT = os.environ.get("SM_STRAT", "label")
# "label": random patients from all centers, test size = held-out center size, stratified by MIBC (original)
# "label_vol": case-mix matched -- for each held-out center c, draw from ALL centers exactly as many patients per
#   (MIBC status x patient max-volume tertile) stratum as center c has, so the mixed test set reproduces center c's case mix
SUFFIX = "" if STRAT == "label" else "_volstrat"
pat["vt"] = pd.qcut(df.groupby("patient_id").vol_ml.max().reindex(pat.patient_id).values, 3, labels=False)


def splits():
    for rep in range(3):
        rng = np.random.default_rng(1000 + rep)
        for c in CENTERS:
            n_c = int((pat.center == c).sum())
            frac = n_c / len(pat)
            te_p = []
            if STRAT == "label":
                for yv in (0, 1):
                    ids = pat.patient_id[pat.y == yv].values
                    te_p += list(rng.choice(ids, int(round(frac * len(ids))), replace=False))
            else:
                for yv in (0, 1):
                    for v in (0, 1, 2):
                        m = (pat.y == yv) & (pat.vt == v)
                        k = int(((pat.center == c) & m).sum())
                        ids = pat.patient_id[m].values
                        te_p += list(rng.choice(ids, min(k, len(ids)), replace=False))
            te = np.where(df.patient_id.isin(te_p))[0]
            tr = np.where(~df.patient_id.isin(te_p))[0]
            yield rep, c, tr, te


def run(which):
    f = os.path.join(OUT, f"{which}{SUFFIX}.csv")
    if os.path.exists(f):
        return
    rows = []
    if which == "cnn3d":
        import s04_cnn3d as m3
    if which == "cnn2d":
        import s06_cnn2d_designs as m2
    for rep, c, tr, te in splits():
        if which == "volume":
            lv = df.log_vol.values[:, None]
            p = LogisticRegression(C=1e6).fit(lv[tr], df.y.values[tr]).predict_proba(lv[te])[:, 1]
        elif which == "cnn3d":
            p, _, _ = m3.train(tr, te, seed=700 + rep * 10 + c, track=False)
        else:
            s_tr = np.where(np.isin(m2.S_les, tr))[0]
            s_te = np.where(np.isin(m2.S_les, te))[0]
            ps = m2.train_predict(s_tr, s_te, seed=800 + rep * 10 + c)
            p = pd.DataFrame(dict(l=m2.S_les[s_te], p=ps)).groupby("l").p.mean().reindex(te).values
        rows.append(pd.DataFrame(dict(rep=rep, fold=c, lesion_id=df.lesion_id.values[te], p=p)))
        print(which, rep, c, len(tr), len(te), flush=True)
    pd.concat(rows).to_csv(f, index=False)


if __name__ == "__main__":
    for w in sys.argv[1:]:
        run(w)
