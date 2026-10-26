"""Post hoc analysis: center-adjusted added-value test on held-out (LOCO) predictions, patient level.
Model 0: MIBC ~ center fixed effects + log(max lesion volume); Model 1: + standardised logit(image score).
Reports OR per SD of image logit, LR-test p-value (1 df), and change in within-center (center-stratified) C-index."""
import os, sys
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(__file__))
from common import load_preds, to_patient, logit, load_index, CENTERS, auc
from s07_stats import RES

df = load_index()
vol = df.groupby("patient_id").log_vol.max()
def strat_c(y, lp, cen):
    num = den = 0.0
    for c in CENTERS:
        m = cen == c; yy, pp = y[m], lp[m]
        pos, neg = pp[yy == 1], pp[yy == 0]
        num += ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()); den += len(pos) * len(neg)
    return num / den
rows = []
arms = ["B_radiomics", "B_texture", "C_cnn", "C2_mednet", "D_dino", "E_medsam"]
for a in arms:
    pt = to_patient(load_preds(a)); pt["lv"] = pt.patient_id.map(vol)
    z = logit(pt.p.values); z = (z - z.mean()) / z.std()
    C = pd.get_dummies(pt.center, prefix="c", drop_first=True).astype(float)
    X0 = sm.add_constant(pd.concat([C, pt[["lv"]]], axis=1)); X1 = X0.assign(img=z)
    m0 = sm.Logit(pt.y.values, X0).fit(disp=0); m1 = sm.Logit(pt.y.values, X1).fit(disp=0)
    lr = 2 * (m1.llf - m0.llf)
    ci = np.exp(m1.conf_int().loc["img"].values)
    rows.append(dict(arm=a, OR_per_SD=np.exp(m1.params["img"]), OR_lo=ci[0], OR_hi=ci[1], LR_chi2=lr, p=chi2.sf(lr, 1),
                     cindex_strat_vol=strat_c(pt.y.values, m0.fittedvalues.values, pt.center.values),
                     cindex_strat_vol_img=strat_c(pt.y.values, m1.fittedvalues.values, pt.center.values)))
t = pd.DataFrame(rows); t["delta_c"] = t.cindex_strat_vol_img - t.cindex_strat_vol
t.to_csv(os.path.join(RES, "posthoc_added_value_center_adjusted.csv"), index=False)
print(t.round(3).to_string(index=False))
print("note: in-sample refit on held-out predictions; delta C is optimistic by construction")
