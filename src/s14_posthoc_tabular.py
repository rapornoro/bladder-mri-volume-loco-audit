"""Post hoc analyses on tabular arms (LOCO, same tuning protocol as s05).
- maximum axial diameter baseline (A_diameter; log max in-plane Feret diameter over slices, mm).
- nonlinear (natural cubic spline) volume residualisation (*_res_spl); texture-only radiomics (shape class removed).
- MedSAM mask-pooled features (E_medsam*), when work/medsam.csv exists.
- selection/tuning uncertainty: B_radiomics_vol, D_dino_vol re-run with 5 inner-CV seeds -> preds/<arm>__seed<k>.csv
Usage: python s14_posthoc_tabular.py [diameter] [arms] [medsam] [seeds]"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.measure import regionprops, label as cc_label
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import s05_tabular as t5  # noqa: E402
from common import WORK, CENTERS, load_index, save_preds, logit  # noqa: E402

df = load_index()
y, g, lv = df.y.values, df.patient_id.values, df.log_vol.values


# ---------------- diameter ----------------
def max_axial_diameter(mask_path, sx, sy):
    m = np.asarray(nib.load(mask_path).dataobj) > 0
    best = 0.0
    for z in range(m.shape[2]):
        sl = m[:, :, z]
        if sl.sum() == 0:
            continue
        for r in regionprops(cc_label(sl)):
            # anisotropic pixels: scale coordinates before computing the Feret diameter
            coords = r.coords * np.array([sx, sy])
            if len(coords) == 1:
                d = max(sx, sy)
            else:
                from scipy.spatial.distance import pdist
                from scipy.spatial import ConvexHull
                pts = coords
                try:
                    pts = coords[ConvexHull(coords).vertices]
                except Exception:
                    pass
                d = pdist(pts).max() + max(sx, sy)  # + one pixel for edge-to-edge
            best = max(best, d)
    return best


def run_diameter():
    f = os.path.join(WORK, "diameter.csv")
    if not os.path.exists(f):
        d = [max_axial_diameter(r.mask_path, r.sx, r.sy) for r in df.itertuples()]
        pd.DataFrame(dict(lesion_id=df.lesion_id, max_axial_diam_mm=d)).to_csv(f, index=False)
    dm = pd.read_csv(f).max_axial_diam_mm.values
    print("diameter median (IQR) by MIBC:", [np.percentile(dm[y == k], [50, 25, 75]).round(1) for k in (0, 1)],
          " spearman with volume:", round(pd.Series(dm).corr(pd.Series(df.vol_ml), method="spearman"), 3))
    ld = np.log(dm)[:, None]
    p = np.zeros(len(df))
    for c in CENTERS:
        tr, te = df.center.values != c, df.center.values == c
        p[te] = LogisticRegression(C=1e6).fit(ld[tr], y[tr]).predict_proba(ld[te])[:, 1]
    save_preds("A_diameter", df, p)


# ---------------- spline residualisation ----------------
class ResidualiseSpline(BaseEstimator, TransformerMixin):
    """Remove dependence of each feature on log-volume (last column) via natural-cubic-spline basis (4 knots)."""

    def fit(self, X, yy=None):
        X = np.asarray(X, float)
        self.sp_ = SplineTransformer(n_knots=4, degree=3, extrapolation="linear").fit(X[:, -1:])
        B = np.c_[np.ones(len(X)), self.sp_.transform(X[:, -1:])]
        self.beta_ = np.linalg.lstsq(B, np.nan_to_num(X[:, :-1]), rcond=None)[0]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        B = np.c_[np.ones(len(X)), self.sp_.transform(X[:, -1:])]
        return X[:, :-1] - B @ self.beta_


def make_model(kind, residual):
    pipe, grid = t5.make_model(kind, residual is not None and residual != "spline" and residual)
    if residual == "spline":
        pipe.steps.insert(0, ("res", ResidualiseSpline()))
    return pipe, grid


def fit_tuned(kind, residual, X, yy, groups):
    from sklearn.model_selection import GridSearchCV
    pipe, grid = make_model(kind, residual)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=t5.inner_cv(), n_jobs=8)
    gs.fit(X, yy, groups=groups)
    return gs.best_estimator_


def loco_arm(name, kind, X, mode, sel_rows=None, suffix=""):
    """mode: plain | res_lin | res_spl | vol"""
    p = np.full(len(df), np.nan)
    for c in CENTERS:
        tr, te = np.where(df.center != c)[0], np.where(df.center == c)[0]
        if mode == "plain":
            est = fit_tuned(kind, None, X[tr], y[tr], g[tr])
            p[te] = est.predict_proba(X[te])[:, 1]
        elif mode in ("res_lin", "res_spl"):
            Xr = np.c_[X, lv]
            est = fit_tuned(kind, True if mode == "res_lin" else "spline", Xr[tr], y[tr], g[tr])
            p[te] = est.predict_proba(Xr[te])[:, 1]
        else:  # vol stacking
            est = fit_tuned(kind, None, X[tr], y[tr], g[tr])
            oof = t5.inner_oof(est, X[tr], y[tr], g[tr])
            S = np.c_[logit(oof), lv[tr]]
            stk = LogisticRegression(C=1e6, max_iter=5000).fit(S, y[tr])
            p[te] = stk.predict_proba(np.c_[logit(est.predict_proba(X[te])[:, 1]), lv[te]])[:, 1]
            if sel_rows is not None:
                so = t5.inner_oof(LogisticRegression(C=1e6, max_iter=5000), S, y[tr], g[tr])
                sel_rows.append(dict(center=c, arm=name, inner_auc=roc_auc_score(y[tr], so)))
    save_preds(name + suffix, df, p)
    print(name + suffix, "lesion AUC %.3f" % roc_auc_score(y, p), flush=True)
    return p


def features(kind):
    if kind == "radiomics":
        r = pd.read_csv(os.path.join(WORK, "radiomics.csv"))
        return df[["lesion_id"]].merge(r, how="left").drop(columns="lesion_id").values
    if kind == "texture":
        r = pd.read_csv(os.path.join(WORK, "radiomics.csv"))
        r = r[[c for c in r.columns if "_shape_" not in c]]
        return df[["lesion_id"]].merge(r, how="left").drop(columns="lesion_id").values
    if kind == "dino":
        r = pd.read_csv(os.path.join(WORK, "dinov2.csv"))
    else:
        r = pd.read_csv(os.path.join(WORK, "medsam.csv"))
    return df[["lesion_id"]].merge(r, how="left").drop(columns="lesion_id").values


if __name__ == "__main__":
    todo = sys.argv[1:]
    sel = []
    if "diameter" in todo:
        run_diameter()
    if "arms" in todo:
        loco_arm("B_radiomics_res_spl", "radiomics", features("radiomics"), "res_spl")
        loco_arm("D_dino_res_spl", "dino", features("dino"), "res_spl")
        Xt = features("texture")
        loco_arm("B_texture", "radiomics", Xt, "plain")
        loco_arm("B_texture_vol", "radiomics", Xt, "vol", sel)
        loco_arm("B_texture_res_spl", "radiomics", Xt, "res_spl")
    if "medsam" in todo:
        Xm = features("medsam")
        loco_arm("E_medsam", "dino", Xm, "plain")
        loco_arm("E_medsam_vol", "dino", Xm, "vol", sel)
        loco_arm("E_medsam_res", "dino", Xm, "res_lin")
        loco_arm("E_medsam_res_spl", "dino", Xm, "res_spl")
    if "seeds" in todo:
        for k in range(5):
            t5.SEED = 3000 + k
            seed_sel = []
            for name, kind in [("B_radiomics_vol", "radiomics"), ("D_dino_vol", "dino")]:
                loco_arm(name, kind, features(kind), "vol", seed_sel, suffix=f"__seed{k}")
            pd.DataFrame(seed_sel).assign(seed=k).to_csv(os.path.join(WORK, f"inner_selection_seed{k}.csv"), index=False)
        t5.SEED = 2026
    if sel:
        pd.DataFrame(sel).to_csv(os.path.join(WORK, "inner_selection_posthoc.csv"), mode="a",
                                 header=not os.path.exists(os.path.join(WORK, "inner_selection_posthoc.csv")), index=False)
