"""LOCO evaluation of tabular arms.
A: log-volume; Aplus: log-volume+age+sex+lesion count;
B: radiomics L1-LR; D: DINOv2 PCA+L2-LR; each with _res (volume-residualised features) and _vol (stacked with log-volume).
All tuning inside training centers via patient-grouped stratified 5-fold CV.
Writes work/preds/<arm>.csv and work/inner_selection.csv"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, CENTERS, load_index, save_preds, logit  # noqa: E402

SEED = 2026


class Clean(BaseEstimator, TransformerMixin):
    """Drop non-finite / near-constant columns, optional greedy |spearman|>thr filter (fit on train)."""

    def __init__(self, corr_thr=None):
        self.corr_thr = corr_thr

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        ok = np.all(np.isfinite(X), 0) & (np.nanstd(X, 0) > 1e-8)
        self.med_ = np.nanmedian(X, 0)
        keep = np.where(ok)[0]
        if self.corr_thr is not None and len(keep) > 1:
            r = pd.DataFrame(X[:, keep]).rank().corr().abs().values
            sel = []
            for j in range(len(keep)):
                if all(r[j, k] <= self.corr_thr for k in sel):
                    sel.append(j)
            keep = keep[sel]
        self.keep_ = keep
        return self

    def transform(self, X):
        X = np.asarray(X, float)[:, self.keep_]
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.med_[self.keep_], np.where(bad)[1])
        return X


class Residualise(BaseEstimator, TransformerMixin):
    """Remove linear dependence of each feature on log-volume (last column), then drop log-volume."""

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        v = np.c_[np.ones(len(X)), X[:, -1]]
        self.beta_ = np.linalg.lstsq(v, X[:, :-1], rcond=None)[0]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return X[:, :-1] - np.c_[np.ones(len(X)), X[:, -1]] @ self.beta_


def inner_cv():
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_model(kind, residual):
    steps = []
    if residual:
        steps.append(("res", Residualise()))
    if kind == "radiomics":
        steps += [("clean", Clean(corr_thr=0.9)), ("sc", StandardScaler()),
                  ("lr", LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))]
        grid = {"lr__C": [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]}
    else:  # dino
        steps += [("clean", Clean()), ("sc", StandardScaler()), ("pca", PCA(random_state=SEED)),
                  ("lr", LogisticRegression(max_iter=5000))]
        grid = {"pca__n_components": [8, 16, 32], "lr__C": [0.01, 0.1, 1.0]}
    return Pipeline(steps), grid


def fit_tuned(kind, residual, X, y, groups):
    pipe, grid = make_model(kind, residual)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv(), n_jobs=8)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def inner_oof(est, X, y, groups):
    """Inner OOF probabilities with fixed (already tuned) hyperparameters."""
    oof = np.zeros(len(y))
    from sklearn.base import clone
    for tr, va in inner_cv().split(X, y, groups):
        m = clone(est).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def lr_plain():
    return LogisticRegression(C=1e6, max_iter=5000)


def main():
    df = load_index()
    rad = pd.read_csv(os.path.join(WORK, "radiomics.csv"))
    dino = pd.read_csv(os.path.join(WORK, "dinov2.csv"))
    feats = {"radiomics": df[["lesion_id"]].merge(rad, how="left").drop(columns="lesion_id").values,
             "dino": df[["lesion_id"]].merge(dino, how="left").drop(columns="lesion_id").values}
    lv = df.log_vol.values
    y = df.y.values
    g = df.patient_id.values

    preds = {k: np.full(len(df), np.nan) for k in
             ["A_volume", "Aplus_clinvol"] + [f"{a}{s}" for a in ["B_radiomics", "D_dino"] for s in ["", "_res", "_vol"]]}
    sel_rows = []
    for c in CENTERS:
        tr, te = np.where(df.center != c)[0], np.where(df.center == c)[0]
        # A
        m = lr_plain().fit(lv[tr, None], y[tr])
        preds["A_volume"][te] = m.predict_proba(lv[te, None])[:, 1]
        # A+
        Xc = np.c_[lv, df.age.values, df.male.values, np.log(df.n_lesions_patient.values)]
        sc = StandardScaler().fit(Xc[tr])
        m = lr_plain().fit(sc.transform(Xc[tr]), y[tr])
        preds["Aplus_clinvol"][te] = m.predict_proba(sc.transform(Xc[te]))[:, 1]
        inner_vol = inner_oof(lr_plain(), lv[tr, None], y[tr], g[tr])
        sel_rows.append(dict(center=c, arm="A_volume", inner_auc=roc_auc_score(y[tr], inner_vol)))
        for kind, name in [("radiomics", "B_radiomics"), ("dino", "D_dino")]:
            X = feats[kind]
            Xr = np.c_[X, lv]
            # plain
            est, bp = fit_tuned(kind, False, X[tr], y[tr], g[tr])
            preds[name][te] = est.predict_proba(X[te])[:, 1]
            # residualised (volume-independent signal)
            est_r, bpr = fit_tuned(kind, True, Xr[tr], y[tr], g[tr])
            preds[f"{name}_res"][te] = est_r.predict_proba(Xr[te])[:, 1]
            # stacked with volume: stacker trained on inner OOF image scores
            oof = inner_oof(est, X[tr], y[tr], g[tr])
            S_tr = np.c_[logit(oof), lv[tr]]
            stk = lr_plain().fit(S_tr, y[tr])
            S_te = np.c_[logit(est.predict_proba(X[te])[:, 1]), lv[te]]
            preds[f"{name}_vol"][te] = stk.predict_proba(S_te)[:, 1]
            stk_oof = inner_oof(lr_plain(), S_tr, y[tr], g[tr])
            sel_rows.append(dict(center=c, arm=f"{name}_vol", inner_auc=roc_auc_score(y[tr], stk_oof),
                                 inner_auc_img=roc_auc_score(y[tr], oof), params=str(bp), params_res=str(bpr)))
            print(c, name, bp, "inner img AUC %.3f  +vol %.3f" % (sel_rows[-1]["inner_auc_img"], sel_rows[-1]["inner_auc"]), flush=True)
    for k, p in preds.items():
        save_preds(k, df, p)
    pd.DataFrame(sel_rows).to_csv(os.path.join(WORK, "inner_selection_tabular.csv"), index=False)


if __name__ == "__main__":
    main()
