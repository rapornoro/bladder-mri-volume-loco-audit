"""Shared utilities: data loading, LOCO folds, patient aggregation, statistics."""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss

WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work"))
PREDS = os.path.join(WORK, "preds")
CENTERS = [1, 2, 3, 4]
EPS = 1e-6


def load_index():
    df = pd.read_csv(os.path.join(WORK, "lesions.csv"))
    df["male"] = (df.sex == "Male").astype(float)
    return df


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def save_preds(name, df, p):
    """df: lesion rows of the held-out centers (all 4 folds concatenated); p: predicted prob."""
    os.makedirs(PREDS, exist_ok=True)
    out = df[["lesion_id", "patient_id", "center", "y"]].copy()
    out["p"] = np.asarray(p, dtype=float)
    out.to_csv(os.path.join(PREDS, f"{name}.csv"), index=False)
    return out


def load_preds(name):
    return pd.read_csv(os.path.join(PREDS, f"{name}.csv"))


def to_patient(pr):
    """Patient label = any MIBC lesion; patient score = max lesion probability."""
    return pr.groupby("patient_id").agg(center=("center", "first"), y=("y", "max"), p=("p", "max")).reset_index()


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


# ---------------- bootstrap -----------------
def cluster_boot_idx(units_by_center, rng):
    """Resample patient clusters with replacement, stratified by center. Returns list of unit ids."""
    out = []
    for units in units_by_center:
        out.append(rng.choice(units, size=len(units), replace=True))
    return np.concatenate(out)


def boot_metric(frames, fn, level="patient", B=2000, seed=0):
    """frames: dict name->pred frame (lesion-level, same rows). fn(dict name->frame)->scalar/array.
    Cluster bootstrap by patient, stratified by center. Returns (point, lo, hi, samples)."""
    rng = np.random.default_rng(seed)
    names = list(frames)
    base = frames[names[0]]
    if level == "patient":
        fr = {k: to_patient(v).set_index("patient_id") for k, v in frames.items()}
    else:
        fr = {k: v.set_index("patient_id") for k, v in frames.items()}
    pats = base.groupby("center").patient_id.unique()
    units = [np.asarray(pats[c]) for c in sorted(pats.index)]
    point = fn({k: v.reset_index() for k, v in fr.items()})
    samples = []
    for _ in range(B):
        ids = cluster_boot_idx(units, rng)
        samples.append(fn({k: v.loc[ids].reset_index() for k, v in fr.items()}))
    samples = np.asarray(samples, dtype=float)
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return point, lo, hi, samples


# ---------------- calibration -----------------
def calibration(y, p):
    """Calibration-in-the-large (intercept with slope fixed to 1, via offset), slope, O/E, Brier."""
    y, lp = np.asarray(y, float), logit(np.asarray(p, float))
    try:
        cil = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
        a, a_se = cil.params[0], cil.bse[0]
        b, b_se = slope.params[1], slope.bse[1]
    except Exception:
        a = a_se = b = b_se = np.nan
    return dict(cal_intercept=a, cal_intercept_se=a_se, cal_slope=b, cal_slope_se=b_se,
                oe=y.sum() / np.sum(p), brier=brier_score_loss(y, p))


def dl_meta(est, se):
    """DerSimonian-Laird random-effects pooling. Returns pooled, lo, hi, tau2, I2."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    w = 1 / se ** 2
    fe = np.sum(w * est) / np.sum(w)
    q = np.sum(w * (est - fe) ** 2)
    df = len(est) - 1
    tau2 = max(0.0, (q - df) / (np.sum(w) - np.sum(w ** 2) / np.sum(w))) if df > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    pooled = np.sum(wr * est) / np.sum(wr)
    pse = np.sqrt(1 / np.sum(wr))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return pooled, pooled - 1.96 * pse, pooled + 1.96 * pse, tau2, i2


def auc_se_hanley(y, p):
    a = auc(y, p)
    n1, n0 = int(np.sum(y)), int(len(y) - np.sum(y))
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


# ---------------- decision curve -----------------
def net_benefit(y, p, thresholds):
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        pos = p >= t
        tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.asarray(out)


def net_benefit_all(y, thresholds):
    prev = np.mean(y)
    return np.asarray([prev - (1 - prev) * t / (1 - t) for t in thresholds])
