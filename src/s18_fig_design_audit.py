"""Evaluation-design figure. (a) AUC by evaluation design for volume, 2D and 3D ResNet-18; leakage-type designs highlighted.
(b) LOCO vs size-matched center-mixed splits (mean per-test-set AUC, min-max) for the same three models."""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from s07_stats import RES, COL  # noqa: E402
import numpy as np
from sklearn.metrics import roc_auc_score
from common import WORK, load_index
LES = load_index()[["lesion_id", "y"]]

t = pd.read_csv(os.path.join(RES, "table6_inflation.csv"))
pb = pd.read_csv(os.path.join(RES, "posthoc_partB_sizematched_singleseed.csv"))
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.55, 1]))

ORDER = [("slice_random", "Random slice split", True), ("lesion_cv", "Random lesion 5-fold", False),
         ("patient_cv", "Patient-grouped 5-fold", False), ("loco_testselected", "LOCO, epoch chosen on test", True),
         ("loco_adabn", "LOCO + test-time AdaBN", False), ("loco", "LOCO (reference)", False)]
models = [("Volume (logistic)", COL["ink2"], "s", "Volume"), ("2D ResNet-18", COL["blue"], "o", "2D ResNet-18"),
          ("3D ResNet-18", COL["orange"], "D", "3D ResNet-18")]
n = len(ORDER)
for i, (des, lab, leak) in enumerate(ORDER):
    if leak:
        ax.axhspan(n - 1 - i - 0.45, n - 1 - i + 0.45, color="#fbe3d6", lw=0, zorder=0)
for j, (m, col, mk, lab) in enumerate(models):
    for i, (des, _, _) in enumerate(ORDER):
        r = t[(t.model == m) & (t.design == des)]
        if r.empty:
            continue
        r = r.iloc[0]
        yv = n - 1 - i + (1 - j) * 0.22
        if des == "slice_random":
            ax.plot([r.slice_min, r.slice_max], [yv, yv], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.slice_auc, yv, mk, color=col, ms=6, mec="white", mew=1.1)
            ax.plot(r.lesion_auc, yv - 0.22, "^", ms=6.5, mfc="white", mec=col, mew=1.4, label="2D, random slice split aggregated to lesions")
        else:
            ax.plot([r.lesion_lo, r.lesion_hi], [yv, yv], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.lesion_auc, yv, mk, color=col, ms=6, mec="white", mew=1.1, label=lab if des == "loco" else None)
r2 = t[(t.model == "2D ResNet-18") & (t.design == "slice_random")].iloc[0]
ax.plot(r2.loco_slice_auc, 0 + 0.22, "o", ms=6, mfc="white", mec=COL["blue"], mew=1.4, label="2D LOCO, slice-level AUC (comparator for slice split)")
ax.set_yticks(range(n), [o[1] for o in ORDER][::-1])
ax.set_xlim(0.6, 1.0)
ax.set_xlabel("AUC (lesion level; filled 2D marker at random slice split = slice level)")
ax.set_title("a  Evaluation designs (shaded = test information leaks)", loc="left", fontsize=9)
h, l = ax.get_legend_handles_labels()
u = dict(zip(l, h))
ax.legend(u.values(), u.keys(), frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3)

# (b)
rows = [("volume", "Volume", COL["ink2"], "s"), ("cnn2d", "2D ResNet-18", COL["blue"], "o"), ("cnn3d", "3D ResNet-18", COL["orange"], "D")]
for k, (key, lab, col, mk) in enumerate(rows):
    lo_ = pb[(pb.model == key) & (pb.design == "LOCO mean fold AUC")].iloc[0]
    sm = pb[(pb.model == key) & (pb.design.str.startswith("size-matched"))].iloc[0]
    vs = pd.read_csv(os.path.join(WORK, "sizematched", f"{key}_volstrat.csv")).merge(LES, on="lesion_id")
    fa = [roc_auc_score(g.y, g.p) for _, g in vs.groupby(["rep", "fold"])]
    cm = pd.Series(dict(lesion_auc=np.mean(fa), lo=np.min(fa), hi=np.max(fa)))
    yv = 2 - k
    for rr, off, fill, m_ in [(lo_, 0.22, col, mk), (sm, 0.0, "white", mk), (cm, -0.22, "white", "^")]:
        bx.plot([rr.lo, rr.hi], [yv + off] * 2, color=col, lw=2, solid_capstyle="round")
        bx.plot(rr.lesion_auc, yv + off, m_, ms=6, mfc=fill, mec=col, mew=1.4)
    bx.text(1.0, yv, f"+{sm.inflation_vs_loco:.3f}\n+{cm.lesion_auc - lo_.lesion_auc:.3f}", va="center", ha="left",
            fontsize=7.5, color=COL["ink"], transform=bx.get_yaxis_transform(), clip_on=False, linespacing=1.6)
bx.set_yticks([2, 1, 0], [r[1] for r in rows])
bx.set_xlim(0.55, 1.0)
bx.set_xlabel("Mean per-test-set AUC (min–max)")
bx.set_title("b  LOCO (filled) · size-matched (open) ·\n    case-mix-matched (triangle) center-mixed splits", loc="left", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(RES, "fig_design_audit.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_design_audit.pdf"), bbox_inches="tight")
print("ok")
