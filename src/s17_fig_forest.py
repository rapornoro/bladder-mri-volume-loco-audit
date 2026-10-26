"""Forest plot: dAUC vs volume per arm, two estimates per arm:
pooled patient-level dAUC (cluster bootstrap) and within-center paired dAUC (random-effects pooled).
Residualized arms omitted (supplement)."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from s07_stats import RES, COL  # noqa: E402

t2 = pd.concat([pd.read_csv(os.path.join(RES, "table2_discrimination.csv")),
                pd.read_csv(os.path.join(RES, "posthoc_discrimination_new_arms.csv"))], ignore_index=True).drop_duplicates("arm")
wc = pd.read_csv(os.path.join(RES, "posthoc_within_center_estimands.csv")).rename(columns={"strat_dauc": "re_dauc", "strat_lo": "re_lo", "strat_hi": "re_hi"})
ORDER = [("Selected_vol", "Pre-selected image + volume (primary)"),
         ("A_diameter", "Maximum axial diameter†"), ("Aplus_clinvol", "Volume + age, sex, lesion count"),
         ("B_radiomics", "Radiomics"), ("B_radiomics_vol", "Radiomics + volume"),
         ("B_texture", "Texture-only radiomics†"), ("B_texture_vol", "Texture-only radiomics + volume†"),
         ("D_dino", "DINOv2"), ("D_dino_vol", "DINOv2 + volume"),
         ("E_medsam", "MedSAM†"), ("E_medsam_vol", "MedSAM + volume†"),
         ("C_cnn", "3D CNN (scratch)"), ("C_cnn_vol", "3D CNN + volume"),
         ("C2_mednet", "3D CNN (MedicalNet)†"), ("C2_mednet_vol", "3D CNN (MedicalNet) + volume†")]
ORDER = [o for o in ORDER if o[0] in set(t2.arm) and o[0] in set(wc.arm)]
fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(ORDER) + 1.2))
ax.axvspan(-0.05, 0.05, color=COL["grid"], alpha=0.6, lw=0)
ax.axvline(0, color=COL["ink2"], lw=1)
n = len(ORDER)
for i, (arm, lab) in enumerate(ORDER):
    yv = n - 1 - i
    a = t2[t2.arm == arm].iloc[0]
    w = wc[wc.arm == arm].iloc[0]
    ax.plot([a.pt_dlo, a.pt_dhi], [yv + 0.14] * 2, color=COL["blue"], lw=2, solid_capstyle="round")
    ax.plot(a.pt_dauc, yv + 0.14, "o", ms=5.5, color=COL["blue"], mec="white", mew=1, label="Pooled patient-level ΔAUC" if i == 0 else None)
    ax.plot([w.re_lo, w.re_hi], [yv - 0.14] * 2, color=COL["orange"], lw=2, solid_capstyle="round")
    ax.plot(w.re_dauc, yv - 0.14, "D", ms=5, color=COL["orange"], mec="white", mew=1, label="Within-center ΔAUC (center-stratified)" if i == 0 else None)
    ax.text(1.02, yv, f"{round(a.pt_dauc, 3) + 0.0:+.3f} | {round(w.re_dauc, 3) + 0.0:+.3f}", va="center", fontsize=7.5, color=COL["ink"],
            transform=ax.get_yaxis_transform(), clip_on=False)
ax.text(1.02, n - 0.35, "pooled | within", fontsize=7.5, color=COL["ink2"], transform=ax.get_yaxis_transform(), va="bottom")
ax.set_yticks(range(n), [o[1] for o in ORDER][::-1])
lbls = ax.get_yticklabels()
lbls[-1].set_fontweight("bold")
ax.set_xlim(-0.30, 0.15)
ax.set_xlabel("ΔAUC vs volume-only model (LOCO), 95% CI; shaded band = ±0.05 pre-specified margin; † post hoc")
ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
fig.savefig(os.path.join(RES, "fig_forest_dauc.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_forest_dauc.pdf"), bbox_inches="tight")
print("ok", n)
