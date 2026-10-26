"""Patient flowchart (European Radiology Figure 1): dataset -> included patients/lesions per center -> analysis sets."""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK  # noqa: E402
from s07_stats import RES, COL  # noqa: E402

df = pd.read_csv(os.path.join(WORK, "lesions.csv"))
pat = df.groupby("patient_id").agg(center=("center", "first"), y=("y", "max"), n=("lesion_id", "count"))
fig, ax = plt.subplots(figsize=(8.6, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, t, bold=False, edge=COL["ink2"], size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1", fc="white", ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=size, weight="bold" if bold else "normal",
            color=COL["ink"], linespacing=1.35)


def arr(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=COL["ink2"], lw=1))


box(15, 88, 70, 9, "FedBCa public dataset (Zenodo 10409145)\nPatients with urothelial carcinoma, MRI ≤ 2 weeks before TURBT or cystectomy,\nno prior treatment; four Chinese centers, 2019–2022", size=7.8)
box(15, 72, 70, 11, f"Released cases with T2WI, expert lesion mask(s) and pathology label\n{len(pat)} patients · {len(df)} lesions · no exclusions in this study\n(descriptor text states 228 patients; its Table 1 and the released files contain 221)", size=7.8)
arr(50, 87.6, 50, 83.4)
vend = {1: "Siemens Skyra", 2: "United Imaging uMR780", 3: "GE MR750w", 4: "Siemens Verio"}
for i, c in enumerate([1, 2, 3, 4]):
    s = pat[pat.center == c]
    l = df[df.center == c]
    box(1 + i * 25, 52, 23, 14, f"Center {c}\n{vend[c]}\n{len(s)} patients ({int(s.y.sum())} MIBC)\n{len(l)} lesions ({int(l.y.sum())} MIBC)\nmultifocal patients: {int((s.n > 1).sum())}", size=7.2)
    arr(50, 71.6, 12.5 + i * 25, 66.4)
box(1, 28, 47, 15, f"Patient-level analysis (primary)\n{len(pat)} patients · {int(pat.y.sum())} MIBC ({pat.y.mean() * 100:.1f}%)\npatient score = maximum lesion probability\nsensitivity: unifocal patients only, n = {int((pat.n == 1).sum())} ({int(pat[pat.n == 1].y.sum())} MIBC)", size=7.6)
box(52, 28, 47, 15, f"Lesion-level analysis (secondary, Part B)\n{len(df)} lesions · {int(df.y.sum())} MIBC ({df.y.mean() * 100:.1f}%)\nMIBC = pT2–T4; NMIBC = pTa/pT1\npatient-clustered bootstrap for all CIs", size=7.6)
for x in (12.5, 37.5, 62.5, 87.5):
    ax.plot([x, x], [51.6, 47.5], color=COL["ink2"], lw=1)
ax.plot([12.5, 87.5], [47.5, 47.5], color=COL["ink2"], lw=1)
arr(24.5, 47.5, 24.5, 43.4)
arr(75.5, 47.5, 75.5, 43.4)
box(10, 6, 80, 13, "Leave-one-center-out validation: each center held out once (4 folds);\n"
    "all preprocessing, tuning and model selection within the three training centers.\n"
    "Grade and T stage were never used as predictors.", size=7.6)
arr(24.5, 27.6, 40, 19.4)
arr(75.5, 27.6, 60, 19.4)
fig.savefig(os.path.join(RES, "fig_patient_flowchart.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_patient_flowchart.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_patient_flowchart.tiff"), dpi=300, bbox_inches="tight")
print("ok")
