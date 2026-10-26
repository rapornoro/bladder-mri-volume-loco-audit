"""Figure 1: study design schematic (cohort -> preprocessing -> Part A arms under LOCO -> Part B designs)."""
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
pat = df.groupby("patient_id").agg(center=("center", "first"), y=("y", "max"))
cen = pat.groupby("center").y.agg(["count", "sum"])
les = df.groupby("center").size()
vendor = {1: "Siemens Skyra", 2: "United Imaging uMR780", 3: "GE MR750w", 4: "Siemens Verio"}

fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(-1.5, 62); ax.axis("off")


def box(x, y, w, h, text, edge=COL["ink2"], fill="#ffffff", size=8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=COL["ink"], weight=weight, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=COL["ink2"], lw=1.1))


# cohort row
for i, c in enumerate([1, 2, 3, 4]):
    box(2 + i * 24.5, 50, 21, 10,
        f"Center {c}\n{vendor[c]} (3.0 T)\n{cen.loc[c, 'count']} patients · {les[c]} lesions\nMIBC patients: {cen.loc[c, 'sum']}", size=7.5)
box(25, 38.5, 50, 7,
    f"FedBCa public T2WI + expert lesion masks\n{len(pat)} patients · {len(df)} lesions · {int(pat.y.sum())} MIBC patients (pT2–T4)", size=8.5, weight="bold")
for i in range(4):
    arrow(12.5 + i * 24.5, 49.6, 50, 46)
box(14, 28.5, 72, 6.5, "Automated preprocessing: N4 bias correction · resampling 0.75×0.75×3.5 mm · z-score\n"
    "lesion-centered 96×96×56 mm crops · mask volume (log mL)", size=7.5)
arrow(50, 38.2, 50, 35.3)

# Part A
box(1, 0, 52, 24.5, "", edge=COL["blue"])
ax.text(3, 21.6, "Part A · added value over volume (LOCO, 4 folds)", fontsize=8.5, weight="bold", color=COL["blue"], va="top")
ax.text(3, 18.4,
        "Size: log-volume (reference), max axial diameter†\n"
        "Image: radiomics, texture-only radiomics†,\n"
        "   3D CNN (scratch; MedicalNet-pretrained†),\n"
        "   frozen DINOv2, frozen MedSAM†; each alone and + volume\n"
        "Primary arm pre-selected on inner CV of training centers\n"
        "Primary: pooled patient ΔAUC vs volume, ±0.05 margin\n"
        "   sensitivity: within-center ΔAUC (center-stratified)†\n"
        "Secondary: calibration, DCA, center-adjusted added value†", fontsize=7.5, va="top", color=COL["ink"], linespacing=1.4)
# Part B
box(56, 0, 43, 24.5, "", edge=COL["orange"])
ax.text(58, 21.6, "Part B · evaluation-design sensitivity", fontsize=8.5, weight="bold", color=COL["orange"], va="top")
ax.text(58, 18.4,
        "2D / 3D ResNet-18 and volume under:\n"
        " 1-2 random slice split (slice / lesion level)\n"
        " 3 random lesion 5-fold · 4 patient-grouped 5-fold\n"
        " 5 LOCO (reference) · 6 LOCO + test-time AdaBN\n"
        " 7 LOCO, epoch chosen on test center\n"
        " 8 size-matched · 9 case-mix-matched center-mixed†\n"
        " 10 single-seed comparison†\n"
        "Endpoint: AUC(design) − AUC(LOCO)\n"
        "† post hoc (not in the SAP)", fontsize=7.5, va="top", color=COL["ink"], linespacing=1.4)
arrow(40, 28.2, 27, 24.6)
arrow(60, 28.2, 77.5, 24.6)
fig.savefig(os.path.join(RES, "fig_study_design.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_study_design.tiff"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_study_design.pdf"), bbox_inches="tight")
print("ok")
