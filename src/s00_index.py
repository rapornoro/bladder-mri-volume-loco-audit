"""Build lesion-level index with QC. Output: study/work/lesions.csv"""
import os
import numpy as np
import pandas as pd
import nibabel as nib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "study", "work")
os.makedirs(OUT, exist_ok=True)

rows = []
for c in [1, 2, 3, 4]:
    base = os.path.join(ROOT, f"Center{c}", f"Center{c}")
    d = pd.read_excel(os.path.join(base, f"Center{c}_label.xlsx"))
    d.columns = [x.strip() for x in d.columns]
    img_col = "image" if "image" in d.columns else "image_name"
    mask_col = "mask_new" if "mask_new" in d.columns else "mask_name"
    for _, r in d.iterrows():
        ip = os.path.join(base, "T2WI", str(r[img_col]))
        mp = os.path.join(base, "Annotation", str(r[mask_col]))
        im, mk = nib.load(ip), nib.load(mp)
        m = np.asarray(mk.dataobj) > 0
        zooms = np.array(im.header.get_zooms()[:3], dtype=float)
        vox = float(np.prod(zooms))
        idx = np.argwhere(m)
        rows.append(dict(
            center=c,
            patient_id=f"C{c}_{str(r[img_col]).replace('.nii.gz', '')}",
            lesion_id=f"C{c}_{str(r[mask_col]).replace('.nii.gz', '')}",
            img_path=ip, mask_path=mp,
            y=int(r["label"]),
            t_stage=str(r["Pathological T stage"]).strip(),
            grade=str(r["Pathological grade"]).strip(),
            age=float(r["Age (years)"]),
            sex=str(r["Gender"]).strip(),
            sx=zooms[0], sy=zooms[1], sz=zooms[2],
            nx=im.shape[0], ny=im.shape[1], nz=im.shape[2],
            vol_ml=m.sum() * vox / 1000.0,
            n_vox=int(m.sum()),
            n_slices=int(len(np.unique(idx[:, 2]))) if len(idx) else 0,
            shape_match=im.shape[:3] == mk.shape[:3],
            affine_maxdiff=float(np.abs(im.affine - mk.affine).max()),
        ))

df = pd.DataFrame(rows)
df["n_lesions_patient"] = df.groupby("patient_id").lesion_id.transform("count")
df["log_vol"] = np.log(df.vol_ml)
pat = df.groupby("patient_id").agg(center=("center", "first"), y=("y", "max"))
print(f"lesions={len(df)} MIBC={df.y.sum()} patients={len(pat)} MIBC patients={pat.y.sum()}")
print(pat.groupby("center").y.agg(["count", "sum"]))
print("QC shape mismatches:", (~df.shape_match).sum(), " max affine diff:", round(df.affine_maxdiff.max(), 4))
print("empty masks:", (df.n_vox == 0).sum())
print(df.groupby("center")[["sx", "sz"]].median())
df.to_csv(os.path.join(OUT, "lesions.csv"), index=False)
