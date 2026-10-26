"""Resample (0.75x0.75x3.5 mm), N4 bias-correct, z-score normalise; save per-patient image,
per-lesion resampled mask, and fixed-physical-size lesion-centred crops (128x128x16).
Outputs: work/pre/<patient>.nii.gz, work/pre/<lesion>_mask.nii.gz, work/crops/<lesion>.npz"""
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool

HERE = os.path.dirname(__file__)
WORK = os.path.abspath(os.path.join(HERE, "..", "work"))
PRE, CROPS = os.path.join(WORK, "pre"), os.path.join(WORK, "crops")
SPACING = (0.75, 0.75, 3.5)
CROP = (128, 128, 16)


def resample(img, interp, ref=None):
    if ref is not None:
        return sitk.Resample(img, ref, sitk.Transform(), interp, 0.0, img.GetPixelID())
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), SPACING)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), SPACING,
                         img.GetDirection(), 0.0, img.GetPixelID())


def process_patient(args):
    pid, img_path, lesions = args
    out_img = os.path.join(PRE, f"{pid}.nii.gz")
    img = sitk.ReadImage(img_path)
    if "vector" in img.GetPixelIDTypeAsString():  # some C3 files are stored as 1-component vectors
        img = sitk.VectorIndexSelectionCast(img, 0)
    img = sitk.Cast(img, sitk.sitkFloat32)
    if os.path.exists(out_img):
        z = sitk.ReadImage(out_img)
    else:
        # N4 on a shrunken copy, bias field applied at full resolution
        head = sitk.OtsuThreshold(img, 0, 1, 200)
        shrink = [4, 4, 1]
        small, small_mask = sitk.Shrink(img, shrink), sitk.Shrink(head, shrink)
        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetMaximumNumberOfIterations([30, 30, 20])
        n4.Execute(small, small_mask)
        logbias = n4.GetLogBiasFieldAsImage(img)
        corr = img / sitk.Exp(logbias)
        r = resample(corr, sitk.sitkBSpline)
        a = sitk.GetArrayFromImage(r)  # z,y,x
        fg = sitk.GetArrayFromImage(resample(sitk.Cast(head, sitk.sitkFloat32), sitk.sitkNearestNeighbor, r)) > 0.5
        mu, sd = a[fg].mean(), a[fg].std() + 1e-6
        a = np.clip((a - mu) / sd, -5, 5).astype(np.float32)
        z = sitk.GetImageFromArray(a)
        z.CopyInformation(r)
        sitk.WriteImage(z, out_img)
    za = sitk.GetArrayFromImage(z)
    for lid, mask_path in lesions:
        m = sitk.ReadImage(mask_path)
        m.CopyInformation(img)  # QC showed <0.001 mm affine differences
        mr = resample(sitk.Cast(m > 0, sitk.sitkUInt8), sitk.sitkNearestNeighbor, z)
        sitk.WriteImage(mr, os.path.join(PRE, f"{lid}_mask.nii.gz"))
        ma = sitk.GetArrayFromImage(mr).astype(np.uint8)
        if ma.sum() == 0:
            print("EMPTY after resample", lid, flush=True)
            continue
        cz, cy, cx = [int(round(v)) for v in np.argwhere(ma).mean(0)]
        half = (CROP[2] // 2, CROP[1] // 2, CROP[0] // 2)
        pad = [(h, h) for h in half]
        zp, mp = np.pad(za, pad, constant_values=za.min()), np.pad(ma, pad)
        sl = tuple(slice(c, c + 2 * h) for c, h in zip((cz, cy, cx), half))
        np.savez_compressed(os.path.join(CROPS, f"{lid}.npz"), img=zp[sl], mask=mp[sl],
                            mask_vox_total=int(ma.sum()), mask_vox_in_crop=int(mp[sl].sum()))
    return pid


if __name__ == "__main__":
    os.makedirs(PRE, exist_ok=True)
    os.makedirs(CROPS, exist_ok=True)
    df = pd.read_csv(os.path.join(WORK, "lesions.csv"))
    jobs = [(pid, g.img_path.iloc[0], list(zip(g.lesion_id, g.mask_path))) for pid, g in df.groupby("patient_id")]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as p:
        for i, pid in enumerate(p.imap_unordered(process_patient, jobs), 1):
            if i % 25 == 0:
                print(i, "/", len(jobs), flush=True)
    # crop coverage QC
    cov = []
    for lid in df.lesion_id:
        f = os.path.join(CROPS, f"{lid}.npz")
        if os.path.exists(f):
            d = np.load(f)
            cov.append((lid, int(d["mask_vox_in_crop"]) / max(int(d["mask_vox_total"]), 1)))
    cov = pd.DataFrame(cov, columns=["lesion_id", "crop_coverage"])
    cov.to_csv(os.path.join(WORK, "crop_coverage.csv"), index=False)
    print("crops:", len(cov), " coverage<0.95:", (cov.crop_coverage < 0.95).sum(), " min:", round(cov.crop_coverage.min(), 3))
