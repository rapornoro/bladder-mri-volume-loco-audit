"""IBSI-compliant PyRadiomics features on preprocessed (resampled, N4, z-scored) images.
Texture computed in-plane (force2D) given 3.5 mm slices. Output: work/radiomics.csv"""
import os
import logging
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)
WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work"))
PARAMS = {
    "setting": {"binWidth": 0.25, "force2D": True, "force2Ddimension": 0, "label": 1,
                "minimumROIDimensions": 1, "correctMask": True, "geometryTolerance": 1e-3},
    "imageType": {"Original": {}, "LoG": {"sigma": [1.0, 2.0, 3.0]}, "Wavelet": {}},
    "featureClass": {"shape": [], "firstorder": [], "glcm": [], "glrlm": [], "glszm": [], "gldm": [], "ngtdm": []},
}


def run(args):
    lid, pid = args
    ext = featureextractor.RadiomicsFeatureExtractor(PARAMS)
    try:
        img = sitk.ReadImage(os.path.join(WORK, "pre", f"{pid}.nii.gz"))
        msk = sitk.ReadImage(os.path.join(WORK, "pre", f"{lid}_mask.nii.gz"))
        f = ext.execute(img, msk)
        out = {k: float(v) for k, v in f.items() if not k.startswith("diagnostics")}
        out["lesion_id"] = lid
        return out
    except Exception as e:
        print("FAIL", lid, e, flush=True)
        return {"lesion_id": lid}


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(WORK, "lesions.csv"))
    with Pool(16) as p:
        res = p.map(run, list(zip(df.lesion_id, df.patient_id)), chunksize=2)
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(WORK, "radiomics.csv"), index=False)
    print(r.shape, "rows with NaN:", r.isna().any(axis=1).sum())
