"""Post hoc analysis: frozen MedSAM (ViT-B) image-encoder features, pooled inside the lesion mask.
Each lesion slice of the 128x128 crop -> 1024x1024 RGB, SAM normalisation -> 256x64x64 embedding;
features = mean embedding inside the (downsampled) lesion mask and inside a 3-pixel peritumoural ring,
then mean- and max-pooled across lesion slices (4 x 256 = 1024 features). Output: work/medsam.csv"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from transformers import SamModel

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, load_index  # noqa: E402

dev = "cuda"
model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(dev).eval()
mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(1, 3, 1, 1)
std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(1, 3, 1, 1)
df = load_index()
rows = []
with torch.no_grad():
    for lid in df.lesion_id:
        d = np.load(os.path.join(WORK, "crops", f"{lid}.npz"))
        img, m = d["img"], d["mask"]
        zs = np.where(m.sum((1, 2)) > 0)[0]
        x = torch.from_numpy((np.clip(img[zs], -3, 3) + 3) / 6 * 255).float().to(dev).unsqueeze(1)
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        emb = []
        for i in range(len(x)):
            with torch.autocast("cuda", dtype=torch.float16):
                emb.append(model.get_image_embeddings((x[i:i + 1] - mean) / std).float())
        emb = torch.cat(emb)  # n,256,64,64
        mk = torch.from_numpy(m[zs].astype(np.float32)).unsqueeze(1).to(dev)
        mk = (F.interpolate(mk, size=(64, 64), mode="area") > 0.25).float()
        ring = torch.from_numpy(np.stack([binary_dilation(a[0], iterations=3) for a in mk.cpu().numpy() > 0]).astype(np.float32)).unsqueeze(1).to(dev)
        ring = ring * (1 - mk)
        mk = torch.where(mk.sum((2, 3), keepdim=True) > 0, mk, torch.ones_like(mk) * 0)  # tiny lesions handled below
        feats = []
        for w in (mk, ring):
            denom = w.sum((2, 3))
            # fall back to the centre pixel if a slice's mask vanishes after downsampling
            w = torch.where(denom[..., None, None] > 0, w, F.pad(torch.ones(len(w), 1, 1, 1, device=dev), (31, 32, 31, 32)))
            v = (emb * w).sum((2, 3)) / w.sum((2, 3))
            feats.append(v)
        f = torch.cat(feats, 1).cpu().numpy()  # n_slices x 512
        rows.append(np.concatenate([f.mean(0), f.max(0)]))
        if len(rows) % 25 == 0:
            print(len(rows), "lesions", flush=True)
cols = [f"medsam_{s}_{r}_{i}" for s in ("mean", "max") for r in ("tumor", "ring") for i in range(256)]
out = pd.DataFrame(np.vstack(rows), columns=cols)
out.insert(0, "lesion_id", df.lesion_id.values)
out.to_csv(os.path.join(WORK, "medsam.csv"), index=False)
print(out.shape)
