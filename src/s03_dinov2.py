"""Frozen DINOv2 ViT-B/14 embeddings of lesion-centred crops.
For each lesion: CLS embeddings of every crop slice containing lesion, mean- and max-pooled.
Input slice: 128x128 crop (96 mm FOV) -> 224x224, z-scored intensities clipped [-3,3] mapped to [0,1],
replicated to 3 channels, ImageNet-normalised. Output: work/dinov2.csv (1536 features)"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm

WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "work"))
dev = "cuda"
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

df = pd.read_csv(os.path.join(WORK, "lesions.csv"))
rows = []
with torch.no_grad():
    for lid in df.lesion_id:
        d = np.load(os.path.join(WORK, "crops", f"{lid}.npz"))
        img, m = d["img"], d["mask"]
        zs = np.where(m.sum((1, 2)) > 0)[0]
        x = torch.from_numpy(np.clip(img[zs], -3, 3)).float().to(dev)
        x = ((x + 3) / 6).unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        e = model((x - mean) / std).cpu().numpy()
        rows.append(np.concatenate([e.mean(0), e.max(0)]))
feat = pd.DataFrame(np.vstack(rows), columns=[f"dino_mean_{i}" for i in range(768)] + [f"dino_max_{i}" for i in range(768)])
feat.insert(0, "lesion_id", df.lesion_id.values)
feat.to_csv(os.path.join(WORK, "dinov2.csv"), index=False)
print(feat.shape)
