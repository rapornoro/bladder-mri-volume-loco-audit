"""Part B: 2D ResNet-18 (ImageNet-pretrained, timm) on lesion-containing slices, under evaluation designs:
  slice_random : random 80/20 slice split, 5 repeats (slices of the same lesion/patient cross train/test)
  lesion_cv    : random lesion 5-fold CV
  patient_cv   : patient-grouped stratified 5-fold CV
  loco         : leave-one-center-out
Slice input: channels (img, img, mask), 128x128 @0.75 mm. Lesion score = mean slice probability.
Output: work/cnn2d/<design>.csv with per-slice predictions (lesion_id, slice, split/rep, p)."""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, CENTERS, load_index  # noqa: E402

OUT = os.path.join(WORK, "cnn2d")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
EPOCHS, BS, LR = 15, 64, 1e-4

df = load_index()
S_img, S_les = [], []
for i, l in enumerate(df.lesion_id):
    d = np.load(os.path.join(WORK, "crops", f"{l}.npz"))
    for z in np.where(d["mask"].sum((1, 2)) > 0)[0]:
        im = np.clip(d["img"][z], -3, 3) / 3
        S_img.append(np.stack([im, im, d["mask"][z].astype(np.float32)]))
        S_les.append(i)
S_img = torch.from_numpy(np.stack(S_img).astype(np.float32))
S_les = np.array(S_les)
S_y = df.y.values[S_les].astype(np.float32)
print("slices:", len(S_les), flush=True)


def train_predict(tr, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = timm.create_model("resnet18", pretrained=True, num_classes=1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    pos = S_y[tr].mean()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=DEV))
    for ep in range(EPOCHS):
        m.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            if len(b) < 2:
                continue
            x = S_img[b].to(DEV)
            if np.random.rand() < 0.5:
                x = x.flip(-1)
            x = torch.rot90(x, np.random.randint(4), dims=(-2, -1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(x).float().squeeze(1), torch.from_numpy(S_y[b]).to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(m(S_img[te[i:i + 128]].to(DEV)).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


def lesion_folds(splitter, groups=None):
    """Map lesion-level folds to slice indices."""
    idx = np.arange(len(df))
    sp = splitter.split(idx, df.y.values, groups) if groups is not None else splitter.split(idx, df.y.values)
    for tr_l, te_l in sp:
        yield np.where(np.isin(S_les, tr_l))[0], np.where(np.isin(S_les, te_l))[0]


def run(design):
    f = os.path.join(OUT, f"{design}.csv")
    if os.path.exists(f):
        return
    rows = []
    if design == "slice_random":
        for rep in range(5):
            rng = np.random.default_rng(rep)
            perm = rng.permutation(len(S_les))
            n_te = int(0.2 * len(perm))
            te, tr = perm[:n_te], perm[n_te:]
            p = train_predict(tr, te, seed=400 + rep)
            rows.append(pd.DataFrame(dict(slice=te, lesion_idx=S_les[te], rep=rep, p=p)))
            print(design, rep, "slice AUC %.3f" % roc_auc_score(S_y[te], p), flush=True)
    else:
        if design == "lesion_cv":
            folds = lesion_folds(StratifiedKFold(5, shuffle=True, random_state=7))
        elif design == "patient_cv":
            folds = lesion_folds(StratifiedGroupKFold(5, shuffle=True, random_state=7), df.patient_id.values)
        else:
            folds = ((np.where(df.center.values[S_les] != c)[0], np.where(df.center.values[S_les] == c)[0]) for c in CENTERS)
        for k, (tr, te) in enumerate(folds):
            p = train_predict(tr, te, seed=500 + k)
            rows.append(pd.DataFrame(dict(slice=te, lesion_idx=S_les[te], rep=k, p=p)))
            print(design, k, "slice AUC %.3f" % roc_auc_score(S_y[te], p), flush=True)
    r = pd.concat(rows)
    r["lesion_id"] = df.lesion_id.values[r.lesion_idx]
    r.to_csv(f, index=False)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
