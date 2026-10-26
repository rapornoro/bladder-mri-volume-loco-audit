"""3D ResNet-18 (MONAI), 2-channel (image, lesion mask) fixed-physical-size crops (16x128x128 @0.75x0.75x3.5mm).
Fixed training schedule (no early stopping) so no held-out data is used for model selection.
Records per-epoch test probabilities (for the optimistic-selection audit) and AdaBN (transductive) predictions.

Usage: python s04_cnn3d.py <design>
  design = loco      : LOCO outer folds, 3 seeds full-train + 5 inner-fold OOF (1 seed) per outer fold
           lesion_cv : random lesion 5-fold CV (pooled centers, lesions of one patient may cross folds)
           patient_cv: patient-grouped stratified 5-fold CV (pooled centers)
Outputs: work/cnn/<design>_*.npz"""
import os
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import resnet18
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import WORK, CENTERS, load_index  # noqa: E402

OUT = os.path.join(WORK, "cnn")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
EPOCHS, BS, LR, WD = 40, 16, 3e-4, 1e-4

df = load_index()
X = np.stack([np.stack([np.load(os.path.join(WORK, "crops", f"{l}.npz"))["img"],
                        np.load(os.path.join(WORK, "crops", f"{l}.npz"))["mask"].astype(np.float32)])
              for l in df.lesion_id]).astype(np.float32)  # N,2,16,128,128
Y = df.y.values.astype(np.float32)
G = df.patient_id.values
Xt = torch.from_numpy(X)


def augment(x, gen):
    # x: B,2,D,H,W on GPU
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    x = torch.rot90(x, k, dims=(-2, -1))
    b = x.shape[0]
    scale = (1 + 0.15 * (torch.rand(b, generator=gen) * 2 - 1)).view(b, 1, 1, 1, 1).to(x.device)
    shift = (0.15 * (torch.rand(b, generator=gen) * 2 - 1)).view(b, 1, 1, 1, 1).to(x.device)
    img = x[:, :1] * scale + shift + 0.05 * torch.randn_like(x[:, :1])
    return torch.cat([img, x[:, 1:]], 1)


def new_model():
    return resnet18(spatial_dims=3, n_input_channels=2, num_classes=1).to(DEV)


@torch.no_grad()
def predict(model, idx):
    model.eval()
    out = []
    for i in range(0, len(idx), 32):
        xb = Xt[idx[i:i + 32]].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(xb).float()).squeeze(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def adabn_predict(model, idx):
    """Transductive: re-estimate BatchNorm statistics on the (unlabelled) test set, then predict."""
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None
    m.train()
    for _ in range(2):
        for i in range(0, len(idx), BS):
            m(Xt[idx[i:i + BS]].to(DEV))
    return predict(m, idx)


def train(tr_idx, te_idx, seed, track=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * int(np.ceil(len(tr_idx) / BS)))
    pos = Y[tr_idx].mean()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=DEV))
    per_epoch = []
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr_idx)
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            if len(b) < 2:
                continue
            xb = augment(Xt[b].to(DEV), gen)
            yb = torch.from_numpy(Y[b]).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(xb).float().squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if track:
            per_epoch.append(predict(model, te_idx))
    final = predict(model, te_idx)
    ada = adabn_predict(model, te_idx)
    return final, ada, np.array(per_epoch)


def inner_split(tr_idx, seed=0):
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(tr_idx[a], tr_idx[b]) for a, b in sgk.split(tr_idx, Y[tr_idx], G[tr_idx])]


def run_loco():
    for c in CENTERS:
        tr = np.where(df.center.values != c)[0]
        te = np.where(df.center.values == c)[0]
        f = os.path.join(OUT, f"loco_c{c}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        finals, adas, epochs = [], [], []
        for s in range(3):
            fi, ad, pe = train(tr, te, seed=100 + s)
            finals.append(fi); adas.append(ad); epochs.append(pe)
            print(f"center {c} seed {s}: AUC {roc_auc_score(Y[te], fi):.3f} adabn {roc_auc_score(Y[te], ad):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        oof = np.zeros(len(df))
        for k, (itr, iva) in enumerate(inner_split(tr)):
            fi, _, _ = train(itr, iva, seed=200 + k, track=False)
            oof[iva] = fi
        print(f"center {c} inner OOF AUC {roc_auc_score(Y[tr], oof[tr]):.3f}", flush=True)
        np.savez(f, te=te, tr=tr, final=np.array(finals), adabn=np.array(adas), epochs=np.array(epochs), inner_oof=oof[tr])


def run_cv(design):
    f = os.path.join(OUT, f"{design}.npz")
    if os.path.exists(f):
        return
    if design == "lesion_cv":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=7).split(X, Y))
    else:
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(X, Y, G))
    oof = np.zeros(len(df))
    for k, (tr, te) in enumerate(splits):
        fi, _, _ = train(tr, te, seed=300 + k, track=False)
        oof[te] = fi
        print(design, k, f"AUC {roc_auc_score(Y[te], fi):.3f}", flush=True)
    np.savez(f, oof=oof, fold=np.concatenate([np.full(len(te), k) for k, (_, te) in enumerate(splits)])[np.argsort(np.concatenate([te for _, te in splits]))])


if __name__ == "__main__":
    design = sys.argv[1]
    if design == "loco":
        run_loco()
    else:
        run_cv(design)
