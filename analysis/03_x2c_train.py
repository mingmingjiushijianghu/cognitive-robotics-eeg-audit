"""Step 3 (design §5.1.1): image -> 30 control values regressor trained on X2C.

Follows X2CNet's mapping network (Li et al. 2025, §3.1-3.2): ResNet18 feature extractor +
MLP head (2 hidden layers, ReLU), AdamW (wd 0.05), cosine schedule with warm-up, lr 1e-3,
Huber loss. Targets are min-max scaled per control using the empirical range of the
100,000 annotations (saved to models/x2c_target_scaling.json).

Split: contiguous blocks of 1,000 frames (animations are stored in order), 10 % of blocks
for validation -> no adjacent frames shared between train and val. The repository's
.pkl split files are never unpickled (security note in 00_数据说明.md).
Training resumes from models/x2c_last.pt if present. Augmentation avoids horizontal flips
(they would swap left/right controls) and adds colour/greyscale jitter so that the model
transfers better to human (AFFEC) and cartoon (Miko) faces.

Outputs
  models/x2c_resnet18.pt (best val MAE), models/x2c_last.pt, models/x2c_target_scaling.json
  results/x2c_training_log.csv, results/x2c_val_mae_per_control.csv
"""
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import config as C


def load_meta():
    m = pd.read_json(C.X2C_DIR / "metadata.jsonl", lines=True)
    m["idx"] = m.file_name.str.replace(".jpg", "", regex=False).astype(int)
    root = C.X2C_IMG_ROOT if (C.X2C_IMG_ROOT / "_mirrored.ok").exists() else C.X2C_DIR
    m["path"] = [str(root / f"images{i // 10000}" / f) for i, f in zip(m.idx, m.file_name)]
    y = np.vstack(m.ctrl_value.values).astype(np.float32)
    return m, y


class X2CSet(Dataset):
    def __init__(self, paths, y, tf):
        self.paths, self.y, self.tf = paths, y, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


def build_model(pretrained=True):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                           nn.Linear(128, len(C.X2C_CONTROLS)))
    return net


NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
EVAL_TF = transforms.Compose([transforms.Resize((C.X2C_IMG, C.X2C_IMG)), transforms.ToTensor(), NORM])
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(C.X2C_IMG, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05), transforms.RandomGrayscale(0.2),
    transforms.ToTensor(), NORM])


def mirror_images():
    """Copy images0-9 to the SSD once (sequential copy; see config.X2C_IMG_ROOT)."""
    import shutil
    ok = C.X2C_IMG_ROOT / "_mirrored.ok"
    if ok.exists():
        return
    for k in range(10):
        src, dst = C.X2C_DIR / f"images{k}", C.X2C_IMG_ROOT / f"images{k}"
        dst.mkdir(parents=True, exist_ok=True)
        have = {p.name for p in dst.iterdir()}
        for f in sorted(src.iterdir()):
            if f.name not in have:
                shutil.copyfile(f, dst / f.name)
        print("mirrored", src.name, flush=True)
    ok.write_text("ok")


def main():
    mirror_images()
    torch.manual_seed(C.SEED)
    m, y = load_meta()
    lo, hi = y.min(0), y.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    ys = (y - lo) / span
    (C.MODEL_DIR / "x2c_target_scaling.json").write_text(json.dumps(
        dict(controls=C.X2C_CONTROLS, min=lo.tolist(), max=hi.tolist())), encoding="utf-8")
    pd.DataFrame(dict(index=range(30), control=C.X2C_CONTROLS, min=lo, max=hi,
                      median=np.median(y, 0), sd=y.std(0))).to_csv(
        C.RESULTS_DIR / "x2c_control_mapping.csv", index=False, encoding="utf-8-sig")

    block = m.idx.values // C.X2C_BLOCK
    rng = np.random.default_rng(C.SEED)
    ub = np.unique(block)
    val_blocks = set(rng.choice(ub, int(round(len(ub) * C.X2C_VAL_FRAC)), replace=False))
    is_val = np.isin(block, list(val_blocks))
    tr = DataLoader(X2CSet(m.path.values[~is_val], ys[~is_val], TRAIN_TF), batch_size=C.X2C_BATCH,
                    shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    va = DataLoader(X2CSet(m.path.values[is_val], ys[is_val], EVAL_TF), batch_size=128,
                    num_workers=4, pin_memory=True, persistent_workers=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    total = C.X2C_EPOCHS * len(tr)
    warm = len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler()
    loss_fn = nn.HuberLoss(delta=0.1)
    start, best, hist = 0, np.inf, []
    last = C.MODEL_DIR / "x2c_last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["hist"]
        print(f"resumed at epoch {start}", flush=True)

    for ep in range(start, C.X2C_EPOCHS):
        net.train(); t0 = time.time(); run = 0.0
        for i, (xb, yb) in enumerate(tr):
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16):
                loss = loss_fn(net(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if i % 200 == 0:
                print(f"epoch {ep} batch {i}/{len(tr)} loss {loss.item():.4f}", flush=True)
        net.eval(); errs = []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16):
            for xb, yb in va:
                errs.append((net(xb.to(dev)).float().cpu() - yb).abs().numpy())
        errs = np.concatenate(errs)
        mae = float(errs.mean())
        hist.append(dict(epoch=ep, train_loss=run / len(tr), val_mae=mae, sec=time.time() - t0))
        print(hist[-1], flush=True)
        if mae < best:
            best = mae
            torch.save(net.state_dict(), C.MODEL_DIR / "x2c_resnet18.pt")
            pd.DataFrame(dict(control=C.X2C_CONTROLS, val_mae=errs.mean(0))).to_csv(
                C.RESULTS_DIR / "x2c_val_mae_per_control.csv", index=False, encoding="utf-8-sig")
        torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep, best=best, hist=hist), last)
        pd.DataFrame(hist).to_csv(C.RESULTS_DIR / "x2c_training_log.csv", index=False)
    print("best val MAE (scaled 0-1):", best, "| paper reports 0.0114", flush=True)


if __name__ == "__main__":
    main()
