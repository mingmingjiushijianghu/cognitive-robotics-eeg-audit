"""Step 7 (design §5.1.2-5.2, V5): stimuli in the X2C 30-control space + kinematic features.

AFFEC: every frame of each clip -> face box from step 4 (expanded 1.5x, square) -> X2C model
       -> c(t) in [0, 1]^30. c0 = first frame (start of the clip).
Miko : the five README faces -> X2C model. c0 = neutral face. Static, so no speed (V5).
Features (design §5.2; controls min-max scaled to [0, 1]):
  A       max_t ||c(t) - c0||  over the 23 non-pose controls
  A_eye   same over the 10-control eye subspace (V5; the only part Miko I can express)
  A_pose  same over the 7 gaze/head/neck controls
  V, V_eye  max_t ||dc/dt|| (Savitzky-Golay derivative, ~0.23 s window)
  S       1 - mean |c_L - c_R| over the 8 left/right pairs at the peak frame
  t_peak  time of peak displacement
Cross-domain validity (§5.1.3): CCA between per-frame X2C predictions and MediaPipe
blendshapes, fitted on half of the clips and evaluated on the other half, plus a priori
control/blendshape pairs. Miko: image-geometry check of the eye region and a coding
template for two independent raters (ICC once filled in).

Outputs results/stim_kinematics.csv, results/x2c_crossdomain_validity.csv,
        results/miko_eye_geometry.csv, results/miko_eye_coding_template.csv,
        derivatives/stim/<stim>_x2c.npy
"""
import json
import numpy as np
import pandas as pd
import torch
import av
from PIL import Image
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import CCA

import config as C
from common import log
import importlib

T = importlib.import_module("03_x2c_train")
STIM = C.DERIV_DIR / "stim"
POSE = [C.X2C_CONTROLS.index(c) for c in C.X2C_POSE]
EXPR = [i for i in range(30) if i not in POSE]
EYE = [C.X2C_CONTROLS.index(c) for c in C.X2C_EYE_SUBSPACE]
PAIRS = [(C.X2C_CONTROLS.index(l), C.X2C_CONTROLS.index(r)) for l, r in C.X2C_PAIRS]
A_PRIORI = [("LipCornerRaiseL", "mouthSmileLeft"), ("LipCornerRaiseR", "mouthSmileRight"),
            ("BrowInnerL", "browInnerUp"), ("BrowInnerR", "browInnerUp"),
            ("JawPitch", "jawOpen"), ("EyelidUpperL", "eyeWideLeft"), ("EyelidUpperR", "eyeWideRight"),
            ("NoseWrinkle", "noseSneerLeft"), ("LipCornerStretchL", "mouthStretchLeft")]


def load_model(dev):
    net = T.build_model(pretrained=False)
    net.load_state_dict(torch.load(C.MODEL_DIR / "x2c_resnet18.pt", map_location=dev))
    return net.to(dev).eval()


def square_crop(img, box, scale=1.5):
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale / 2
    return img.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s)))


@torch.no_grad()
def predict(net, imgs, dev):
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([T.EVAL_TF(im) for im in imgs[i:i + 64]]).to(dev)
        out.append(net(x).float().cpu().numpy())
    return np.clip(np.concatenate(out), 0, 1)


def features(c, times=None):
    c0 = c[0]
    d = lambda idx: np.linalg.norm(c[:, idx] - c0[idx], axis=1)
    disp = d(EXPR)
    ipk = int(np.argmax(disp))
    row = dict(A=float(disp.max()), A_eye=float(d(EYE).max()), A_pose=float(d(POSE).max()),
               S=float(1 - np.mean([abs(c[ipk, l] - c[ipk, r]) for l, r in PAIRS])))
    if times is not None and len(times) > 7:
        fps = 1 / np.median(np.diff(times))
        win = max(5, int(round(0.23 * fps)) | 1)
        vel = savgol_filter(c, win, 2, deriv=1, axis=0) * fps
        row.update(V=float(np.linalg.norm(vel[:, EXPR], axis=1).max()),
                   V_eye=float(np.linalg.norm(vel[:, EYE], axis=1).max()), t_peak=float(times[ipk]))
    return row


def affec(net, dev, stims):
    rows, X, B, groups, bs_names = [], [], [], [], None
    for _, r in stims.iterrows():
        z = np.load(STIM / f"{r.stim}.npz", allow_pickle=False)
        boxes, times, bs = z["boxes"], z["times"], z["blendshapes"]
        bs_names = list(z["names"])
        frames = []
        with av.open(str(C.AFFEC_STIM / r.stim_file)) as cont:
            for fr in cont.decode(video=0):
                frames.append(fr.to_image())
        n = min(len(frames), len(boxes))
        boxes = pd.DataFrame(boxes[:n]).interpolate(limit_direction="both").values
        c = predict(net, [square_crop(frames[i], boxes[i]) for i in range(n)], dev)
        np.save(STIM / f"{r.stim}_x2c.npy", c)
        rows.append(dict(dataset="affec", stim=r.stim, emotion=r.emotion, **features(c, times[:n])))
        X.append(c[:, EXPR]); B.append(np.nan_to_num(bs[:n])); groups += [r.stim] * n
    return pd.DataFrame(rows), np.vstack(X), np.vstack(B), np.array(groups), bs_names


def crossdomain(X, B, groups, bs_names):
    rng = np.random.default_rng(C.SEED)
    u = np.unique(groups)
    half = np.isin(groups, rng.choice(u, len(u) // 2, replace=False))
    cca = CCA(n_components=5, max_iter=2000).fit(X[half], B[half])
    xs, ys = cca.transform(X[~half], B[~half])
    rows = [dict(test="CCA out-of-sample", component=k + 1, r=float(np.corrcoef(xs[:, k], ys[:, k])[0, 1]))
            for k in range(5)]
    ctrl_expr = [C.X2C_CONTROLS[i] for i in EXPR]
    for cn, bn in A_PRIORI:
        if cn in ctrl_expr and bn in bs_names:
            rows.append(dict(test="a priori pair", component=f"{cn} ~ {bn}",
                             r=float(np.corrcoef(X[:, ctrl_expr.index(cn)], B[:, bs_names.index(bn)])[0, 1])))
    return pd.DataFrame(rows)


def miko(net, dev):
    d = C.ANALYSIS_DIR / "stimuli" / "miko"
    names = ["neutral", "angry", "happy", "sad", "surprised"]
    imgs = [Image.open(d / f"{n}.png").convert("RGB") for n in names]
    c = predict(net, imgs, dev)
    rows = []
    for i, n in enumerate(names[1:], start=1):
        f = features(np.vstack([c[0], c[i]]))
        rows.append(dict(dataset="miko", stim=f"miko_{n}", emotion=n, V=np.nan, V_eye=np.nan, t_peak=np.nan, **f))
    # model-free check: cyan eye pixels inside the screen (height, width, area relative to neutral)
    geo = []
    for n, im in zip(names, imgs):
        a = np.asarray(im).astype(int)
        eye = (a[:, :, 2] > 150) & (a[:, :, 1] > 120) & (a[:, :, 0] < 120)
        ys, xs = np.nonzero(eye)
        geo.append(dict(stim=n, eye_area=int(eye.sum()), eye_height=int(np.ptp(ys)) if len(ys) else 0,
                        eye_width=int(np.ptp(xs)) if len(xs) else 0))
    geo = pd.DataFrame(geo)
    for k in ("eye_area", "eye_height", "eye_width"):
        geo[f"{k}_rel_neutral"] = geo[k] / geo.loc[geo.stim == "neutral", k].values[0]
    geo.to_csv(C.RESULTS_DIR / "miko_eye_geometry.csv", index=False, encoding="utf-8-sig")
    tmpl = C.RESULTS_DIR / "miko_eye_coding_template.csv"
    if not tmpl.exists():
        pd.DataFrame([dict(rater="", stim=n, **{k: "" for k in C.X2C_EYE_SUBSPACE}) for n in names]).to_csv(
            tmpl, index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows), pd.DataFrame(c, columns=C.X2C_CONTROLS).assign(stim=names)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(dev)
    stims = pd.read_csv(C.RESULTS_DIR / "affec_stim_features.csv")
    ev = importlib.import_module("02_affec_preprocess").load_tables()
    stims = stims.merge(ev[["stim", "stim_file"]].drop_duplicates("stim"), on="stim")
    a, X, B, g, names = affec(net, dev, stims)
    log("AFFEC clips:", len(a))
    m, mc = miko(net, dev)
    mc.to_csv(C.RESULTS_DIR / "miko_x2c_controls.csv", index=False, encoding="utf-8-sig")
    kin = pd.concat([a.merge(stims.drop(columns=["emotion", "stim_file"]), on="stim", how="left"), m],
                    ignore_index=True)
    kin.to_csv(C.RESULTS_DIR / "stim_kinematics.csv", index=False, encoding="utf-8-sig")
    val = crossdomain(X, B, g, names)
    val.to_csv(C.RESULTS_DIR / "x2c_crossdomain_validity.csv", index=False, encoding="utf-8-sig")
    log(val.round(3).to_string())
    log(kin.groupby("dataset")[["A", "A_eye", "A_pose", "V", "S"]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
