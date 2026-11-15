"""Step 4 (design §5.1-5.2, V1, V2): AFFEC stimulus timing and face-motion features.

For every CREMA-D clip that was shown in runs 1-3:
  * speech_onset (V1): first time the 10 ms audio RMS (30 ms smoothed) exceeds 10 % of its peak
  * MediaPipe FaceLandmarker per frame: 52 blendshapes + 478 landmarks (face box saved for X2C)
  * expr_onset (V2): first time the blendshape displacement from frame 0 exceeds 20 % of its
    peak; blink and gaze blendshapes are excluded so that blinks do not count as expression
  * blendshape kinematics: amplitude, peak speed, time-to-peak, left/right symmetry
Also applies the pre-registered V2 rule: expression-locked epochs are primary if >= 70 % of
clips have expr_onset > 300 ms.

Miko: the five face images from the dataset README (neutral, happy, sad, angry, surprised)
are cut out and saved to stimuli/miko/ for step 07; MediaPipe cannot detect the eyes-only face.

Outputs
  results/affec_stim_features.csv, results/expr_lock_decision.json
  derivatives/stim/<stim>.npz  (per-frame blendshapes, face boxes, times)
"""
import json
import numpy as np
import pandas as pd
import av
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter, uniform_filter1d

import config as C
from common import log

OUT = C.DERIV_DIR / "stim"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = C.MODEL_DIR / "face_landmarker.task"
EXCLUDE_ONSET = ("eyeBlink", "eyeLook", "_neutral")
BS_PAIRS = [("browDownLeft", "browDownRight"), ("browOuterUpLeft", "browOuterUpRight"),
            ("eyeSquintLeft", "eyeSquintRight"), ("eyeWideLeft", "eyeWideRight"),
            ("cheekSquintLeft", "cheekSquintRight"), ("mouthSmileLeft", "mouthSmileRight"),
            ("mouthFrownLeft", "mouthFrownRight"), ("mouthStretchLeft", "mouthStretchRight"),
            ("mouthUpperUpLeft", "mouthUpperUpRight"), ("mouthLowerDownLeft", "mouthLowerDownRight"),
            ("noseSneerLeft", "noseSneerRight"), ("mouthDimpleLeft", "mouthDimpleRight")]


def speech_onset(path):
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return np.nan, np.nan
        st = c.streams.audio[0]
        res = av.AudioResampler(format="flt", layout="mono", rate=16000)
        x = np.concatenate([f.to_ndarray().ravel() for fr in c.decode(st) for f in res.resample(fr)])
    hop = 160
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="valid")[::hop])
    rms = uniform_filter1d(rms, 3)
    if rms.max() <= 0:
        return np.nan, len(x) / 16000
    on = np.argmax(rms > C.SPEECH_RMS_FRAC * rms.max()) * hop / 16000
    return float(on), len(x) / 16000


def face_track(path, landmarker):
    frames_bs, boxes, times, names = [], [], [], None
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            t = float(fr.pts * st.time_base)
            img = np.ascontiguousarray(fr.to_ndarray(format="rgb24"))
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img),
                                              int(round(t * 1000)))
            times.append(t)
            if res.face_blendshapes:
                frames_bs.append([b.score for b in res.face_blendshapes[0]])
                lm = np.array([[p.x, p.y] for p in res.face_landmarks[0]])
                h, w = img.shape[:2]
                boxes.append([lm[:, 0].min() * w, lm[:, 1].min() * h, lm[:, 0].max() * w, lm[:, 1].max() * h])
                names = [b.category_name for b in res.face_blendshapes[0]]
            else:
                frames_bs.append([np.nan] * 52)
                boxes.append([np.nan] * 4)
    times = np.asarray(times) - times[0]
    if names is None:
        raise RuntimeError(f"no face detected in {path.name}")
    return np.asarray(frames_bs, float), np.asarray(boxes, float), times, names


def kinematics(bs, names, times):
    ok = ~np.isnan(bs).any(1)
    bs = pd.DataFrame(bs, columns=names).interpolate(limit_direction="both").values
    use = [i for i, n in enumerate(names) if not any(k in n for k in EXCLUDE_ONSET)]
    disp = np.linalg.norm(bs[:, use] - bs[0, use], axis=1)
    disp_s = median_filter(disp, 5, mode="nearest")
    peak = disp_s.max()
    onset = float(times[np.argmax(disp_s > C.EXPR_ONSET_FRAC * peak)]) if peak > 0 else np.nan
    fps = 1 / np.median(np.diff(times))
    win = max(5, int(round(0.23 * fps)) | 1)
    vel = savgol_filter(bs[:, use], win, 2, deriv=1, axis=0) * fps
    ipk = int(np.argmax(disp_s))
    sym = 1 - np.mean([abs(bs[ipk, names.index(l)] - bs[ipk, names.index(r)]) for l, r in BS_PAIRS])
    return dict(expr_onset=onset, bs_amp=float(peak), bs_speed=float(np.linalg.norm(vel, axis=1).max()),
                bs_t_peak=float(times[ipk]), bs_sym=float(sym), face_detect_rate=float(ok.mean()), fps=float(fps))


def cut_miko_faces():
    """The README shows neutral / happy+sad / surprise / angry as separate images (page 2)."""
    from PIL import Image
    import pymupdf
    dst = C.ANALYSIS_DIR / "stimuli" / "miko"
    dst.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(C.MIKO_DIR / "README.pdf")
    imgs = []
    for im in doc[1].get_images(full=True):
        pix = pymupdf.Pixmap(doc, im[0])
        if pix.n - pix.alpha > 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples) if pix.alpha == 0
                    else Image.frombytes("RGBA", (pix.width, pix.height), pix.samples).convert("RGB"))
    faces = [im for im in imgs if im.height > 150]       # drop the banner strip
    # order on the page (checked visually): neutral | happy+sad | surprise | angry
    neutral, pair, surprised, angry = faces[:4]
    w = pair.width // 2
    out = dict(neutral=neutral, happy=pair.crop((0, 0, w, pair.height)),
               sad=pair.crop((w, 0, pair.width, pair.height)), surprised=surprised, angry=angry)
    for k, im in out.items():
        im.crop((0, 0, im.width, int(im.height * 0.86))).save(dst / f"{k}.png")   # drop caption text
    return sorted(out)


def main():
    log("Miko faces:", cut_miko_faces())
    ev = pd.read_csv(C.RESULTS_DIR / "affec_events_used.csv") if (C.RESULTS_DIR / "affec_events_used.csv").exists() else None
    if ev is None:
        import importlib
        ev = importlib.import_module("02_affec_preprocess").load_tables()
    stims = ev[["stim", "stim_file", "emotion"]].drop_duplicates("stim").reset_index(drop=True)
    log(len(stims), "unique clips")
    # the C library cannot open non-ASCII paths -> hand over the model bytes
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        output_face_blendshapes=True,
                                        min_face_detection_confidence=0.3)
    rows = []
    for _, r in stims.iterrows():
        path = C.AFFEC_STIM / r.stim_file
        on, dur = speech_onset(path)
        with vision.FaceLandmarker.create_from_options(opts) as lmk:
            bs, boxes, times, names = face_track(path, lmk)
        np.savez_compressed(OUT / f"{r.stim}.npz", blendshapes=bs, boxes=boxes, times=times, names=names)
        rows.append(dict(stim=r.stim, emotion=r.emotion, speech_onset=on, clip_dur=dur, n_frames=len(times),
                         **kinematics(bs, names, times)))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS_DIR / "affec_stim_features.csv", index=False, encoding="utf-8-sig")
    frac_late = float((df.expr_onset > C.EXPR_ONSET_RULE[1]).mean())
    decision = dict(frac_clips_onset_after_300ms=frac_late,
                    primary_lock="expression" if frac_late >= C.EXPR_ONSET_RULE[0] else "video",
                    median_expr_onset=float(df.expr_onset.median()),
                    median_speech_onset=float(df.speech_onset.median()),
                    frac_speech_before_200ms=float((df.speech_onset < 0.2).mean()))
    (C.RESULTS_DIR / "expr_lock_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    log(decision)
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
