"""Step 5 (design §3.1-3.3, V1, V2, V4): single-trial ERP features for both datasets.

For every epoch: 30 Hz low-pass + (-0.2, 0) s baseline (common.erp_view), then for each
component/ROI in config.COMPONENTS the window mean amplitude in
  * amp_uv : microvolts (descriptive only)
  * amp_z  : baseline-SD units (V4) - each channel divided by the SD of its own
             pre-stimulus baseline (all trials of that participant pooled)

Versions
  miko/pipeline, miko/author (sensitivity), affec/pipeline (16 ch)
  affec/expr    : AFFEC re-locked to expression-motion onset (V2; from 04_stim_features)
  affec/full62  : 62-ch average reference, unified ROI and standard N170 sites (§3.3)

Outputs
  results/erp_trials.csv            one row per trial x component x ROI
  results/erp_subject_latency.csv   50 % fractional-area latency on subject averages
  derivatives/erp/<dataset>_<version>_<subject>-ave.fif   evokeds per emotion (figures, TANOVA)
"""
import numpy as np
import pandas as pd
import mne

import config as C
from common import log, erp_view

OUT = C.DERIV_DIR / "erp"
OUT.mkdir(parents=True, exist_ok=True)
META_KEEP = ["dataset", "subject", "emotion", "stim", "run", "f_emotion_v", "f_emotion_a",
             "p_emotion_v", "p_emotion_a"]


def baseline_sd(ep):
    b = ep.copy().crop(*C.BASELINE).get_data()          # trials x ch x time
    return b.transpose(1, 0, 2).reshape(len(ep.ch_names), -1).std(axis=1)


def components_for(version):
    comps = dict(C.COMPONENTS)
    if version == "full62":
        comps["N170_std"] = (0.130, 0.200, C.N170_STANDARD, -1, False)
    return comps


def trial_features(ep, version):
    """ep: ERP view (low-passed, baseline-corrected). Returns long-format DataFrame."""
    sd = baseline_sd(ep)
    data = ep.get_data()
    times = ep.times
    meta = ep.metadata.reset_index(drop=True)
    meta = meta.reindex(columns=[c for c in META_KEEP if c in meta.columns] +
                        [c for c in ("speech_onset", "expr_onset") if c in meta.columns])
    out = []
    for comp, (t0, t1, rois, pol, cross) in components_for(version).items():
        tm = (times >= t0) & (times <= t1)
        win = data[:, :, tm].mean(axis=2)                # trials x ch
        for roi, chans in rois.items():
            idx = [ep.ch_names.index(c) for c in chans]
            df = meta.copy()
            df["version"], df["component"], df["roi"], df["cross_dataset"] = version, comp, roi, cross
            df["trial_idx"] = np.arange(len(meta))
            df["amp_uv"] = win[:, idx].mean(axis=1) * 1e6
            df["amp_z"] = (win[:, idx] / sd[idx]).mean(axis=1)
            out.append(df)
    return pd.concat(out, ignore_index=True)


def frac_area_latency(evk, chans, t0, t1, pol, frac=0.5):
    d = evk.copy().pick(chans).data.mean(axis=0) * pol
    tm = (evk.times >= t0) & (evk.times <= t1)
    seg = np.clip(d[tm], 0, None)
    if seg.sum() <= 0:
        return np.nan
    cum = np.cumsum(seg) / seg.sum()
    return float(evk.times[tm][np.searchsorted(cum, frac)])


def subject_evokeds(ep, dataset, version, subject):
    evks, lat = [], []
    for emo in sorted(ep.metadata.emotion.unique()):
        e = ep[ep.metadata.emotion == emo]
        if len(e) == 0:
            continue
        evk = e.average()
        evk.comment = emo
        evks.append(evk)
        for comp, (t0, t1, rois, pol, cross) in components_for(version).items():
            for roi, chans in rois.items():
                lat.append(dict(dataset=dataset, version=version, subject=subject, emotion=emo,
                                n_trials=len(e), component=comp, roi=roi,
                                latency=frac_area_latency(evk, chans, t0, t1, pol)))
    mne.write_evokeds(OUT / f"{dataset}_{version}_{subject}-ave.fif", evks, overwrite=True, verbose=False)
    return lat


def relock_to_expression(lep, stim_tab):
    """Cut -0.5..1.0 s around each clip's expression-motion onset out of the long AFFEC epochs."""
    meta = lep.metadata.reset_index(drop=True).merge(
        stim_tab[["stim", "expr_onset", "speech_onset"]], on="stim", how="left")
    data, keep = lep.get_data(), []
    n = int(round((C.TMAX - C.TMIN) * C.SFREQ)) + 1
    segs = []
    for i, t_e in enumerate(meta.expr_onset.values):
        if not np.isfinite(t_e):
            continue
        s0 = int(round((t_e + C.TMIN - lep.tmin) * C.SFREQ))
        if s0 < 0 or s0 + n > data.shape[2]:
            continue
        segs.append(data[i, :, s0:s0 + n])
        keep.append(i)
    if not segs:
        return None
    ep = mne.EpochsArray(np.stack(segs), lep.info, tmin=C.TMIN, metadata=meta.iloc[keep].reset_index(drop=True),
                         verbose=False)
    view = erp_view(ep)
    bad = np.abs(view.get_data()).max(axis=(1, 2)) > C.REJECT_ABS_UV * 1e-6
    return view[~bad]


def main():
    stim_path = C.RESULTS_DIR / "affec_stim_features.csv"
    stim_tab = pd.read_csv(stim_path) if stim_path.exists() else None
    rows, lats = [], []
    jobs = [("miko", v, f) for v in ("pipeline", "author")
            for f in sorted((C.DERIV_DIR / "miko").glob(f"sub-*_{v}-epo.fif"))]
    jobs += [("affec", v, f) for v in ("pipeline", "full62")
             for f in sorted((C.DERIV_DIR / "affec").glob(f"sub-*_{v}-epo.fif"))]
    if stim_tab is not None:
        jobs += [("affec", "expr", f) for f in sorted((C.DERIV_DIR / "affec").glob("sub-*_long-epo.fif"))]
    for dataset, version, f in jobs:
        subject = f.name.split("_")[0].replace("sub-", "")
        ep = mne.read_epochs(f, preload=True, verbose=False)
        if version == "expr":
            ep = relock_to_expression(ep, stim_tab)
            if ep is None or len(ep) == 0:
                continue
        else:
            if dataset == "affec" and stim_tab is not None:
                ep.metadata = ep.metadata.reset_index(drop=True).merge(
                    stim_tab[["stim", "expr_onset", "speech_onset"]], on="stim", how="left")
            ep = erp_view(ep)
        rows.append(trial_features(ep, version))
        lats += subject_evokeds(ep, dataset, version, subject)
        log(dataset, version, subject, len(ep), "trials")
    trials = pd.concat(rows, ignore_index=True)
    trials.to_csv(C.RESULTS_DIR / "erp_trials.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lats).to_csv(C.RESULTS_DIR / "erp_subject_latency.csv", index=False, encoding="utf-8-sig")
    log("trials table:", trials.shape, "| versions:", trials.groupby(["dataset", "version"]).subject.nunique().to_dict())


if __name__ == "__main__":
    main()
