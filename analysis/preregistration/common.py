"""Preprocessing and I/O helpers shared by the Miko and AFFEC pipelines (design §3.1, V6).

Both datasets go through exactly the same functions:
  16 common channels -> average reference -> extended-Infomax ICA (14 comps, 1 Hz copy)
  -> EOG components via Fp1/Fp2 proxies (z = 3) -> epochs -0.5..1.0 s (broadband 0.5-60 Hz)
  -> autoreject (local) -> +-100 uV check on the 30 Hz low-passed, baseline-corrected copy.
Bad channels are detected (pyprep) and interpolated before the average reference (V12);
ICA is fitted while skipping segments > 400 uV peak-to-peak.

Epochs are stored broadband and without baseline correction; ERP code applies the
30 Hz low-pass and the -0.2..0 s baseline itself, connectivity code uses the broadband data.
"""
import json
import sys
import time
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

import config as C

mne.set_log_level("WARNING")


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


MAX_BAD_FRAC = 0.25


def detect_bads(raw):
    """V12: bad channels by pyprep deviation / correlation / HF-noise / flat criteria (no RANSAC,
    which is unreliable with 16 channels). Same rule for both datasets; at most 25 % of channels."""
    from pyprep.find_noisy_channels import NoisyChannels
    nc = NoisyChannels(raw.copy(), do_detrend=True, random_state=C.SEED)
    nc.find_bad_by_nan_flat(); nc.find_bad_by_deviation(); nc.find_bad_by_correlation(); nc.find_bad_by_hfnoise()
    scores = nc.get_bads(as_dict=True)
    bads = list(dict.fromkeys(nc.get_bads()))
    if len(bads) > MAX_BAD_FRAC * len(raw.ch_names):           # keep the worst ones by deviation z
        z = dict(zip(raw.ch_names, np.abs(nc._extra_info["bad_by_deviation"]["robust_channel_deviations"])))
        bads = sorted(bads, key=lambda c: -z.get(c, 0))[:int(MAX_BAD_FRAC * len(raw.ch_names))]
    return bads, {k: v for k, v in scores.items() if v and k != "bad_all"}


def interpolate_bads(raw):
    bads, why = detect_bads(raw)
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    return bads, why


def run_ica(raw, n_components=C.ICA_N_COMPONENTS):
    """Extended-Infomax ICA on a 1 Hz high-passed copy; blink components via Fp1/Fp2."""
    fit_raw = raw.copy().filter(C.ICA_HIGHPASS, None, verbose=False)
    ica = ICA(n_components=n_components, method=C.ICA_METHOD,
              fit_params=dict(extended=True), random_state=C.ICA_SEED, max_iter="auto")
    ica.fit(fit_raw, reject=dict(eeg=C.ICA_FIT_REJECT_UV * 1e-6), reject_by_annotation=True)
    eog_idx, _ = ica.find_bads_eog(fit_raw, ch_name=C.EOG_PROXY,
                                   threshold=C.EOG_Z_THRESHOLD, measure="zscore")
    _, r = ica.find_bads_eog(fit_raw, ch_name=C.EOG_PROXY, threshold=C.EOG_R_THRESHOLD,
                             measure="correlation")
    r = np.max(np.abs(np.atleast_2d(np.asarray(r))), axis=0)          # best of Fp1 / Fp2
    cand = set(int(i) for i in eog_idx) | set(int(i) for i in np.where(r >= C.EOG_R_THRESHOLD)[0])
    ica.exclude = sorted(sorted(cand, key=lambda i: -r[i])[:C.EOG_MAX_COMPONENTS])
    return ica, ica.exclude, float(r.max())


def erp_view(epochs):
    """30 Hz low-passed (zero-phase IIR) + baseline-corrected copy used for ERPs and rejection."""
    ep = epochs.copy().filter(None, C.ERP_LOWPASS, method="iir",
                              iir_params=dict(order=4, ftype="butter"), verbose=False)
    return ep.apply_baseline(C.BASELINE, verbose=False)


def run_autoreject(ep):
    """V12: autoreject (local) - per-trial, per-channel interpolation or rejection with
    cross-validated thresholds (Jas et al. 2017). Same settings for both datasets."""
    from autoreject import AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 4], consensus=np.linspace(0.2, 1.0, 5),
                    random_state=C.SEED, n_jobs=1, verbose=False)
    ep_clean, log_ = ar.fit_transform(ep, return_log=True)
    n_interp = int((log_.labels == 2).sum())
    return ep_clean, int(log_.bad_epochs.sum()), n_interp


def make_epochs(raw, events, event_id, metadata, tmin=C.TMIN, tmax=C.TMAX, reject_window=None,
                autoreject=True):
    """Broadband epochs -> autoreject -> drop trials whose ERP view still exceeds +-100 uV.
    Returns epochs, n lost to annotations, n dropped (autoreject + amplitude)."""
    ep = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None,
                    reject_by_annotation=True, preload=True, metadata=metadata,
                    on_missing="ignore", verbose=False)
    n_annot = len(metadata) - len(ep)
    n_ar = 0
    if autoreject:
        ep, n_ar, n_interp = run_autoreject(ep)
        ep.info["description"] = f"autoreject: {n_ar} epochs rejected, {n_interp} channel-epochs interpolated"
    view = erp_view(ep)
    if reject_window is not None:
        view = view.crop(*reject_window)
    peak = np.abs(view.get_data()).max(axis=(1, 2))
    bad = peak > C.REJECT_ABS_UV * 1e-6
    ep.drop(np.where(bad)[0], reason="ABS_100uV", verbose=False)
    return ep, n_annot, n_ar + int(bad.sum())


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def subjects_from_argv(all_subjects):
    """Optional CLI filter: `python script.py sub1 sub2` processes only those subjects."""
    wanted = sys.argv[1:]
    return [s for s in all_subjects if not wanted or s in wanted]
