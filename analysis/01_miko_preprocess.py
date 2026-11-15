"""Step 1 (design §2.1, §3.1): Miko preprocessing -> broadband epochs.

Two versions are produced for each participant:
  * pipeline : the authors' `raw` signal processed with the unified pipeline in common.py
               (identical to the AFFEC script) -> main analysis.
  * author   : the authors' manually ICA-cleaned `clean` signal, only re-referenced and
               epoched with the same rules -> sensitivity analysis.

Miko's `raw` is already band-passed 0.5-60 Hz with a 49.5-50.5 Hz band-stop
(README), i.e. exactly config.BAND / config.NOTCH, so it is not filtered again.
Onsets come from a camera co-registered with the EEG (Wairagkar 2021, §IV-D); they are
stored out of order and ~20 % overlap, so they are sorted and overlapping trials dropped.

Outputs
  derivatives/miko/sub-P{k}_{pipeline|author}-epo.fif   (-0.5..1.0 s, 0.5-60 Hz, no baseline)
  derivatives/miko/sub-P{k}_ica.fif
  results/miko_preprocessing_qc.csv
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import mne

import config as C
from common import log, run_ica, make_epochs, interpolate_bads

OUT = C.DERIV_DIR / "miko"
OUT.mkdir(parents=True, exist_ok=True)
EVENT_ID = {name: code for code, name in C.MIKO_EMOTIONS.items()}


def load_subject(k):
    m = sio.loadmat(C.MIKO_DIR / f"P{k}.mat", squeeze_me=True, struct_as_record=False)[f"P{k}"]
    assert float(m.Fs) == C.SFREQ
    assert list(m.channels) == C.COMMON_CHANNELS, m.channels
    return m


def trial_table(run_struct, run_no, n_times, sfreq):
    """All stimulus onsets of one run, sorted in time, with exclusion flags."""
    idx = np.atleast_1d(run_struct.stimuli.index).astype(int)
    emo = np.atleast_1d(run_struct.stimuli.emotion).astype(int)
    order = np.argsort(idx, kind="stable")
    idx, emo = idx[order], emo[order]
    lo, hi = C.OVERLAP_WINDOW
    rows = []
    for i, (s, e) in enumerate(zip(idx, emo)):
        d = (np.delete(idx, i) - s) / sfreq
        overlap = bool(np.any((d > lo) & (d < hi)))
        edge = (s + C.TMIN * sfreq < 0) or (s + C.TMAX * sfreq >= n_times)
        rows.append(dict(run=run_no, sample_in_run=int(s), emotion_code=int(e),
                         emotion=C.MIKO_EMOTIONS[int(e)], overlap=overlap, edge=edge))
    return rows


def build_raw(m, field):
    """Concatenate the runs of one participant; return Raw and the trial table."""
    info = mne.create_info(C.COMMON_CHANNELS, C.SFREQ, "eeg")
    raws, trials, offset = [], [], 0
    for run_no, r in enumerate(np.atleast_1d(m.EEG), start=1):
        data = np.asarray(getattr(r, field), dtype=float) * 1e-6      # uV -> V
        raw = mne.io.RawArray(data, info.copy(), verbose=False)
        for t in trial_table(r, run_no, data.shape[1], C.SFREQ):
            t["sample"] = offset + t["sample_in_run"]
            trials.append(t)
        offset += data.shape[1]
        raws.append(raw)
    raw = mne.concatenate_raws(raws)        # adds BAD/EDGE boundary annotations
    raw.set_montage(C.MONTAGE, on_missing="ignore")
    return raw, pd.DataFrame(trials)


def main():
    qc = []
    for k in range(1, 11):
        m = load_subject(k)
        row = dict(subject=f"P{k}")
        for version, field in (("pipeline", "raw"), ("author", "clean")):
            raw, trials = build_raw(m, field)
            bads, why = interpolate_bads(raw)                 # V12, before the average reference
            row[f"{version}_bads"] = ",".join(bads)
            raw.set_eeg_reference("average", projection=False, verbose=False)
            if version == "pipeline":
                ica, eog_idx, max_r = run_ica(raw)
                ica.apply(raw, verbose=False)
                ica.save(OUT / f"sub-P{k}_ica.fif", overwrite=True, verbose=False)
                row.update(n_onsets=len(trials), n_overlap=int(trials.overlap.sum()),
                           n_edge=int((trials.edge & ~trials.overlap).sum()),
                           ica_eog=str(eog_idx), ica_eog_max_abs_score=round(max_r, 2),
                           ica_n_removed=len(ica.exclude),
                           author_n_ic_removed=len(np.atleast_1d(m.rejected_IC)))
            keep = trials[~trials.overlap & ~trials.edge].reset_index(drop=True)
            events = np.column_stack([keep["sample"], np.zeros(len(keep), int), keep["emotion_code"]])
            meta = keep[["run", "sample_in_run", "emotion"]].assign(
                dataset="miko", subject=f"P{k}", stim=keep["emotion"].map(lambda e: f"miko_{e}"))
            ep, n_annot, n_amp = make_epochs(raw, events, EVENT_ID, meta)
            ep.save(OUT / f"sub-P{k}_{version}-epo.fif", overwrite=True, verbose=False)
            counts = ep.metadata.emotion.value_counts()
            row[f"{version}_n_boundary"] = n_annot
            row[f"{version}_n_amp_reject"] = n_amp
            row[f"{version}_n_final"] = len(ep)
            for emo in C.MIKO_EMOTIONS.values():
                row[f"{version}_{emo}"] = int(counts.get(emo, 0))
        qc.append(row)
        log(row)

    qc = pd.DataFrame(qc)
    qc.to_csv(C.RESULTS_DIR / "miko_preprocessing_qc.csv", index=False, encoding="utf-8-sig")
    print(qc.to_string())


if __name__ == "__main__":
    main()
