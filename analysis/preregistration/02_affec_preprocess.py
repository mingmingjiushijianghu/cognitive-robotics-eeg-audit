"""Step 2 (design §2.2, §3.1, V2, V6, V7): AFFEC preprocessing -> broadband epochs.

Per participant (runs 1-3 only; run-0 holds the 4 practice trials):
  EDF (63 ch, 256 Hz, left-earlobe ref) -> drop A1 -> resample 128 Hz (anti-aliased)
  -> 0.5-60 Hz band-pass + 50 Hz notch (same as Miko) -> concatenate runs, then
  * pipeline : pick the 16 Miko channels -> identical ICA/epoch rules as Miko (common.py)
  * long     : same cleaned 16-ch data, -0.5..3.5 s, no amplitude rejection
               (re-locked to expression-motion onset later, V2)
  * full62   : all 62 scalp channels, average reference, same ICA rules
               (n_components scaled to the montage) -> §3.3 validation only
Time-locking: onset of `flag == video` in the BIDS events file.

Outputs
  derivatives/affec/sub-<id>_{pipeline,long,full62}-epo.fif, sub-<id>_ica.fif
  results/affec_preprocessing_qc.csv
Usage: python 02_affec_preprocess.py [sub-id ...]   (default: all participants)
"""
import io
import zipfile
import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

import config as C
from common import log, run_ica, make_epochs, interpolate_bads

OUT = C.DERIV_DIR / "affec"
OUT.mkdir(parents=True, exist_ok=True)
EMO_CODES = {"angry": 1, "happy": 2, "sad": 3, "fear": 4, "disgust": 5, "neutral": 6}
N_JOBS = 6


def extract_eeg():
    marker = C.AFFEC_BIDS / "_extracted.ok"
    if marker.exists():
        return
    log("extracting eeg.zip ->", C.AFFEC_BIDS)
    C.AFFEC_BIDS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(C.AFFEC_DIR / "eeg.zip") as z:
        for info in z.infolist():
            target = (C.AFFEC_BIDS / info.filename).resolve()
            assert C.AFFEC_BIDS.resolve() in target.parents or target == C.AFFEC_BIDS.resolve()
            if not target.exists() or target.stat().st_size != info.file_size:
                z.extract(info, C.AFFEC_BIDS)
    marker.write_text("ok")


def load_tables():
    """Events (video rows) and ratings for all participants, runs 1-3."""
    ev, beh = [], []
    with zipfile.ZipFile(C.AFFEC_DIR / "core.zip") as z:
        for n in z.namelist():
            if n.endswith("_events.tsv"):
                ev.append(pd.read_csv(io.BytesIO(z.read(n)), sep="\t"))
            elif n.endswith("_beh.tsv"):
                beh.append(pd.read_csv(io.BytesIO(z.read(n)), sep="\t"))
    ev = pd.concat(ev, ignore_index=True)
    beh = pd.concat(beh, ignore_index=True).rename(columns={"user": "subject"})
    ev = ev[ev.flag.eq("video") & ev.run.isin(C.AFFEC_RUNS)].copy()
    ev["subject"] = ev.subject.astype(str)
    beh["subject"] = beh.subject.astype(str)
    ev = ev.merge(beh.drop(columns=["trial_type", "stim_file"]), on=["subject", "run", "trial"], how="left")
    ev["emotion"] = ev.trial_type.str.lower().map(C.AFFEC_EMOTIONS)
    ev["stim"] = ev.stim_file.str.extract(r"([^/]+)\.mp4$")[0]
    return ev.rename(columns={"duration": "video_dur"})


def load_run(sub, run):
    f = C.AFFEC_BIDS / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-fer_run-{run}_eeg.edf"
    raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
    raw.drop_channels([c for c in C.AFFEC_DROP if c in raw.ch_names])
    raw.set_annotations(None)
    raw.resample(C.SFREQ, verbose=False)                     # includes anti-alias low-pass
    raw.filter(*C.BAND, verbose=False)
    raw.notch_filter(C.NOTCH, verbose=False)
    return raw


def process(sub, ev):
    rows = ev[ev.subject.eq(sub)].sort_values(["run", "trial"]).reset_index(drop=True)
    raws, samples, offset = [], [], 0
    for run in C.AFFEC_RUNS:
        r = rows[rows.run.eq(run)]
        if r.empty:
            continue
        raw = load_run(sub, run)
        s = np.round(r.onset.values * C.SFREQ).astype(int)
        assert s.max() < raw.n_times, (sub, run, s.max(), raw.n_times)
        samples.append(offset + s)
        offset += raw.n_times
        raws.append(raw)
    rows = rows[rows.run.isin([r for r in C.AFFEC_RUNS if (rows.run == r).any()])].reset_index(drop=True)
    raw_all = mne.concatenate_raws(raws)
    raw_all.set_montage(C.MONTAGE, on_missing="ignore")
    bads, why = interpolate_bads(raw_all)          # V12: detected and interpolated on all 62 channels
    events = np.column_stack([np.concatenate(samples), np.zeros(len(rows), int),
                              rows.emotion.map(EMO_CODES).values])
    meta = rows[["run", "trial", "emotion", "stim", "stim_file", "video_dur",
                 "f_emotion_v", "f_emotion_a", "p_emotion_v", "p_emotion_a"]].assign(
        dataset="affec", subject=sub)
    qc = dict(subject=sub, n_video=len(rows), sec_recorded=round(float(raw_all.times[-1]), 1),
              bads=",".join(bads), n_bads=len(bads), bads_common16=",".join(b for b in bads if b in C.COMMON_CHANNELS))

    # --- main: 16 common channels, identical rules to Miko ----------------------
    raw16 = raw_all.copy().pick(C.COMMON_CHANNELS)
    raw16.set_eeg_reference("average", projection=False, verbose=False)
    ica, eog_idx, max_r = run_ica(raw16)
    ica.apply(raw16, verbose=False)
    ica.save(OUT / f"sub-{sub}_ica.fif", overwrite=True, verbose=False)
    ep, n_annot, n_amp = make_epochs(raw16, events, EMO_CODES, meta)
    ep.save(OUT / f"sub-{sub}_pipeline-epo.fif", overwrite=True, verbose=False)
    qc.update(ica_eog=str(eog_idx), ica_eog_max_abs_score=round(max_r, 2),
              n_boundary=n_annot, n_amp_reject=n_amp, n_final=len(ep),
              **{f"n_{e}": int((ep.metadata.emotion == e).sum()) for e in EMO_CODES})

    # --- long epochs for expression-onset re-locking (V2) -----------------------
    lep = mne.Epochs(raw16, events, event_id=EMO_CODES, tmin=C.TMIN, tmax=C.AFFEC_LONG_TMAX,
                     baseline=None, reject_by_annotation=True, preload=True, metadata=meta,
                     on_missing="ignore", verbose=False)
    lep.save(OUT / f"sub-{sub}_long-epo.fif", overwrite=True, verbose=False)

    # --- full 62-channel montage for §3.3 validation ----------------------------
    raw62 = raw_all.copy()
    raw62.set_eeg_reference("average", projection=False, verbose=False)
    ica62, _, _ = run_ica(raw62, n_components=30)
    ica62.apply(raw62, verbose=False)
    ep62, _, n_amp62 = make_epochs(raw62, events, EMO_CODES, meta)
    ep62.save(OUT / f"sub-{sub}_full62-epo.fif", overwrite=True, verbose=False)
    qc.update(full62_ica_eog=str(ica62.exclude), full62_n_final=len(ep62))
    log(qc)
    return qc


def main():
    import sys
    extract_eeg()
    ev = load_tables()
    subs = sorted(ev.subject.unique())
    if len(sys.argv) > 1:
        subs = [s for s in subs if s in sys.argv[1:]]
    have = {p.name for p in (C.AFFEC_BIDS).glob("sub-*")}
    subs = [s for s in subs if f"sub-{s}" in have]
    todo = [s for s in subs if not (OUT / f"sub-{s}_full62-epo.fif").exists()]
    log(f"{len(subs)} participants with EEG, {len(todo)} to process")

    def safe(s):
        try:
            return process(s, ev)
        except Exception as e:                     # keep going; failures are listed in the QC file
            log("FAILED", s, repr(e))
            return dict(subject=s, error=repr(e))

    new = Parallel(n_jobs=N_JOBS)(delayed(safe)(s) for s in todo)
    qc_path = C.RESULTS_DIR / "affec_preprocessing_qc.csv"
    old = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()
    qc = pd.concat([old[~old.subject.isin(todo)] if len(old) else old, pd.DataFrame(new)], ignore_index=True)
    qc.sort_values("subject").to_csv(qc_path, index=False, encoding="utf-8-sig")
    print(qc.describe(include="all").T.to_string())
    if "error" in qc and qc.error.notna().any():
        raise SystemExit(f"{qc.error.notna().sum()} participants failed, see {qc_path}")


if __name__ == "__main__":
    main()
