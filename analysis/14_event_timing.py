"""Step 14 (V19/V20): event-timing validity audit - are the stimulus markers time-locked to the EEG?

For each participant the evoked response at the real markers is compared with responses at
surrogate markers. Each surrogate keeps the real inter-marker structure but shifts every run's
markers by one common random offset (circular within the run), so it preserves the marker count
and spacing and destroys only the alignment with the true stimulus times. Statistics:
  * GFP of the subject average, 50-500 ms (30 Hz low-pass, -0.2..0 s baseline)
  * inter-trial phase coherence (ITC), 2-8 Hz, 50-500 ms, mean over the 16 channels
p_surrogate = share of 500 surrogates with a value >= the real one.
Run on Miko (our pipeline and the authors' ICA-cleaned data) and on every AFFEC participant who
meets the V14 inclusion rule (same 16 channels, same cleaning) as the reference.
Also tabulates the Miko marker spacing against the 8-s design (4 s expression + 4 s neutral).

Output results/event_timing_check.csv, results/event_timing_summary.csv, results/miko_marker_intervals.csv
"""
import numpy as np
import pandas as pd
import mne
import importlib

import config as cfg
from common import log, run_ica, interpolate_bads

R = cfg.RESULTS_DIR
rng = np.random.default_rng(cfg.SEED)
N_SURR = 500
P01 = importlib.import_module("01_miko_preprocess")
P02 = importlib.import_module("02_affec_preprocess")


def stats_at(data, onsets, sf):
    """data: ch x samples (already cleaned, average-referenced, 30 Hz low-passed)."""
    pre, post = int(0.2 * sf), int(0.5 * sf)
    s0 = int(0.05 * sf)
    ok = onsets[(onsets - pre >= 0) & (onsets + post < data.shape[1])]
    ep = np.stack([data[:, o - pre:o + post] for o in ok])          # trials x ch x time
    ep = ep - ep[:, :, :pre].mean(2, keepdims=True)
    gfp = ep.mean(0)[:, pre + s0:].std(0).mean()
    spec = np.fft.rfft(ep[:, :, pre:] * np.hanning(post)[None, None, :], axis=2)
    f = np.fft.rfftfreq(post, 1 / sf)
    band = (f >= 2) & (f <= 8)
    ph = spec[:, :, band] / (np.abs(spec[:, :, band]) + 1e-20)
    itc = np.abs(ph.mean(0)).mean()
    return gfp, itc, len(ok)


def surrogate(runs, sf, rng):
    """runs: list of (data, onsets). One random circular shift per run."""
    out = []
    for _ in range(N_SURR):
        segs, ons, off = [], [], 0
        for data, on in runs:
            n = data.shape[1]
            sh = rng.integers(int(2 * sf), n - int(2 * sf))
            ons.append((on + sh) % n + off)
            segs.append(data); off += n
        out.append(stats_at(np.concatenate(segs, 1), np.concatenate(ons), sf)[:2])
    return np.array(out)


def miko_runs(k, version="pipeline"):
    m = P01.load_subject(k)
    raw, trials = P01.build_raw(m, "raw" if version == "pipeline" else "clean")
    interpolate_bads(raw)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    if version == "pipeline":
        ica, _, _ = run_ica(raw)
        ica.apply(raw, verbose=False)
    raw.filter(None, cfg.ERP_LOWPASS, verbose=False)
    d = raw.get_data() * 1e6
    keep = trials[~trials.overlap & ~trials.edge]
    runs, off = [], 0
    for run_no, r in enumerate(np.atleast_1d(m.EEG), start=1):
        n = np.asarray(r.raw).shape[1]
        runs.append((d[:, off:off + n], keep[keep.run == run_no].sample_in_run.values.astype(int)))
        off += n
    return runs


def affec_runs(sub, ev):
    rows = ev[ev.subject.eq(sub)]
    runs = []
    raws = []
    for run in cfg.AFFEC_RUNS:
        f = cfg.AFFEC_BIDS / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-fer_run-{run}_eeg.edf"
        if f.exists() and (rows.run == run).any():
            raws.append((P02.load_run(sub, run), rows[rows.run == run].onset.values))
    lens = [r.n_times for r, _ in raws]                     # concatenate_raws modifies raws[0] in place
    raw = mne.concatenate_raws([r.copy() for r, _ in raws])
    raw.set_montage(cfg.MONTAGE, on_missing="ignore")
    interpolate_bads(raw)
    raw.pick(cfg.COMMON_CHANNELS).set_eeg_reference("average", projection=False, verbose=False)
    ica, _, _ = run_ica(raw)
    ica.apply(raw, verbose=False)
    raw.filter(None, cfg.ERP_LOWPASS, verbose=False)
    d = raw.get_data() * 1e6
    off = 0
    for (r, on), n in zip(raws, lens):
        runs.append((d[:, off:off + n], np.round(on * cfg.SFREQ).astype(int)))
        off += n
    return runs


def check(dataset, subject, runs):
    sf = cfg.SFREQ
    data = np.concatenate([d for d, _ in runs], 1)
    offs = np.cumsum([0] + [d.shape[1] for d, _ in runs[:-1]])
    real = stats_at(data, np.concatenate([o + off for (_, o), off in zip(runs, offs)]), sf)
    import zlib
    sur = surrogate(runs, sf, np.random.default_rng([cfg.SEED, zlib.crc32(f"{dataset}/{subject}".encode())]))
    row = dict(dataset=dataset, subject=subject, n_trials=real[2], gfp_real=real[0], gfp_surr_median=np.median(sur[:, 0]),
               gfp_p=(1 + (sur[:, 0] >= real[0]).sum()) / (N_SURR + 1), itc_real=real[1],
               itc_surr_median=np.median(sur[:, 1]), itc_p=(1 + (sur[:, 1] >= real[1]).sum()) / (N_SURR + 1))
    log(row)
    return row


def marker_intervals():
    rows = []
    for k in range(1, 11):
        m = P01.load_subject(k)
        for run_no, r in enumerate(np.atleast_1d(m.EEG), start=1):
            idx = np.atleast_1d(r.stimuli.index).astype(int)
            n = np.asarray(r.raw).shape[1]
            for kind, d in (("stored order", np.diff(idx)), ("sorted", np.diff(np.sort(idx)))):
                rows += [dict(subject=f"P{k}", run=run_no, order=kind, interval_s=v / cfg.SFREQ) for v in d]
            rows.append(dict(subject=f"P{k}", run=run_no, order="tail without markers",
                             interval_s=(n - idx.max()) / cfg.SFREQ))
    df = pd.DataFrame(rows)
    df.to_csv(R / "miko_marker_intervals.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    from joblib import Parallel, delayed
    from common import eligible  # noqa: F401  (inclusion rule documented in config)
    marker_intervals()
    jobs = [("miko", f"P{k}", ("miko", k, "pipeline")) for k in range(1, 11)]
    jobs += [("miko_author", f"P{k}", ("miko", k, "author")) for k in range(1, 11)]
    qc = pd.read_csv(R / "affec_preprocessing_qc.csv")
    qc = qc[qc.n_final.notna()]
    ok = qc[qc[["n_angry", "n_happy", "n_sad"]].min(axis=1) >= cfg.MIN_TRIALS_SHARED].subject.tolist()
    jobs += [("affec", s, ("affec", s, None)) for s in ok]
    ev = P02.load_tables()

    def one(ds, subj, spec):
        kind, a, b = spec
        runs = miko_runs(a, b) if kind == "miko" else affec_runs(a, ev)
        return check(ds, subj, runs)
    rows = Parallel(n_jobs=4)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(R / "event_timing_check.csv", index=False, encoding="utf-8-sig")
    from scipy import stats
    stouffer = lambda p: stats.norm.sf(stats.norm.isf(np.clip(p, 1e-6, 1 - 1e-6)).sum() / np.sqrt(len(p)))
    summ = df.groupby("dataset").agg(n=("subject", "size"),
                                     group_itc_p=("itc_p", stouffer), group_gfp_p=("gfp_p", stouffer),
                                     gfp_real=("gfp_real", "median"), gfp_surr=("gfp_surr_median", "median"),
                                     frac_gfp_p05=("gfp_p", lambda p: (p < .05).mean()),
                                     itc_real=("itc_real", "median"), itc_surr=("itc_surr_median", "median"),
                                     frac_itc_p05=("itc_p", lambda p: (p < .05).mean()))
    summ.to_csv(R / "event_timing_summary.csv", encoding="utf-8-sig")
    log(summ.round(3).to_string())


if __name__ == "__main__":
    main()
