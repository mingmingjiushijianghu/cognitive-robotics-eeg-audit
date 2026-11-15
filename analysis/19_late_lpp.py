"""Step 19 (review round 5): does a later positivity follow facial-motion amplitude in AFFEC?

The LPP window (400-800 ms after video onset) precedes the median motion peak (~1.4 s), so two
later windows are measured on the long AFFEC epochs (Pz/POz, 30 Hz low-pass, -0.2..0 s baseline,
baseline-SD units, +-100 uV rejection inside each window):
  sustained   1.0-2.5 s after video onset
  peak-locked 400-800 ms after each clip's motion peak (clips with peak <= 2.7 s)
Each is related to standardized motion amplitude (MediaPipe) with the H4 model: trial-level LMM,
amplitude + emotion, crossed random intercepts for participant and clip.

Output results/late_lpp.csv
"""
import importlib
import numpy as np
import pandas as pd
import mne

import config as cfg
from common import log, erp_view, eligible

R = cfg.RESULTS_DIR
S6 = importlib.import_module("06_erp_stats")
ROI = ["Pz", "POz"]


def main():
    stim = pd.read_csv(R / "affec_stim_features.csv").set_index("stim")
    za = (stim.bs_amp - stim.bs_amp.mean()) / stim.bs_amp.std()
    rows = []
    for f in sorted((cfg.DERIV_DIR / "affec").glob("sub-*_long-epo.fif")):
        sub = f.name.split("_")[0][4:]
        short = cfg.DERIV_DIR / "affec" / f"sub-{sub}_pipeline-epo.fif"
        if not eligible(mne.read_epochs(short, preload=False, verbose=False)):
            continue
        ep = mne.read_epochs(f, preload=True, verbose=False)
        ep = ep[ep.metadata.emotion.isin(cfg.SHARED_EMOTIONS).values]
        v = erp_view(ep)
        d = v.get_data(picks=ROI) * 1e6                                   # trials x 2 x time
        t = v.times
        allch = v.get_data() * 1e6
        sd = v.copy().crop(*cfg.BASELINE).get_data(picks=ROI).transpose(1, 0, 2).reshape(2, -1).std(1) * 1e6
        md = v.metadata.reset_index(drop=True)
        for i, r in md.iterrows():
            tp = stim.bs_t_peak.get(r.stim, np.nan)
            for name, (t0, t1) in (("sustained", (1.0, 2.5)), ("peak-locked", (tp + 0.4, tp + 0.8))):
                if not np.isfinite(t0) or t1 > t[-1]:
                    continue
                m = (t >= t0) & (t <= t1)
                if np.abs(allch[i][:, m | (t < 0)]).max() > cfg.REJECT_ABS_UV:
                    continue
                rows.append(dict(subject=sub, stim=r.stim, emotion=r.emotion, window=name,
                                 amp_z=float((d[i][:, m].mean(1) / sd).mean()), zA=za.get(r.stim, np.nan)))
    df = pd.DataFrame(rows).dropna()
    out = []
    for name, g in df.groupby("window"):
        fit = S6.fit_lmm(g, "zA + C(emotion)", dv="amp_z")
        ci = fit.conf_int().loc["zA"]
        out.append(dict(window=name, n_trials=len(g), n_participants=g.subject.nunique(), beta=fit.fe_params["zA"],
                        ci_lo=ci[0], ci_hi=ci[1], p=fit.pvalues["zA"], mean_amp=g.amp_z.mean()))
    out = pd.DataFrame(out)
    out.to_csv(R / "late_lpp.csv", index=False, encoding="utf-8-sig")
    log(out.round(4).to_string())


if __name__ == "__main__":
    main()
