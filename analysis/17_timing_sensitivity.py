"""Step 17 (review round 1): sensitivity of the C1 event-timing test.

Could the C1 null in Miko reflect a small response to the robot's eye changes rather than invalid
markers? A known response is injected into each Miko participant's continuous cleaned EEG at the
recorded markers, and the C1 test is repeated.
Template: the AFFEC grand-average response time-locked to expression-motion onset (16 channels,
-0.2..1.0 s, baseline-SD units), i.e. the response to an expression change on a visible face, the
event type that Miko presents. It is converted to each Miko participant's microvolts with the
participant's own channel baseline SD and scaled by k in SCALES. Each injection receives the
+-33 ms timing jitter of the 30-fps camera that produced the Miko markers.
Detection rate = share of participants with ITC p < .05 (200 surrogates).

Output results/timing_sensitivity.csv, results/timing_sensitivity_summary.csv
"""
import importlib
import numpy as np
import pandas as pd

import config as cfg
from common import log

R = cfg.RESULTS_DIR
T14 = importlib.import_module("14_event_timing")
R12 = importlib.import_module("12_robustness")
SCALES = [0.25, 0.5, 1.0]
JITTER_S = 1 / 30


def template():
    _, af = R12.build("pipeline", "expression")
    t = next(iter(af.values()))[1]
    z = np.stack([v[0] for v in af.values() if v[2] >= cfg.MIN_TRIALS_SHARED]).mean(0)    # ch x time
    keep = (t >= -0.2) & (t <= 1.0)
    return z[:, keep], t[keep]


def main():
    T14.N_SURR = 200
    tz, tt = template()
    sf = cfg.SFREQ
    n_pre = int(round(-tt[0] * sf))
    rows = []
    for k in range(1, 11):
        runs = T14.miko_runs(k, "pipeline")
        allx = np.concatenate([d for d, _ in runs], 1)
        on_all = np.concatenate([o + sum(r[0].shape[1] for r in runs[:i]) for i, (_, o) in enumerate(runs)])
        segs = np.stack([allx[:, o - n_pre:o] for o in on_all if o - n_pre >= 0])
        base_sd = segs.transpose(1, 0, 2).reshape(allx.shape[0], -1).std(1)          # uV per channel
        rng = np.random.default_rng([cfg.SEED, k])
        for scale in [0.0] + SCALES:
            inj = []
            for d, on in runs:
                d = d.copy()
                for o in on:
                    s0 = o - n_pre + int(round(rng.uniform(-JITTER_S, JITTER_S) * sf))
                    if s0 >= 0 and s0 + tz.shape[1] < d.shape[1]:
                        d[:, s0:s0 + tz.shape[1]] += scale * tz * base_sd[:, None]
                inj.append((d, on))
            row = T14.check("miko", f"P{k}", inj)
            rows.append(dict(subject=f"P{k}", scale=scale, itc_p=row["itc_p"], gfp_p=row["gfp_p"],
                             itc_real=row["itc_real"], itc_surr=row["itc_surr_median"]))
        log("done", k)
    df = pd.DataFrame(rows)
    df.to_csv(R / "timing_sensitivity.csv", index=False, encoding="utf-8-sig")
    from scipy import stats
    stouffer = lambda p: stats.norm.sf(stats.norm.isf(np.clip(p, 1e-6, 1 - 1e-6)).sum() / np.sqrt(len(p)))
    summ = df.groupby("scale").agg(detect_itc=("itc_p", lambda p: (p < .05).mean()),
                                   detect_gfp=("gfp_p", lambda p: (p < .05).mean()),
                                   group_itc_p=("itc_p", stouffer), group_gfp_p=("gfp_p", stouffer),
                                   itc_real=("itc_real", "median"), itc_surr=("itc_surr", "median")).reset_index()
    summ.to_csv(R / "timing_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    log(summ.round(3).to_string())


if __name__ == "__main__":
    main()
