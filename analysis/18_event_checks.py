"""Step 18 (review round 2): robustness of the C2 event-type check.

A. Expression-onset threshold: onset = first frame at which blendshape displacement from frame 0
   exceeds 10 / 20 / 30 % of its peak (20 % is the pre-specified value). For each threshold:
   median onset, share of clips with onset > 300 ms and the resulting primary-lock decision.
B. Event type within AFFEC: participant-level P300 and LPP (Pz/POz, baseline-SD units) for the same
   trials locked to video onset and to expression-motion onset; paired t-test and dz.

Output results/event_onset_thresholds.csv, results/event_type_within_affec.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import median_filter

import config as cfg
from common import log

R = cfg.RESULTS_DIR
EXCLUDE = ("eyeBlink", "eyeLook", "_neutral")


def onsets(frac):
    out = []
    for f in sorted((cfg.DERIV_DIR / "stim").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        bs, t, names = z["blendshapes"], z["times"], list(z["names"])
        bs = pd.DataFrame(bs).interpolate(limit_direction="both").values
        use = [i for i, n in enumerate(names) if not any(k in n for k in EXCLUDE)]
        d = median_filter(np.linalg.norm(bs[:, use] - bs[0, use], axis=1), 5, mode="nearest")
        out.append(float(t[np.argmax(d > frac * d.max())]))
    return np.array(out)


def main():
    rows = []
    for frac in (0.10, 0.20, 0.30):
        o = onsets(frac)
        late = float((o > cfg.EXPR_ONSET_RULE[1]).mean())
        rows.append(dict(threshold=frac, n_clips=len(o), median_onset_ms=1000 * np.median(o),
                         pct_after_300ms=100 * late,
                         primary_lock="expression" if late >= cfg.EXPR_ONSET_RULE[0] else "video"))
    a = pd.DataFrame(rows)
    a.to_csv(R / "event_onset_thresholds.csv", index=False, encoding="utf-8-sig")
    log(a.round(1).to_string())

    tr = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    tr = tr[(tr.dataset == "affec") & tr.component.isin(["P300", "LPP"]) & (tr.min_shared_trials >= cfg.MIN_TRIALS_SHARED)
            & tr.emotion.isin(cfg.SHARED_EMOTIONS)]
    m = tr.groupby(["component", "version", "subject"]).amp_z.mean().unstack("version")
    out = []
    for comp in ("P300", "LPP"):
        d = m.loc[comp][["pipeline", "expr"]].dropna()
        t, p = stats.ttest_rel(d.pipeline, d.expr)
        diff = d.pipeline - d.expr
        out.append(dict(component=comp, n=len(d), mean_video=d.pipeline.mean(), mean_expression=d.expr.mean(),
                        t=t, p=p, dz=diff.mean() / diff.std(ddof=1)))
    b = pd.DataFrame(out)
    b.to_csv(R / "event_type_within_affec.csv", index=False, encoding="utf-8-sig")
    log(b.round(4).to_string())


if __name__ == "__main__":
    main()
