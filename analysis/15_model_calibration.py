"""Step 15 (V13/V20): false-positive rate of the two candidate models for the embodiment test.

Null data: AFFEC participants only (V14 inclusion), so there is no true group difference.
Each simulation draws 10 participants as a pseudo-"physical" group and gives them Miko's
stimulus structure (one stimulus per emotion: stim := emotion), mirroring the real design.
Models compared on the same split (shared emotions, P300 and LPP, primary AFFEC version):
  A  single-trial LMM with crossed random intercepts for participant and stimulus
     (the original §3.4 model), LR test for the group effect
  B  participant x emotion cell-mean LMM with a participant intercept (V13), LR test
N_SIM splits; false-positive rate = share of p < .05, with a 95 % Clopper-Pearson interval.

Output results/model_calibration.csv, results/model_calibration_summary.csv
"""
import importlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from common import log

warnings.filterwarnings("ignore")
S6 = importlib.import_module("06_erp_stats")
R = cfg.RESULTS_DIR
N_SIM = 100
COMPS = ["P300", "LPP"]


def cp_interval(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    rng = np.random.default_rng(cfg.SEED)
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    av = "expr" if S6.primary_lock() == "expression" and (trials.version == "expr").any() else "pipeline"
    a = trials[(trials.dataset == "affec") & (trials.version == av) & trials.component.isin(COMPS) &
               trials.emotion.isin(cfg.SHARED_EMOTIONS) & (trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED)].copy()
    a["stim"] = a.stim.astype(str)
    subs = np.array(sorted(a.subject.unique()))
    rows = []
    for i in range(N_SIM):
        pseudo = set(rng.choice(subs, 10, replace=False))
        d = a.copy()
        d["embodiment"] = np.where(d.subject.isin(pseudo), "physical", "virtual")
        d.loc[d.embodiment == "physical", "stim"] = "pseudo_" + d.loc[d.embodiment == "physical", "emotion"]
        for comp in COMPS:
            x = d[d.component == comp].copy()
            x["amp_std"] = x.amp_z / x.amp_z.std()
            emb = "C(embodiment, Treatment('virtual'))"
            _, _, pA = S6.lr_test(x, f"{emb} + C(emotion)", "C(emotion)", dv="amp_std")
            c = S6.cell_means(x, "amp_std")
            _, _, pB = S6.cell_lr(c, f"{emb} + C(emotion)", "C(emotion)", "amp_std")
            rows.append(dict(sim=i, component=comp, p_single_trial_crossed=pA, p_cell_mean=pB))
        if i % 10 == 0:
            log("simulation", i)
    df = pd.DataFrame(rows)
    df.to_csv(R / "model_calibration.csv", index=False, encoding="utf-8-sig")
    out = []
    for model in ("p_single_trial_crossed", "p_cell_mean"):
        for comp, g in df.groupby("component"):
            k, n = int((g[model] < .05).sum()), len(g)
            lo, hi = cp_interval(k, n)
            out.append(dict(model=model.replace("p_", ""), component=comp, n_sim=n, false_positive_rate=k / n,
                            ci_lo=lo, ci_hi=hi, median_p=g[model].median()))
    out = pd.DataFrame(out)
    out.to_csv(R / "model_calibration_summary.csv", index=False, encoding="utf-8-sig")
    log(out.round(3).to_string())


if __name__ == "__main__":
    main()
