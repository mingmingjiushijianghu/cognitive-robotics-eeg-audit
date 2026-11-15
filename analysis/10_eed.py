"""Step 10 (design §6, V10): Embodied Emotional Distance (EED).

Feature vector per participant x shared emotion (14 features):
  ERP  : P300, LPP subject means in baseline-SD units (V4; N170 dropped, V15)
  conn : Delta dwPLI F-P, F-T, P-T in theta/alpha/beta; Delta global-efficiency AUC per band
Kinematic correction (V17): every feature is shifted by beta_k * (A - mean A of AFFEC), where
  A = objective eye-opening amplitude max |log(eye area / neutral)| (same definition for both
  datasets, step 04) and beta_k = AFFEC slope (ERP: OLS on subject x clip means; conn: clip-level
  OLS on the step-9 clip connectivity), so both forms are compared at equal eye movement.
Scaling (V10): each feature divided by its SD over both datasets pooled - no per-dataset
  centring, so the overall embodiment difference is retained.
EED_e = Mahalanobis distance between the physical and virtual mean vectors with a
  Ledoit-Wolf shrunk pooled within-group covariance; EED = mean over the 3 emotions.
Inference: label permutation at participant level (5,000) and participant bootstrap (5,000).
Decomposition: d_k * (Sigma^-1 d)_k (sums to EED_e^2).
Profile EED: same after centring each participant's features across their 3 emotions
  (the original §6.1 variant; answers "different emotion-differentiation pattern?").

Outputs results/eed_*.csv, results/eed_summary.md
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.covariance import LedoitWolf

import config as cfg
from common import log
import importlib

R = cfg.RESULTS_DIR
S6 = importlib.import_module("06_erp_stats")
rng = np.random.default_rng(cfg.SEED)
N = cfg.N_PERM


def erp_features():
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    av = "expr" if S6.primary_lock() == "expression" and (trials.version == "expr").any() else "pipeline"
    trials = trials[trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED]      # V14
    t = trials[trials.emotion.isin(cfg.SHARED_EMOTIONS) & trials.component.isin(["P300", "LPP"]) &
               (((trials.dataset == "miko") & (trials.version == "pipeline")) |
                ((trials.dataset == "affec") & (trials.version == av)))]
    f = t.groupby(["dataset", "subject", "emotion", "component"]).amp_z.mean().unstack("component")
    clip = t.groupby(["dataset", "subject", "emotion", "stim", "component"]).amp_z.mean().unstack("component").reset_index()
    return f.add_prefix("erp_"), clip


def conn_features():
    mod = pd.read_csv(R / "conn_module.csv")
    top = pd.read_csv(R / "conn_topology.csv")
    sel = lambda d: d[d.version == "pipeline"]
    m = sel(mod).melt(id_vars=["dataset", "subject", "emotion", "band"], value_vars=["F-P", "F-T", "P-T"])
    m["feat"] = "conn_" + m.band + "_" + m.variable
    g = sel(top).assign(feat=lambda d: "geff_" + d.band, value=lambda d: d.geff)
    both = pd.concat([m[["dataset", "subject", "emotion", "feat", "value"]], g[["dataset", "subject", "emotion", "feat", "value"]]])
    return both.pivot_table(index=["dataset", "subject", "emotion"], columns="feat", values="value")


def slopes_eye(clip_erp, kin):
    """Per-feature slope on A_eye estimated in AFFEC only."""
    b = {}
    ce = clip_erp.merge(kin[["stim", "A_eye"]], on="stim")
    ce = ce[ce.dataset == "affec"]
    for c in ("P300", "LPP"):                                  # V15
        b[f"erp_{c}"] = smf.ols(f"{c} ~ A_eye + C(emotion)", ce).fit().params["A_eye"]
    cc = R / "bb_clip_connectivity.csv"
    if cc.exists():
        d = pd.read_csv(cc)
        d = d[d.emotion.isin(cfg.SHARED_EMOTIONS)].drop(columns=["A_eye"], errors="ignore")
        d = d.merge(kin[["stim", "A_eye"]], on="stim")
        for band in cfg.BANDS:
            for mod in ("F-P", "F-T", "P-T"):
                b[f"conn_{band}_{mod}"] = smf.ols(f"Q('{mod}') ~ A_eye + C(emotion)", d[d.band == band]).fit().params["A_eye"]
    return b


def mean_A_eye(clip_erp, kin):
    k = kin.set_index("stim").A_eye
    ce = clip_erp.assign(A_eye=clip_erp.stim.map(k))
    return ce.groupby(["dataset", "subject", "emotion"]).A_eye.mean()


def eed(F, phys):
    """F: subjects x emotions x features; phys: bool per subject. Returns EED_e, EED, contributions."""
    out, contrib = [], []
    for e in range(F.shape[1]):
        X = F[:, e, :]
        a, b = X[phys], X[~phys]
        d = a.mean(0) - b.mean(0)
        resid = np.vstack([a - a.mean(0), b - b.mean(0)])
        S = LedoitWolf().fit(resid).covariance_
        w = np.linalg.solve(S, d)
        out.append(np.sqrt(max(d @ w, 0)))
        contrib.append(d * w)
    return np.array(out), float(np.mean(out)), np.array(contrib)


def run(F, phys, feats, tag):
    obs_e, obs, contrib = eed(F, phys)
    null = np.array([eed(F, rng.permutation(phys))[1] for _ in range(N)])
    ip, iv = np.where(phys)[0], np.where(~phys)[0]
    boot = np.array([eed(F[np.r_[rng.choice(ip, len(ip)), rng.choice(iv, len(iv))]],
                         np.r_[np.ones(len(ip), bool), np.zeros(len(iv), bool)])[1] for _ in range(N)])
    summ = dict(analysis=tag, EED=obs, p_perm=(1 + (null >= obs).sum()) / (N + 1),
                ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), null_median=np.median(null),
                **{f"EED_{e}": v for e, v in zip(cfg.SHARED_EMOTIONS, obs_e)})
    con = pd.DataFrame(contrib, index=cfg.SHARED_EMOTIONS, columns=feats).T.assign(analysis=tag)
    return summ, con


def main():
    ferp, clip = erp_features()
    fcon = conn_features()
    # V17: objective eye-opening amplitude for both datasets (column named A_eye below)
    kin = pd.read_csv(R / "eye_geometry_kinematics.csv").rename(columns={"eye_logratio_max": "A_eye"})
    F = ferp.join(fcon, how="inner").dropna()
    feats = list(F.columns)

    beta = slopes_eye(clip, kin)
    aeye = mean_A_eye(clip, kin).reindex(F.index)
    ref = aeye.xs("affec", level="dataset").mean()
    Fk = F.copy()
    for c in feats:
        if c in beta and np.isfinite(beta[c]):
            Fk[c] = F[c] - beta[c] * (aeye - ref).fillna(0)
    Fk = Fk / Fk.std()                                                   # V10: pooled scaling

    subjects = Fk.reset_index()[["dataset", "subject"]].drop_duplicates()
    full = [s for s in subjects.itertuples(index=False)
            if all((s.dataset, s.subject, e) in Fk.index for e in cfg.SHARED_EMOTIONS)]
    arr = np.stack([[Fk.loc[(s.dataset, s.subject, e)].values for e in cfg.SHARED_EMOTIONS] for s in full])
    phys = np.array([s.dataset == "miko" for s in full])
    log(f"EED on {phys.sum()} physical vs {(~phys).sum()} virtual participants, {len(feats)} features")

    # V17: primary = kinematics-corrected with the objective eye-opening amplitude (§6.1)
    raw = np.stack([[(F / F.std()).loc[(s.dataset, s.subject, e)].values for e in cfg.SHARED_EMOTIONS] for s in full])
    keep = [i for i, f in enumerate(feats) if "beta" not in f]          # muscle-sensitive band removed (V16)
    results = [run(arr, phys, feats, "EED (primary, eye-geometry corrected)"),
               run(arr - arr.mean(1, keepdims=True), phys, feats, "EED profile (within-participant centred)"),
               run(arr[:, :, keep], phys, [feats[i] for i in keep], "EED without beta-band features"),
               run(raw, phys, feats, "EED (no kinematic correction)")]
    summary = pd.DataFrame([r[0] for r in results])
    contrib = pd.concat([r[1] for r in results]).rename_axis("feature").reset_index()
    summary.to_csv(R / "eed_summary.csv", index=False, encoding="utf-8-sig")
    contrib.to_csv(R / "eed_contributions.csv", index=False, encoding="utf-8-sig")
    pd.Series(beta, name="slope_A_eye").to_csv(R / "eed_kinematic_slopes.csv", encoding="utf-8-sig")
    with open(R / "eed_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# 具身情感距离 EED（自动生成）\n\n" + summary.round(4).to_markdown(index=False) +
                 "\n\n## 各特征贡献（d_k·(Σ⁻¹d)_k，和为 EED_e²）\n\n" + contrib.round(4).to_markdown(index=False) + "\n")
    log(summary.round(4).to_string())


if __name__ == "__main__":
    main()
