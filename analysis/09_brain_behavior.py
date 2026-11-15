"""Step 9 (design §5.3, V5, V11): route 3 - kinematics -> neural response, subjective contagion.

AFFEC (84 clips with varying intensity) carries the slope estimates:
  ERP  : amp_z ~ zA + zV + zS + emotion + speech_onset [+ roi] + (1|subject) + (1|stim)
  conn : clip-level Delta dwPLI (trials of all participants who saw the clip pooled,
         CONN_N per draw, 100 draws) ~ zA + zV + zS + emotion          (OLS, n = 84 clips)
  SCI  : subjective contagion = felt arousal x congruence (felt/perceived valence on the same
         side of 5 and |difference| <= 2) ~ zA + zV + zS + emotion + (1|subject) + (1|stim)
Marker screening (§5.3): FDR-significant slope on A, LOSO sign stability >= 90 %.
Embodiment modulation (V5/V17): AFFEC model on the objective eye-opening amplitude
A_geo = max |log(eye area / neutral)| (same definition for both datasets, step 04), evaluated
at Miko's A_geo -> physical gain = Miko observed - AFFEC prediction, participant bootstrap
(5,000). Compensation = gain / slope (A_geo a virtual face would need to add).
H4 (V11): Miko has no trial ratings, so the neural/subjective slope ratio is estimable in
AFFEC only; it is reported with a participant bootstrap CI.

Outputs results/bb_*.csv, results/bb_summary.md
"""
import importlib
import warnings
import numpy as np
import pandas as pd
import mne
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import config as cfg
from common import log, eligible

warnings.filterwarnings("ignore")
R = cfg.RESULTS_DIR
S6 = importlib.import_module("06_erp_stats")
S8 = importlib.import_module("08_connectivity")
rng = np.random.default_rng(cfg.SEED)
COMPS = ["P300", "LPP"]                 # V15: 16-ch N170 failed the §3.3 validity check
N_BOOT = 5000


def load():
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    kin = pd.read_csv(R / "stim_kinematics.csv")
    lock = S6.primary_lock()
    av = "expr" if lock == "expression" and (trials.version == "expr").any() else "pipeline"
    trials = trials[trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED]      # V14
    t = trials[trials.component.isin(COMPS) & (((trials.dataset == "miko") & (trials.version == "pipeline")) |
                                              ((trials.dataset == "affec") & (trials.version == av)))].copy()
    return t, kin


# V16: two kinematic sets. "bs" = MediaPipe blendshapes (built for human faces) is primary;
# "x2c" = the framework's 30-control space, kept as the secondary analysis because its
# cross-domain validity on AFFEC faces is weak (results/x2c_crossdomain_validity.csv).
KINSETS = {"bs": {"A": "bs_amp", "V": "bs_speed", "S": "bs_sym"}, "x2c": {"A": "A", "V": "V", "S": "S"}}


def with_kinematics(t, kin, kset):
    """Attach z-scored A/V/S of one kinematic set (z over the 84 AFFEC clips)."""
    k = kin[kin.dataset == "affec"].copy()
    cols = {}
    for short, col in KINSETS[kset].items():
        cols[f"z{short}"] = (k[col] - k[col].mean()) / k[col].std()
    k = k[["stim"]].assign(**cols)
    return t.drop(columns=[c for c in ("zA", "zV", "zS") if c in t], errors="ignore").merge(k, on="stim", how="left")


def sci(df):
    fv, pv = df.f_emotion_v - 5, df.p_emotion_v - 5
    cong = (np.sign(fv) == np.sign(pv)) & ((df.f_emotion_v - df.p_emotion_v).abs() <= 2)
    return df.f_emotion_a * cong.astype(float)


def erp_slopes(t):
    a = t[t.dataset == "affec"].dropna(subset=["zA", "zV", "zS"])
    rows, loso = [], []
    for comp in COMPS:
        d = a[a.component == comp]
        hemi = " + C(roi)" if d.roi.nunique() > 1 else ""
        res = S6.fit_lmm(d, "zA + zV + zS + C(emotion) + speech_onset" + hemi)
        for k in ("zA", "zV", "zS"):
            rows.append(dict(feature=comp, predictor=k, beta=res.fe_params[k], se=res.bse[k], p=res.pvalues[k]))
        signs = []
        full_sign = np.sign(res.fe_params["zA"])
        for s in d.subject.unique():
            o = smf.ols("amp_z ~ zA + zV + zS + C(emotion) + speech_onset" + hemi, d[d.subject != s]).fit()
            signs.append(np.sign(o.params["zA"]) == full_sign)
        loso.append(dict(feature=comp, loso_sign_stability=float(np.mean(signs))))
    return pd.DataFrame(rows), pd.DataFrame(loso)


def clip_connectivity(kin):
    """Delta dwPLI per clip, pooling the trials of all AFFEC participants who saw it."""
    eps = [mne.read_epochs(f, preload=True, verbose=False)
           for f in sorted((cfg.DERIV_DIR / "affec").glob("sub-*_pipeline-epo.fif"))]
    eps = [e for e in eps if len(e) and eligible(e)]                 # V14
    ep = mne.concatenate_epochs(eps, verbose=False)
    rows = []
    for stim in ep.metadata.stim.unique():
        e = ep[(ep.metadata.stim == stim).values]
        if len(e) < cfg.CONN_N_SENS:
            continue
        w, sf = S8.windows(e)
        ims = {k: S8.imag_cross(v, sf) for k, v in w.items()}
        for b in cfg.BANDS:
            acc = 0
            for _ in range(cfg.CONN_REPEATS):
                pick = rng.choice(len(e), cfg.CONN_N_SENS, replace=False)
                post = 0.5 * (S8.dwpli(ims["post1"][b][pick]) + S8.dwpli(ims["post2"][b][pick]))
                acc = acc + (post - S8.dwpli(ims["base"][b][pick]))
            rows.append(dict(stim=stim, band=b, n_trials=len(e), **S8.module_means(acc / cfg.CONN_REPEATS)))
    df = pd.DataFrame(rows).merge(kin, on="stim")
    df.to_csv(R / "bb_clip_connectivity.csv", index=False, encoding="utf-8-sig")
    return df


def clip_slopes(df):
    out = []
    for b in cfg.BANDS:
        for mod in ("F-P", "F-T", "P-T"):
            d = df[df.band == b]
            o = smf.ols(f"Q('{mod}') ~ zA + zV + zS + C(emotion)", d).fit()
            for k in ("zA", "zV", "zS"):
                out.append(dict(feature=f"conn_{b}_{mod}", predictor=k, beta=o.params[k], se=o.bse[k], p=o.pvalues[k]))
    return pd.DataFrame(out)


def eye_geo():
    """V17: objective eye-opening kinematics shared by both datasets (from step 04)."""
    g = pd.read_csv(R / "eye_geometry_kinematics.csv")
    return g.set_index("stim").eye_logratio_max.to_dict()


def physical_gain(t, kin):
    """V5/V17: AFFEC model  amp ~ A_geo + emotion  (A_geo = max |log eye area / neutral|),
    evaluated at Miko's A_geo -> physical gain = Miko observed - AFFEC prediction."""
    geo = eye_geo()
    rows = []
    for comp in COMPS:
        d = t[(t.component == comp) & t.emotion.isin(cfg.SHARED_EMOTIONS)]
        sm = d.groupby(["dataset", "subject", "emotion", "stim"], as_index=False).amp_z.mean()
        sm["A_geo"] = sm.stim.map(geo)
        aff, mik = sm[sm.dataset == "affec"].dropna(), sm[sm.dataset == "miko"].dropna()

        def gain(aff_s, mik_s):
            o = smf.ols("amp_z ~ A_geo + C(emotion)", aff_s).fit()
            return float((mik_s.amp_z - o.predict(mik_s)).mean()), float(o.params["A_geo"])
        g, slope = gain(aff, mik)
        sa, sm_ = aff.subject.unique(), mik.subject.unique()
        boots = np.array([gain(pd.concat([aff[aff.subject == s] for s in rng.choice(sa, len(sa))]),
                               pd.concat([mik[mik.subject == s] for s in rng.choice(sm_, len(sm_))]))[0]
                          for _ in range(N_BOOT)])
        lo, hi = aff.A_geo.min(), aff.A_geo.max()
        rows.append(dict(feature=comp, slope_A_geo=slope, miko_A_geo=mik.groupby("stim").A_geo.first().round(3).to_dict(),
                         affec_A_geo_range=f"{lo:.3f}-{hi:.3f}",
                         miko_inside_affec_range=bool(mik.A_geo.between(lo, hi).all()), physical_gain=g,
                         ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         p_boot=float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
                         compensation_A_geo=g / slope if abs(slope) > 1e-9 else np.nan))
    return pd.DataFrame(rows)


def subjective(t, kset="bs"):
    a = t[(t.dataset == "affec") & (t.component == "LPP")].drop_duplicates(["subject", "stim", "trial_idx"]).copy()
    a["SCI"] = sci(a)
    res = S6.fit_lmm(a.dropna(subset=["SCI", "zA"]), "zA + zV + zS + C(emotion)", dv="SCI")
    rows = [dict(feature="SCI", predictor=k, beta=res.fe_params[k], se=res.bse[k], p=res.pvalues[k]) for k in ("zA", "zV", "zS")]
    # H4 (V11/V16): neural and subjective slopes with participant-bootstrap CIs; the ratio is
    # reported only when the subjective slope's CI excludes 0 (otherwise it is undefined).
    lpp = t[(t.dataset == "affec") & (t.component == "LPP")].groupby(["subject", "stim"], as_index=False).agg(
        amp_z=("amp_z", "mean"), zA=("zA", "first"), emotion=("emotion", "first"))
    sc = a.groupby(["subject", "stim"], as_index=False).SCI.mean()
    m = lpp.merge(sc, on=["subject", "stim"]).dropna()

    def slopes_(df):
        return (smf.ols("amp_z ~ zA + C(emotion)", df).fit().params["zA"],
                smf.ols("SCI ~ zA + C(emotion)", df).fit().params["zA"])
    bn, bs = slopes_(m)
    subs = m.subject.unique()
    boot = np.array([slopes_(pd.concat([m[m.subject == s] for s in rng.choice(subs, len(subs))])) for _ in range(1000)])
    bs_lo, bs_hi = np.percentile(boot[:, 1], [2.5, 97.5])
    defined = bs_lo > 0 or bs_hi < 0
    ratio = boot[:, 0] / boot[:, 1]
    # review round 1: the same predictors (amplitude + emotion) at the trial level with crossed
    # random intercepts for participant and clip, so the H4 slopes rest on one model family
    lmm = {}
    lt = t[(t.dataset == "affec") & (t.component == "LPP")].dropna(subset=["zA"])
    for name, df, dv in (("SCI", a.dropna(subset=["SCI", "zA"]), "SCI"), ("LPP", lt, "amp_z")):
        fit = S6.fit_lmm(df, "zA + C(emotion)", dv=dv)
        ci = fit.conf_int().loc["zA"]
        lmm[name] = dict(beta=fit.fe_params["zA"], lo=ci[0], hi=ci[1], p=fit.pvalues["zA"])
    # review round 4: which part of contagion follows motion amplitude?
    comp_rows = []
    fv, pv = a.f_emotion_v - 5, a.p_emotion_v - 5
    a["congruent"] = ((np.sign(fv) == np.sign(pv)) & ((a.f_emotion_v - a.p_emotion_v).abs() <= 2)).astype(float)
    for dv, nm in (("f_emotion_a", "felt arousal"), ("congruent", "valence congruence"), ("p_emotion_a", "perceived arousal")):
        df = a.dropna(subset=[dv, "zA"])
        fit = S6.fit_lmm(df, "zA + C(emotion)", dv=dv)
        ci = fit.conf_int().loc["zA"]
        comp_rows.append(dict(outcome=nm, beta=fit.fe_params["zA"], ci_lo=ci[0], ci_hi=ci[1], p=fit.pvalues["zA"],
                              mean=df[dv].mean()))
    pd.DataFrame(comp_rows).to_csv(R / f"bb_sci_components_{kset}.csv", index=False, encoding="utf-8-sig")
    h4 = pd.DataFrame([dict(dataset="affec", beta_neural_LPP=bn, bn_ci_lo=np.percentile(boot[:, 0], 2.5),
                            bn_ci_hi=np.percentile(boot[:, 0], 97.5), beta_subjective=bs, bs_ci_lo=bs_lo, bs_ci_hi=bs_hi,
                            ratio=bn / bs if defined else np.nan,
                            ratio_ci_lo=np.percentile(ratio, 2.5) if defined else np.nan,
                            ratio_ci_hi=np.percentile(ratio, 97.5) if defined else np.nan,
                            lmm_sci_beta=lmm["SCI"]["beta"], lmm_sci_lo=lmm["SCI"]["lo"], lmm_sci_hi=lmm["SCI"]["hi"],
                            lmm_sci_p=lmm["SCI"]["p"], lmm_lpp_beta=lmm["LPP"]["beta"], lmm_lpp_lo=lmm["LPP"]["lo"],
                            lmm_lpp_hi=lmm["LPP"]["hi"], lmm_lpp_p=lmm["LPP"]["p"],
                            note="" if defined else "subjective slope CI includes 0 -> ratio undefined"),
                       dict(dataset="miko", note="no trial ratings (V11)")])
    return pd.DataFrame(rows), h4


def run_kinset(t, kin, clip_df, kset):
    tk = with_kinematics(t, kin, kset)
    erp, loso = erp_slopes(tk)
    conn = clip_slopes(with_kinematics(clip_df, kin, kset))
    subj, h4 = subjective(tk, kset)
    slopes = pd.concat([erp, conn, subj], ignore_index=True)
    slopes["p_fdr"] = multipletests(slopes.p, method="fdr_bh")[1]
    for df in (slopes, loso, h4):
        df.insert(0, "kinematics", kset)
    return slopes, loso, h4


def main():
    t, kin = load()
    clip_df = clip_connectivity(kin)
    res = {k: run_kinset(t, kin, clip_df, k) for k in KINSETS}
    slopes = pd.concat([r[0] for r in res.values()], ignore_index=True)
    loso = pd.concat([r[1] for r in res.values()], ignore_index=True)
    h4 = pd.concat([r[2] for r in res.values()], ignore_index=True)
    gain = physical_gain(t, kin)
    markers = slopes[(slopes.kinematics == "bs") & (slopes.predictor == "zA")].merge(
        loso[loso.kinematics == "bs"].drop(columns="kinematics"), on="feature", how="left")
    markers = markers.merge(gain[["feature", "physical_gain", "ci_lo", "ci_hi", "p_boot"]], on="feature", how="left")
    markers["is_marker"] = (markers.p_fdr < .05) & (markers.loso_sign_stability.fillna(0) >= .9) & (markers.p_boot < .05)
    out = dict(slopes=slopes, loso=loso, physical_gain=gain, markers=markers, h4=h4)
    for k, v in out.items():
        v.to_csv(R / f"bb_{k}.csv", index=False, encoding="utf-8-sig")
    with open(R / "bb_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# 路线 3：行为–脑电关联（自动生成）\n\n主分析运动学 = MediaPipe blendshape（`bs`）；"
                 "框架规定的 X2C 30 维参数（`x2c`）为次要分析（V16）。\n\n")
        for k, v in out.items():
            fh.write(f"## {k}\n\n{v.round(4).to_markdown(index=False)}\n\n")
    log(markers.round(3).to_string())


if __name__ == "__main__":
    main()
