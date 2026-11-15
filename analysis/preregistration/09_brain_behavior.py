"""Step 9 (design §5.3, V5, V11): route 3 - kinematics -> neural response, subjective contagion.

AFFEC (84 clips with varying intensity) carries the slope estimates:
  ERP  : amp_z ~ zA + zV + zS + emotion + speech_onset [+ roi] + (1|subject) + (1|stim)
  conn : clip-level Delta dwPLI (trials of all participants who saw the clip pooled,
         CONN_N per draw, 100 draws) ~ zA + zV + zS + emotion          (OLS, n = 84 clips)
  SCI  : subjective contagion = felt arousal x congruence (felt/perceived valence on the same
         side of 5 and |difference| <= 2) ~ zA + zV + zS + emotion + (1|subject) + (1|stim)
Marker screening (§5.3): FDR-significant slope on A, LOSO sign stability >= 90 %.
Embodiment modulation (V5): model refit on the eye subspace (A_eye, shared emotions) and
evaluated at Miko's A_eye -> physical gain = Miko observed - AFFEC prediction, with a
participant bootstrap (5,000). Compensation = gain / slope (A_eye units a virtual face would
need to add to match the physical response).
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

import config as C
from common import log

warnings.filterwarnings("ignore")
R = C.RESULTS_DIR
S6 = importlib.import_module("06_erp_stats")
S8 = importlib.import_module("08_connectivity")
rng = np.random.default_rng(C.SEED)
COMPS = ["N170", "P300", "LPP"]
N_BOOT = 5000


def load():
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    kin = pd.read_csv(R / "stim_kinematics.csv")
    lock = S6.primary_lock()
    av = "expr" if lock == "expression" and (trials.version == "expr").any() else "pipeline"
    t = trials[trials.component.isin(COMPS) & (((trials.dataset == "miko") & (trials.version == "pipeline")) |
                                              ((trials.dataset == "affec") & (trials.version == av)))].copy()
    a_kin = kin[kin.dataset == "affec"]
    for k in ("A", "V", "S", "A_eye"):
        mu, sd = a_kin[k].mean(), a_kin[k].std()
        kin[f"z{k}"] = (kin[k] - mu) / sd
    t = t.merge(kin[["stim", "A", "V", "S", "A_eye", "zA", "zV", "zS", "zA_eye"]], on="stim", how="left")
    return t, kin


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
           for f in sorted((C.DERIV_DIR / "affec").glob("sub-*_pipeline-epo.fif"))]
    ep = mne.concatenate_epochs(eps, verbose=False)
    rows = []
    for stim in ep.metadata.stim.unique():
        e = ep[(ep.metadata.stim == stim).values]
        if len(e) < S8.CONN_N:
            continue
        w, sf = S8.windows(e)
        ims = {k: S8.imag_cross(v, sf) for k, v in w.items()}
        for b in C.BANDS:
            acc = 0
            for _ in range(C.CONN_REPEATS):
                pick = rng.choice(len(e), S8.CONN_N, replace=False)
                post = 0.5 * (S8.dwpli(ims["post1"][b][pick]) + S8.dwpli(ims["post2"][b][pick]))
                acc = acc + (post - S8.dwpli(ims["base"][b][pick]))
            rows.append(dict(stim=stim, band=b, n_trials=len(e), **S8.module_means(acc / C.CONN_REPEATS)))
    df = pd.DataFrame(rows).merge(kin, on="stim")
    df.to_csv(R / "bb_clip_connectivity.csv", index=False, encoding="utf-8-sig")
    out = []
    for b in C.BANDS:
        for mod in ("F-P", "F-T", "P-T"):
            d = df[df.band == b]
            o = smf.ols(f"Q('{mod}') ~ zA + zV + zS + C(emotion)", d).fit()
            for k in ("zA", "zV", "zS"):
                out.append(dict(feature=f"conn_{b}_{mod}", predictor=k, beta=o.params[k], se=o.bse[k], p=o.pvalues[k]))
    return pd.DataFrame(out)


def physical_gain(t, kin):
    """V5: eye-subspace model fitted on AFFEC, evaluated at Miko's A_eye."""
    rows = []
    for comp in COMPS:
        d = t[(t.component == comp) & t.emotion.isin(C.SHARED_EMOTIONS)]
        sm = d.groupby(["dataset", "subject", "emotion", "stim"], as_index=False).agg(amp_z=("amp_z", "mean"),
                                                                                      A_eye=("A_eye", "first"))
        aff, mik = sm[sm.dataset == "affec"].dropna(), sm[sm.dataset == "miko"]
        def gain(aff_s, mik_s):
            o = smf.ols("amp_z ~ A_eye + C(emotion)", aff_s).fit()
            pred = o.predict(mik_s)
            return float((mik_s.amp_z - pred).mean()), float(o.params["A_eye"])
        g, slope = gain(aff, mik)
        sa, sm_ = aff.subject.unique(), mik.subject.unique()
        boots = []
        for _ in range(N_BOOT):
            ba = pd.concat([aff[aff.subject == s] for s in rng.choice(sa, len(sa))])
            bm = pd.concat([mik[mik.subject == s] for s in rng.choice(sm_, len(sm_))])
            boots.append(gain(ba, bm)[0])
        boots = np.array(boots)
        rows.append(dict(feature=comp, slope_A_eye=slope, miko_A_eye_mean=mik.A_eye.mean(),
                         affec_A_eye_mean=aff.A_eye.mean(), physical_gain=g,
                         ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                         p_boot=float(2 * min((boots <= 0).mean(), (boots >= 0).mean())),
                         compensation_A_eye=g / slope if abs(slope) > 1e-9 else np.nan))
    return pd.DataFrame(rows)


def subjective(t):
    a = t[(t.dataset == "affec") & (t.component == "LPP")].drop_duplicates(["subject", "stim", "trial_idx"]).copy()
    a["SCI"] = sci(a)
    res = S6.fit_lmm(a.dropna(subset=["SCI", "zA"]), "zA + zV + zS + C(emotion)", dv="SCI")
    rows = [dict(feature="SCI", predictor=k, beta=res.fe_params[k], se=res.bse[k], p=res.pvalues[k]) for k in ("zA", "zV", "zS")]
    # H4 ratio, participant bootstrap on subject x clip means (OLS)
    lpp = t[(t.dataset == "affec") & (t.component == "LPP")].groupby(["subject", "stim"], as_index=False).agg(
        amp_z=("amp_z", "mean"), zA=("zA", "first"), emotion=("emotion", "first"))
    sc = a.groupby(["subject", "stim"], as_index=False).SCI.mean()
    m = lpp.merge(sc, on=["subject", "stim"]).dropna()
    def ratio(df):
        bn = smf.ols("amp_z ~ zA + C(emotion)", df).fit().params["zA"]
        bs = smf.ols("SCI ~ zA + C(emotion)", df).fit().params["zA"]
        return bn, bs, bn / bs if abs(bs) > 1e-9 else np.nan
    bn, bs, r = ratio(m)
    subs = m.subject.unique()
    boot = np.array([ratio(pd.concat([m[m.subject == s] for s in rng.choice(subs, len(subs))]))[2] for _ in range(1000)])
    h4 = pd.DataFrame([dict(dataset="affec", beta_neural_LPP=bn, beta_subjective=bs, ratio=r,
                            ci_lo=np.nanpercentile(boot, 2.5), ci_hi=np.nanpercentile(boot, 97.5)),
                       dict(dataset="miko", beta_neural_LPP=np.nan, beta_subjective=np.nan, ratio=np.nan,
                            ci_lo=np.nan, ci_hi=np.nan)])
    return pd.DataFrame(rows), h4


def main():
    t, kin = load()
    erp, loso = erp_slopes(t)
    conn = clip_connectivity(kin)
    subj, h4 = subjective(t)
    slopes = pd.concat([erp, conn, subj], ignore_index=True)
    slopes["p_fdr"] = multipletests(slopes.p, method="fdr_bh")[1]
    gain = physical_gain(t, kin)
    markers = slopes[slopes.predictor == "zA"].merge(loso, on="feature", how="left").merge(
        gain[["feature", "physical_gain", "ci_lo", "ci_hi", "p_boot"]], on="feature", how="left")
    markers["is_marker"] = (markers.p_fdr < .05) & (markers.loso_sign_stability.fillna(0) >= .9) & (markers.p_boot < .05)
    out = dict(slopes=slopes, loso=loso, physical_gain=gain, markers=markers, h4=h4)
    for k, v in out.items():
        v.to_csv(R / f"bb_{k}.csv", index=False, encoding="utf-8-sig")
    with open(R / "bb_summary.md", "w", encoding="utf-8") as fh:
        fh.write("# 路线 3：行为–脑电关联（自动生成）\n\n")
        for k, v in out.items():
            fh.write(f"## {k}\n\n{v.round(4).to_markdown(index=False)}\n\n")
    log(markers.round(3).to_string())


if __name__ == "__main__":
    main()
