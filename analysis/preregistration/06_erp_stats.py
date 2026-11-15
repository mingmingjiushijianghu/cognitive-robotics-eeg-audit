"""Step 6 (design §3.2-3.5, V1-V4, V9): ERP statistics, route 1.

  A. Single-trial LMMs (shared emotions angry/happy/sad), DV = amp_z (baseline-SD units, V4)
       amp_z ~ embodiment * emotion [* hemisphere]  + crossed random intercepts subject, stim
     LR tests (ML fits) for the embodiment main effect and the embodiment x emotion interaction.
  B. Within-emotion contrasts (emotion - subject mean over the 3 emotions) ~ embodiment * emotion
  C. Latency (50 % fractional area, subject averages) ~ embodiment * emotion + (1 | subject)
  D. N170 equivalence (H1): Welch TOST (+-0.3 pooled SD) and BF01 on subject means
  E. Matched subsampling: 1,000 draws of 10 AFFEC participants matched to Miko (22-29 y, 9 m / 1 f)
  F. §3.3 validation in AFFEC: 16-ch unified ROI vs 62-ch montage / standard N170 sites
  G. TANOVA on GFP-normalised component topographies (5,000 permutations)
  H. Sensitivity: Miko author-cleaned data, Cz ROIs, video- vs expression-locked AFFEC,
     speech-onset covariate (AFFEC only)
FDR (BH, q = .05) over the confirmatory family: 3 components x {amplitude, latency} x
{embodiment, embodiment x emotion}.

Outputs results/erp_stats_*.csv, results/erp_stats_summary.md
"""
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import mne

import config as C
from common import log

warnings.filterwarnings("ignore")
R = C.RESULTS_DIR
rng = np.random.default_rng(C.SEED)
CONFIRM = ["N170", "P300", "LPP"]


def primary_lock():
    p = R / "expr_lock_decision.json"
    return json.loads(p.read_text())["primary_lock"] if p.exists() else "video"


def select(trials, affec_version, miko_version="pipeline", comps=None):
    t = trials[((trials.dataset == "miko") & (trials.version == miko_version)) |
               ((trials.dataset == "affec") & (trials.version == affec_version))]
    t = t[t.emotion.isin(C.SHARED_EMOTIONS)].copy()
    if comps is not None:
        t = t[t.component.isin(comps)]
    t["embodiment"] = np.where(t.dataset == "miko", "physical", "virtual")
    t["stim"] = t.stim.astype(str)
    return t


def fit_lmm(df, fixed, dv="amp_z", reml=True):
    """Crossed random intercepts for subject and stimulus (single group trick)."""
    df = df.assign(_g=1)
    vc = {"subject": "0 + C(subject)", "stim": "0 + C(stim)"}
    m = smf.mixedlm(f"{dv} ~ {fixed}", df, groups="_g", vc_formula=vc, re_formula="0")
    try:
        return m.fit(reml=reml, method=["lbfgs", "powell"])
    except Exception:                                   # simplify: subject intercept only
        return smf.mixedlm(f"{dv} ~ {fixed}", df, groups="subject").fit(reml=reml)


def lr_test(df, full, reduced, dv="amp_z"):
    f1, f0 = fit_lmm(df, full, dv, reml=False), fit_lmm(df, reduced, dv, reml=False)
    stat = 2 * (f1.llf - f0.llf)
    dfree = len(f1.fe_params) - len(f0.fe_params)
    return float(stat), int(dfree), float(stats.chi2.sf(stat, dfree))


def coef_table(res, label):
    ci = res.conf_int()
    out = pd.DataFrame(dict(term=res.fe_params.index, beta=res.fe_params.values,
                            ci_lo=ci.loc[res.fe_params.index, 0].values,
                            ci_hi=ci.loc[res.fe_params.index, 1].values,
                            p=res.pvalues[res.fe_params.index].values))
    out.insert(0, "model", label)
    return out


def amplitude_models(t, tag):
    coefs, tests = [], []
    for comp in sorted(t.component.unique()):
        d = t[t.component == comp].copy()
        d["amp_std"] = d.amp_z / d.amp_z.std()                    # standardised beta
        hemi = d.roi.nunique() > 1
        base = "C(embodiment, Treatment('virtual')) * C(emotion)" + (" * C(roi)" if hemi else "")
        res = fit_lmm(d, base, dv="amp_std")
        coefs.append(coef_table(res, f"{tag}:{comp}"))
        emb = "C(embodiment, Treatment('virtual'))"
        no_int = f"{emb} + C(emotion)" + (" + C(roi)" if hemi else "")
        no_emb = "C(emotion)" + (" + C(roi)" if hemi else "")
        s1, d1, p1 = lr_test(d, no_int, no_emb, dv="amp_std")
        s2, d2, p2 = lr_test(d, base, no_int, dv="amp_std")
        tests += [dict(analysis=tag, component=comp, measure="amplitude", effect="embodiment", chi2=s1, df=d1, p=p1),
                  dict(analysis=tag, component=comp, measure="amplitude", effect="embodiment x emotion", chi2=s2, df=d2, p=p2)]
        log(tag, comp, f"emb p={p1:.4f}  int p={p2:.4f}")
    return pd.concat(coefs), pd.DataFrame(tests)


def contrast_models(t, tag):
    d = t.copy()
    d["contrast"] = d.amp_z - d.groupby(["subject", "component", "roi"]).amp_z.transform("mean")
    out = []
    for comp in sorted(d.component.unique()):
        x = d[d.component == comp]
        res = fit_lmm(x, "C(embodiment, Treatment('virtual')) * C(emotion)", dv="contrast")
        out.append(coef_table(res, f"{tag}:{comp}:contrast"))
    return pd.concat(out)


def latency_models(lat, affec_version):
    lat = lat[(((lat.dataset == "miko") & (lat.version == "pipeline")) |
               ((lat.dataset == "affec") & (lat.version == affec_version))) &
              lat.emotion.isin(C.SHARED_EMOTIONS) & lat.component.isin(CONFIRM)].dropna(subset=["latency"]).copy()
    lat["embodiment"] = np.where(lat.dataset == "miko", "physical", "virtual")
    lat["latency_ms"] = lat.latency * 1000
    tests = []
    for comp in CONFIRM:
        d = lat[lat.component == comp].groupby(["subject", "embodiment", "emotion"], as_index=False).latency_ms.mean()
        f = lambda fx: smf.mixedlm(f"latency_ms ~ {fx}", d, groups="subject").fit(reml=False)
        full, noint, noemb = f("embodiment * emotion"), f("embodiment + emotion"), f("emotion")
        for eff, a, b in (("embodiment", noint, noemb), ("embodiment x emotion", full, noint)):
            s = 2 * (a.llf - b.llf); k = len(a.fe_params) - len(b.fe_params)
            tests.append(dict(analysis="primary", component=comp, measure="latency", effect=eff,
                              chi2=s, df=k, p=stats.chi2.sf(s, k)))
    return pd.DataFrame(tests)


def subject_means(t, comp):
    return t[t.component == comp].groupby(["embodiment", "subject"]).amp_z.mean()


def equivalence(t):
    rows = []
    for comp in CONFIRM:
        s = subject_means(t, comp)
        x, y = s["physical"].values, s["virtual"].values
        sd = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        tost = pg.tost(x, y, bound=0.3 * sd, paired=False)
        tt = pg.ttest(x, y, correction=True)
        bf10 = float(pg.bayesfactor_ttest(float(tt["T"].iloc[0]), len(x), len(y)))
        rows.append(dict(component=comp, n_phys=len(x), n_virt=len(y), mean_phys=x.mean(), mean_virt=y.mean(),
                         hedges_g=float(pg.compute_effsize(x, y, eftype="hedges")),
                         t=float(tt["T"].iloc[0]), p=float(tt["p-val"].iloc[0]),
                         tost_bound=0.3 * sd, tost_p=float(tost["pval"].iloc[0]), BF01=1 / bf10))
    return pd.DataFrame(rows)


def matched_subsampling(t, n_draw=1000):
    part = pd.read_csv(C.AFFEC_DIR / "participants.tsv", sep="\t")
    part.columns = [c.strip() for c in part.columns]
    part["subject"] = part.participant_id.str.replace("sub-", "", regex=False)
    have = set(t[t.embodiment == "virtual"].subject)
    pool = part[part.subject.isin(have)]
    age_ok = pool[(pool.age >= 22) & (pool.age <= 29)]
    males, females = age_ok[age_ok.gender == "m"].subject.values, age_ok[age_ok.gender == "f"].subject.values
    if len(males) < 9 or len(females) < 1:                 # relax age window if needed
        males, females = pool[pool.gender == "m"].subject.values, pool[pool.gender == "f"].subject.values
    rows = []
    for comp in CONFIRM:
        s = subject_means(t, comp)
        x = s["physical"].values
        g = []
        for _ in range(n_draw):
            pick = list(rng.choice(males, 9, replace=False)) + list(rng.choice(females, 1, replace=False))
            y = s["virtual"].reindex(pick).dropna().values
            g.append(float(pg.compute_effsize(x, y, eftype="hedges")))
        g = np.array(g)
        rows.append(dict(component=comp, n_pool_m=len(males), n_pool_f=len(females), g_median=np.median(g),
                         g_p2_5=np.percentile(g, 2.5), g_p97_5=np.percentile(g, 97.5), frac_positive=(g > 0).mean()))
    return pd.DataFrame(rows)


def validation_16ch(trials):
    a = trials[trials.dataset == "affec"]
    rows = []
    for comp, ref_version, ref_comp in [("N170", "full62", "N170_std"), ("N170", "full62", "N170"),
                                        ("P300", "full62", "P300"), ("LPP", "full62", "LPP")]:
        u = a[(a.version == "pipeline") & (a.component == comp)].groupby(["subject", "emotion"]).amp_z.mean()
        v = a[(a.version == ref_version) & (a.component == ref_comp)].groupby(["subject", "emotion"]).amp_z.mean()
        j = pd.concat([u.rename("u"), v.rename("v")], axis=1).dropna().reset_index()
        if j.empty:
            continue
        subj = j.groupby("subject")[["u", "v"]].mean()
        emo = j.groupby("emotion")[["u", "v"]].mean()
        emo_c = emo - emo.mean()
        rows.append(dict(component=comp, reference=f"{ref_version}:{ref_comp}", n_subjects=len(subj),
                         r_subjects=subj.u.corr(subj.v), r_emotion_profile=emo.u.corr(emo.v),
                         same_sign_emotion_effects=float((np.sign(emo_c.u) == np.sign(emo_c.v)).mean())))
    return pd.DataFrame(rows)


def tanova(affec_version, n_perm=C.N_PERM):
    """Group difference of GFP-normalised topographies per component window (subject averages)."""
    maps = {}
    for ds, ver in (("miko", "pipeline"), ("affec", affec_version)):
        for f in sorted((C.DERIV_DIR / "erp").glob(f"{ds}_{ver}_*-ave.fif")):
            ev = [e for e in mne.read_evokeds(f, verbose=False) if e.comment in C.SHARED_EMOTIONS]
            if ev:
                maps[(ds, f.stem)] = mne.combine_evoked(ev, "equal").pick(C.COMMON_CHANNELS)
    rows = []
    for comp in CONFIRM:
        t0, t1 = C.COMPONENTS[comp][:2]
        X, lab = [], []
        for (ds, _), e in maps.items():
            m = e.copy().crop(t0, t1).data.mean(axis=1)
            m = m - m.mean()
            X.append(m / np.sqrt(np.mean(m ** 2)))
            lab.append(ds == "miko")
        X, lab = np.array(X), np.array(lab)

        def diss(l):
            a, b = X[l].mean(0), X[~l].mean(0)
            a, b = a / np.sqrt(np.mean(a ** 2)), b / np.sqrt(np.mean(b ** 2))
            return np.sqrt(np.mean((a - b) ** 2))
        obs = diss(lab)
        null = np.array([diss(rng.permutation(lab)) for _ in range(n_perm)])
        rows.append(dict(component=comp, DISS=obs, p_perm=(1 + (null >= obs).sum()) / (n_perm + 1)))
    return pd.DataFrame(rows)


def main():
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    lat = pd.read_csv(R / "erp_subject_latency.csv")
    lock = primary_lock()
    av = "expr" if lock == "expression" and (trials.version == "expr").any() else "pipeline"
    alt = "pipeline" if av == "expr" else "expr"
    log("primary AFFEC version:", av)

    t = select(trials, av, comps=CONFIRM)
    coefs, tests = amplitude_models(t, "primary")
    tests = pd.concat([tests, latency_models(lat, av)], ignore_index=True)
    tests["p_fdr"] = multipletests(tests.p, method="fdr_bh")[1]
    contrasts = contrast_models(t, "primary")

    sens_c, sens_t = [], []
    t_late = select(trials, av, comps=CONFIRM)
    t_late = t_late[(t_late.embodiment == "physical") | (t_late.speech_onset >= 0.2)]   # V1: speech-free N170 window
    for tag, tt in [("miko_author", select(trials, av, "author", CONFIRM)),
                    ("affec_speech_after_200ms", t_late),
                    ("cz_rois", select(trials, av, comps=["P300_Cz", "LPP_Cz"])),
                    (f"affec_{alt}_locked", select(trials, alt, comps=CONFIRM))]:
        if tt.embodiment.nunique() == 2:
            c_, t_ = amplitude_models(tt, tag)
            sens_c.append(c_); sens_t.append(t_)
    a = trials[(trials.dataset == "affec") & (trials.version == av) & trials.component.isin(CONFIRM)].dropna(subset=["speech_onset"])
    speech = []
    for comp in CONFIRM:
        d = a[a.component == comp]
        if len(d):
            res = fit_lmm(d, "C(emotion) + speech_onset" + (" + C(roi)" if d.roi.nunique() > 1 else ""))
            speech.append(coef_table(res, f"affec_speech:{comp}"))

    eq = equivalence(t)
    matched = matched_subsampling(t)
    val = validation_16ch(trials)
    tan = tanova(av)

    out = dict(lmm_coefs=pd.concat([coefs] + sens_c + speech), tests=pd.concat([tests] + sens_t),
               contrasts=contrasts, equivalence=eq, matched=matched, validation16=val, tanova=tan)
    for k, v in out.items():
        v.to_csv(R / f"erp_stats_{k}.csv", index=False, encoding="utf-8-sig")
    with open(R / "erp_stats_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# 路线 1：ERP 统计结果（自动生成）\n\nAFFEC 主分析版本：`{av}`（锁定规则见 expr_lock_decision.json）。DV = 基线标准化波幅（V4）。\n\n")
        for k, v in out.items():
            fh.write(f"## {k}\n\n{v.round(4).to_markdown(index=False)}\n\n")
    log("done")


if __name__ == "__main__":
    main()
