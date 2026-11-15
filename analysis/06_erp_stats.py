"""Step 6 (design §3.2-3.5, V1-V4, V9): ERP statistics, route 1.

  A. LMMs on participant x emotion cell means (V13; shared emotions angry/happy/sad),
     DV = amp_z (baseline-SD units, V4): amp ~ embodiment * emotion [* hemisphere] + (1 | subject).
     LR tests (ML fits) for embodiment and embodiment x emotion; participant-level permutation
     p (1,000) for the embodiment effect. Single-trial crossed models are used only inside AFFEC.
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

import config as cfg
from common import log

warnings.filterwarnings("ignore")
R = cfg.RESULTS_DIR
rng = np.random.default_rng(cfg.SEED)
CONFIRM = ["N170", "P300", "LPP"]


def primary_lock():
    p = R / "expr_lock_decision.json"
    return json.loads(p.read_text())["primary_lock"] if p.exists() else "video"


def select(trials, affec_version, miko_version="pipeline", comps=None, min_trials=cfg.MIN_TRIALS_SHARED):
    trials = trials[trials.min_shared_trials >= min_trials]               # V14 inclusion rule
    t = trials[((trials.dataset == "miko") & (trials.version == miko_version)) |
               ((trials.dataset == "affec") & (trials.version == affec_version))]
    t = t[t.emotion.isin(cfg.SHARED_EMOTIONS)].copy()
    if comps is not None:
        t = t[t.component.isin(comps)]
    t["embodiment"] = np.where(t.dataset == "miko", "physical", "virtual")
    t["stim"] = t.stim.astype(str)
    return t


def fit_lmm(df, fixed, dv="amp_z", reml=True):
    """Crossed random intercepts for subject and stimulus (single group trick)."""
    df = df.reset_index(drop=True).assign(_g=1)
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


def cell_means(d, dv):
    keys = ["subject", "embodiment", "emotion"] + (["roi"] if d.roi.nunique() > 1 else [])
    return d.groupby(keys, as_index=False)[dv].mean()


def cell_lr(c, full, reduced, dv):
    f = lambda fx: smf.mixedlm(f"{dv} ~ {fx}", c, groups="subject").fit(reml=False)
    a, b = f(full), f(reduced)
    stat = 2 * (a.llf - b.llf)
    k = len(a.fe_params) - len(b.fe_params)
    return float(stat), int(k), float(stats.chi2.sf(stat, k))


def perm_p(c, full, reduced, dv, n=1000):
    """Participant-level label permutation for the embodiment effect (V13)."""
    obs = cell_lr(c, full, reduced, dv)[0]
    lab = c.groupby("subject").embodiment.first()
    null = []
    for _ in range(n):
        m = dict(zip(lab.index, rng.permutation(lab.values)))
        null.append(cell_lr(c.assign(embodiment=c.subject.map(m)), full, reduced, dv)[0])
    return (1 + np.sum(np.array(null) >= obs)) / (n + 1)


def amplitude_models(t, tag, n_perm=1000):
    """V13: embodiment effects on participant x emotion cell means, (1 | subject).
    (Single-trial crossed models are degenerate here: Miko's 4 stimuli = its 4 emotions.)"""
    coefs, tests = [], []
    for comp in sorted(t.component.unique()):
        d = t[t.component == comp].copy()
        d["amp_std"] = d.amp_z / d.amp_z.std()                    # standardised beta
        c = cell_means(d, "amp_std")
        roi = " + C(roi)" if "roi" in c else ""
        emb = "C(embodiment, Treatment('virtual'))"
        full = f"{emb} * C(emotion)" + (" * C(roi)" if roi else "")
        res = smf.mixedlm(f"amp_std ~ {full}", c, groups="subject").fit(reml=True)
        coefs.append(coef_table(res, f"{tag}:{comp}"))
        with_int, no_int, no_emb = f"{emb} * C(emotion){roi}", f"{emb} + C(emotion){roi}", f"C(emotion){roi}"
        s1, d1, p1 = cell_lr(c, no_int, no_emb, "amp_std")
        s2, d2, p2 = cell_lr(c, with_int, no_int, "amp_std")
        pp1 = perm_p(c, no_int, no_emb, "amp_std", n_perm) if n_perm else np.nan
        tests += [dict(analysis=tag, component=comp, measure="amplitude", effect="embodiment", chi2=s1, df=d1, p=p1, p_perm=pp1),
                  dict(analysis=tag, component=comp, measure="amplitude", effect="embodiment x emotion", chi2=s2, df=d2, p=p2)]
        log(tag, comp, f"emb p={p1:.4f} (perm {pp1:.4f})  int p={p2:.4f}")
    return pd.concat(coefs), pd.DataFrame(tests)


def contrast_models(t, tag):
    d = t.copy()
    d["contrast"] = d.amp_z - d.groupby(["subject", "component", "roi"]).amp_z.transform("mean")
    out = []
    for comp in sorted(d.component.unique()):
        x = d[d.component == comp]
        c = cell_means(x, "contrast")
        res = smf.mixedlm("contrast ~ C(embodiment, Treatment('virtual')) * C(emotion)" + (" + C(roi)" if "roi" in c else ""),
                          c, groups="subject").fit(reml=True)
        out.append(coef_table(res, f"{tag}:{comp}:contrast"))
    return pd.concat(out)


def latency_models(lat, affec_version):
    lat = lat[(((lat.dataset == "miko") & (lat.version == "pipeline")) |
               ((lat.dataset == "affec") & (lat.version == affec_version))) &
              lat.emotion.isin(cfg.SHARED_EMOTIONS) & lat.component.isin(CONFIRM)].dropna(subset=["latency"]).copy()
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
                         t=float(tt["T"].iloc[0]), p=float(tt["p_val"].iloc[0]),
                         tost_bound=0.3 * sd, tost_p=float(tost["pval"].iloc[0]), BF01=1 / bf10))
    return pd.DataFrame(rows)


def matched_subsampling(t, n_draw=1000):
    part = pd.read_csv(cfg.AFFEC_DIR / "participants.tsv", sep="\t")
    part.columns = [c.strip().strip('"').strip() for c in part.columns]     # header is literally "age "
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
    """§3.3: participants included by the V14 rule; subject x emotion means, then subject means.
    The participant-level table is saved for Fig. 3 so text and figure use the same values."""
    a = trials[trials.dataset == "affec"]
    incl = set(a[(a.version == "pipeline") & (a.min_shared_trials >= cfg.MIN_TRIALS_SHARED)].subject)
    a = a[a.subject.isin(incl)]
    rows, subj_tables = [], []
    for comp, ref_version, ref_comp in [("N170", "full62", "N170_std"), ("N170", "full62", "N170"),
                                        ("P300", "full62", "P300"), ("LPP", "full62", "LPP")]:
        u = a[(a.version == "pipeline") & (a.component == comp)].groupby(["subject", "emotion"]).amp_z.mean()
        v = a[(a.version == ref_version) & (a.component == ref_comp)].groupby(["subject", "emotion"]).amp_z.mean()
        j = pd.concat([u.rename("u"), v.rename("v")], axis=1).dropna().reset_index()
        if j.empty:
            continue
        subj = j.groupby("subject")[["u", "v"]].mean()
        subj_tables.append(subj.reset_index().assign(component=comp, reference=f"{ref_version}:{ref_comp}"))
        emo = j.groupby("emotion")[["u", "v"]].mean()
        emo_c = emo - emo.mean()
        rows.append(dict(component=comp, reference=f"{ref_version}:{ref_comp}", n_subjects=len(subj),
                         r_subjects=subj.u.corr(subj.v), r_emotion_profile=emo.u.corr(emo.v),
                         same_sign_emotion_effects=float((np.sign(emo_c.u) == np.sign(emo_c.v)).mean())))
    pd.concat(subj_tables).rename(columns={"u": "amp_16ch", "v": "amp_full"}).to_csv(
        R / "erp_validation16_subjects.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def tanova(affec_version, n_perm=cfg.N_PERM):
    """Group difference of GFP-normalised topographies per component window (subject averages)."""
    maps = {}
    for ds, ver in (("miko", "pipeline"), ("affec", affec_version)):
        for f in sorted((cfg.DERIV_DIR / "erp").glob(f"{ds}_{ver}_*-ave.fif")):
            ev = [e for e in mne.read_evokeds(f, verbose=False) if e.comment in cfg.SHARED_EMOTIONS]
            if ev:
                maps[(ds, f.stem)] = mne.combine_evoked(ev, "equal").pick(cfg.COMMON_CHANNELS)
    rows = []
    for comp in CONFIRM:
        t0, t1 = cfg.COMPONENTS[comp][:2]
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
    for tag, tt in [(f"min_trials_{k}", select(trials, av, comps=CONFIRM, min_trials=k)) for k in cfg.MIN_TRIALS_SENS] + [
                    ("miko_author", select(trials, av, "author", CONFIRM)),
                    ("affec_speech_after_200ms", t_late),
                    ("cz_rois", select(trials, av, comps=["P300_Cz", "LPP_Cz"])),
                    (f"affec_{alt}_locked", select(trials, alt, comps=CONFIRM))]:
        if tt.embodiment.nunique() == 2:
            c_, t_ = amplitude_models(tt, tag, n_perm=0)
            sens_c.append(c_); sens_t.append(t_)
    a = trials[(trials.dataset == "affec") & (trials.version == av) & trials.component.isin(CONFIRM) &
               (trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED)].dropna(subset=["speech_onset"])
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
    # §3.3 / V15: a component supports a cross-dataset test only if the 16-ch measure tracks the
    # full-montage reference across participants (r >= .7); N170 is checked against P7/PO7/P8/PO8.
    ref = {"N170": "full62:N170_std", "P300": "full62:P300", "LPP": "full62:LPP"}
    verdict = []
    for comp, rr in ref.items():
        row = val[(val.component == comp) & (val.reference == rr)]
        r = float(row.r_subjects.iloc[0]) if len(row) else np.nan
        verdict.append(dict(component=comp, reference=rr, r_subjects=r, valid_16ch=bool(r >= 0.7)))
    verdict = pd.DataFrame(verdict)
    verdict.to_csv(R / "erp_stats_validity.csv", index=False, encoding="utf-8-sig")
    with open(R / "erp_stats_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# 路线 1：ERP 统计结果（自动生成）\n\nAFFEC 主分析版本：`{av}`（锁定规则见 expr_lock_decision.json）。DV = 基线标准化波幅（V4）。\n\n")
        fh.write("## §3.3 16 导有效性判定（V15）\n\n" + verdict.round(3).to_markdown(index=False) +
                 "\n\n`valid_16ch = False` 的成分不能做跨数据集推断，只作描述。\n\n")
        for k, v in out.items():
            fh.write(f"## {k}\n\n{v.round(4).to_markdown(index=False)}\n\n")
    log("done")


if __name__ == "__main__":
    main()
