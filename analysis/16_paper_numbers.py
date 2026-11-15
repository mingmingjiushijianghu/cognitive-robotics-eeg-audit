"""Step 16: every number quoted in the manuscript, written as LaTeX macros.

Reads the result tables and writes ../submission/manuscript/numbers.tex. The manuscript uses only
these macros for empirical values, so a re-run of any analysis step updates the text and no
number is transcribed by hand. A missing input leaves the macro as "??" (visible in the PDF).
"""
import json
import numpy as np
import pandas as pd

import config as cfg
from common import log

R = cfg.RESULTS_DIR
OUT = cfg.ANALYSIS_DIR.parent / "submission" / "manuscript" / "numbers.tex"
M = {}


def put(name, value, fmt="{}"):
    assert name.isalpha(), name
    try:
        M[name] = fmt.format(value) if not isinstance(value, str) else value
    except Exception:
        M[name] = "??"


def p_fmt(p):
    if p is None or not np.isfinite(p):
        return "??"
    return "< .001" if p < .001 else ("= " + f"{p:.3f}".lstrip("0") if p < .1 else "= " + f"{p:.2f}".lstrip("0"))


def r_fmt(r):
    return ("\ensuremath{-}" if r < 0 else "") + f"{abs(r):.2f}".lstrip("0")


def safe(fn):
    try:
        fn()
    except Exception as e:                 # leave affected macros as ?? and keep going
        log("WARN", fn.__name__, repr(e))


def data_quality():
    mq = pd.read_csv(R / "miko_preprocessing_qc.csv")
    aq = pd.read_csv(R / "affec_preprocessing_qc.csv")
    put("NMiko", len(mq))
    put("MikoTrialsMin", int(mq.pipeline_n_final.min())); put("MikoTrialsMax", int(mq.pipeline_n_final.max()))
    eeg = aq[aq.n_final.notna()]
    put("NAffecEEG", len(eeg))
    put("AffecTrialsMedian", int(eeg.n_final.median()))
    incl = eeg[eeg[["n_angry", "n_happy", "n_sad"]].min(axis=1) >= cfg.MIN_TRIALS_SHARED]
    put("NAffecIncl", len(incl))
    put("NAffecMissingRuns", int(eeg.missing_runs.notna().sum()) if "missing_runs" in eeg else 0)
    put("AffecBadsMedian", int(eeg.n_bads.median()))
    stim = pd.read_csv(R / "affec_stim_features.csv")
    put("NClips", len(stim))


def timing():
    s = pd.read_csv(R / "event_timing_summary.csv").set_index("dataset")
    d = pd.read_csv(R / "event_timing_check.csv")
    for ds, tag in (("miko", "Miko"), ("miko_author", "MikoAuth"), ("affec", "Affec")):
        g = d[d.dataset == ds]
        put(f"Tim{tag}Sig", int((g.itc_p < .05).sum())); put(f"Tim{tag}N", len(g))
        put(f"Tim{tag}ItcReal", s.loc[ds, "itc_real"], "{:.3f}"); put(f"Tim{tag}ItcSurr", s.loc[ds, "itc_surr"], "{:.3f}")
        put(f"Tim{tag}PMin", f"{g.itc_p.min():.2f}".lstrip("0")); put(f"Tim{tag}PMax", f"{g.itc_p.max():.2f}".lstrip("0"))
    from scipy import stats as st
    for ds, tag in (("miko", "Miko"), ("miko_author", "MikoAuth")):
        g = d[d.dataset == ds]
        k = int((g.itc_p < .05).sum())
        put(f"Tim{tag}Binom", f"{st.binom.sf(k - 1, len(g), .05):.3f}".lstrip("0") if k else "1")
        put(f"Tim{tag}GfpSig", int((g.gfp_p < .05).sum()))
    put("TimAffecPct", 100 * (d[d.dataset == "affec"].itc_p < .05).mean(), "{:.0f}")
    iv = pd.read_csv(R / "miko_marker_intervals.csv")
    so = iv[iv.order == "stored order"].interval_s
    srt = iv[iv.order == "sorted"].interval_s
    put("MikoNegIntervalPct", 100 * (so < 0).mean(), "{:.0f}")
    put("MikoSortedMedian", srt.median(), "{:.1f}")
    put("MikoSortedQlo", srt.quantile(.1), "{:.1f}"); put("MikoSortedQhi", srt.quantile(.9), "{:.1f}")
    put("MikoTailMedian", iv[iv.order == "tail without markers"].interval_s.median(), "{:.0f}")


def stimulus():
    st = pd.read_csv(R / "affec_stim_features.csv")
    dec = json.loads((R / "expr_lock_decision.json").read_text())
    put("ExprOnsetMedian", 1000 * st.expr_onset.median(), "{:.0f}")
    put("ExprOnsetLatePct", 100 * dec["frac_clips_onset_after_300ms"], "{:.0f}")
    put("SpeechOnsetMedian", 1000 * st.speech_onset.median(), "{:.0f}")
    put("SpeechEarlyPct", 100 * (st.speech_onset < 0.2).mean(), "{:.0f}")


def montage():
    v = pd.read_csv(R / "erp_stats_validation16.csv")
    get = lambda c, ref: v[(v.component == c) & (v.reference == ref)].r_subjects.iloc[0]
    put("RNstd", r_fmt(get("N170", "full62:N170_std"))); put("RPthree", r_fmt(get("P300", "full62:P300")))
    put("RLPP", r_fmt(get("LPP", "full62:LPP")))
    put("NValidation", int(v.n_subjects.iloc[0]))


def kinematics():
    t = pd.read_csv(R / "x2c_training_log.csv")
    put("XMAE", t.val_mae.min(), "{:.4f}")
    v = pd.read_csv(R / "x2c_crossdomain_validity.csv")
    pr = v[v.test == "a priori pair"].set_index("component").r
    smile = pr[pr.index.str.contains("mouthSmile")]
    put("XSmileLo", r_fmt(smile.min())); put("XSmileHi", r_fmt(smile.max()))
    put("XJaw", r_fmt(pr[pr.index.str.contains("jawOpen")].iloc[0]))
    brow = pr[pr.index.str.contains("browInnerUp|eyeWide")]
    put("XBrowLo", r_fmt(brow.min())); put("XBrowHi", r_fmt(brow.max()))
    put("XCCA", r_fmt(v[v.test == "CCA out-of-sample"].r.iloc[0]))
    mk = pd.read_csv(R / "miko_eye_geometry.csv").set_index("emotion").eye_logratio_peak
    for e in ("angry", "happy", "sad", "surprised"):
        put("Geo" + e.capitalize(), f"{mk[e]:+.2f}".replace("-", "\ensuremath{-}"))
    g = pd.read_csv(R / "eye_geometry_kinematics.csv")
    a = g[g.dataset == "affec"].eye_logratio_max
    put("GeoAffecLo", a.min(), "{:.2f}"); put("GeoAffecHi", a.max(), "{:.2f}")
    k = pd.read_csv(R / "stim_kinematics.csv")
    km = k[k.dataset == "miko"].set_index("emotion").A_eye
    put("XMikoSurpriseRank", int(km.rank().loc["surprised"]))


def cross_dataset():
    t = pd.read_csv(R / "erp_stats_tests.csv")
    g = lambda a, c, e: t[(t.analysis == a) & (t.component == c) & (t.measure == "amplitude") & (t.effect == e)].p.iloc[0]
    for c, tag in (("N170", "Nsev"), ("P300", "Pthree"), ("LPP", "LPP")):
        put(f"P{tag}Emb", p_fmt(g("primary", c, "embodiment")))
    eq = pd.read_csv(R / "erp_stats_equivalence.csv").set_index("component")
    for c, tag in (("P300", "Pthree"), ("LPP", "LPP")):
        put(f"G{tag}", r_fmt(eq.loc[c, "hedges_g"])); put(f"BF{tag}", eq.loc[c, "BF01"], "{:.2f}")
    rs = pd.read_csv(R / "robust_summary.csv").set_index("component")
    for c, tag in (("P300", "Pthree"), ("LPP", "LPP")):
        put(f"MV{tag}P", p_fmt(rs.loc[c, "p_joint_perm"])); put(f"MV{tag}G", r_fmt(rs.loc[c, "median_g"]))
        put(f"MV{tag}N", int(rs.loc[c, "n_specs"]))
    sp = pd.read_csv(R / "robust_spec_curve.csv")
    for c, tag in (("P300", "Pthree"), ("LPP", "LPP")):
        for lock, lt in (("video", "Vid"), ("expression", "Expr")):
            put(f"MV{tag}{lt}G", r_fmt(sp[(sp.component == c) & (sp.affec_lock == lock)].hedges_g.median()))
    cl = pd.read_csv(R / "robust_clusters.csv")
    for lock, lt in (("video", "Vid"), ("expression", "Expr")):
        d = cl[(cl.affec_lock == lock)].sort_values("p")
        put(f"Cl{lt}P", p_fmt(d.p.iloc[0])); put(f"Cl{lt}Start", 1000 * d.t_start.iloc[0], "{:.0f}")
        put(f"Cl{lt}End", 1000 * d.t_end.iloc[0], "{:.0f}")
        put(f"Cl{lt}Nch", len(str(d.channels.iloc[0]).split(",")))
    cs = pd.read_csv(R / "conn_stats.csv")
    e = cs[(cs.effect == "embodiment") & (cs.metric == "geff") & (cs.band == "beta")].set_index("analysis")
    put("BetaGeffP", p_fmt(e.loc["primary", "p"])); put("BetaGeffFDR", p_fmt(e.loc["primary", "p_fdr"]))
    put("BetaGeffAuthP", p_fmt(e.loc["miko_author", "p"]))
    ed = pd.read_csv(R / "eed_summary.csv")
    put("EED", ed.EED.iloc[0], "{:.2f}"); put("EEDP", p_fmt(ed.p_perm.iloc[0]))
    put("EEDNoBetaP", p_fmt(ed[ed.analysis.str.contains("beta")].p_perm.iloc[0]))


def calibration():
    c = pd.read_csv(R / "model_calibration_summary.csv")
    for model, mt in (("single_trial_crossed", "Cross"), ("cell_mean", "Cell")):
        for comp, ct in (("P300", "Pthree"), ("LPP", "LPP")):
            r = c[(c.model == model) & (c.component == comp)].iloc[0]
            put(f"Cal{mt}{ct}", 100 * r.false_positive_rate, "{:.0f}")
            put(f"Cal{mt}{ct}Lo", 100 * r.ci_lo, "{:.0f}"); put(f"Cal{mt}{ct}Hi", 100 * r.ci_hi, "{:.0f}")
    put("CalN", int(c.n_sim.iloc[0]))


def affec_internal():
    h = pd.read_csv(R / "bb_h4.csv")
    r = h[(h.kinematics == "bs") & (h.dataset == "affec")].iloc[0]
    put("SlopeSCI", r.beta_subjective, "{:.2f}"); put("SlopeSCILo", r.bs_ci_lo, "{:.2f}"); put("SlopeSCIHi", r.bs_ci_hi, "{:.2f}")
    f = lambda x: f"{x:.3f}".replace("-", "\ensuremath{-}")
    put("SlopeLPP", f(r.beta_neural_LPP)); put("SlopeLPPLo", f(r.bn_ci_lo)); put("SlopeLPPHi", f(r.bn_ci_hi))
    for k, tag in (("lmm_sci", "LmmSCI"), ("lmm_lpp", "LmmLPP")):
        put(tag, f"{r[k + '_beta']:.3f}".replace("-", "\ensuremath{-}"))
        put(tag + "Lo", f"{r[k + '_lo']:.3f}".replace("-", "\ensuremath{-}")); put(tag + "Hi", f"{r[k + '_hi']:.3f}".replace("-", "\ensuremath{-}"))
        put(tag + "P", p_fmt(r[k + "_p"]))
    c = pd.read_csv(R / "bb_sci_components_bs.csv").set_index("outcome")
    for o, tag in (("felt arousal", "Felt"), ("valence congruence", "Cong"), ("perceived arousal", "Perc")):
        put(f"Comp{tag}", f"{c.loc[o, 'beta']:.2f}" if abs(c.loc[o, "beta"]) >= .05 else f"{c.loc[o, 'beta']:.3f}")
        put(f"Comp{tag}Lo", f"{c.loc[o, 'ci_lo']:.2f}".replace("-", "\ensuremath{-}")); put(f"Comp{tag}Hi", f"{c.loc[o, 'ci_hi']:.2f}")
        put(f"Comp{tag}P", p_fmt(c.loc[o, "p"]))
    st2 = pd.read_csv(R / "affec_stim_features.csv")
    put("PeakMotionMedian", st2.bs_t_peak.median(), "{:.1f}")
    x = h[(h.kinematics == "x2c") & (h.dataset == "affec")].iloc[0]
    put("XLmmSCI", f"{x.lmm_sci_beta:.3f}".replace("-", "\ensuremath{-}")); put("XLmmSCIP", p_fmt(x.lmm_sci_p))
    s = pd.read_csv(R / "bb_slopes.csv")
    bs = s[(s.kinematics == "bs") & (s.predictor == "zA")]
    put("NSlopesBs", len(bs)); put("MinFDRBs", p_fmt(bs.p_fdr.min()))
    st_ = pd.read_csv(R / "affec_stim_features.csv")
    put("RAmpSpeed", r_fmt(st_.bs_amp.corr(st_.bs_speed)))


def sensitivity():
    s = pd.read_csv(R / "timing_sensitivity_summary.csv").set_index("scale")
    for sc, tag in ((0.0, "Zero"), (0.25, "Quarter"), (0.5, "Half"), (1.0, "Full")):
        put(f"Sens{tag}", 100 * s.loc[sc, "detect_itc"], "{:.0f}")
        put(f"Sens{tag}Gfp", 100 * s.loc[sc, "detect_gfp"], "{:.0f}")
        put(f"Sens{tag}GrpItc", p_fmt(s.loc[sc, "group_itc_p"])); put(f"Sens{tag}GrpGfp", p_fmt(s.loc[sc, "group_gfp_p"]))
    t = pd.read_csv(R / "event_timing_summary.csv").set_index("dataset")
    for ds, tag in (("miko", "Miko"), ("miko_author", "MikoAuth"), ("affec", "Affec")):
        put(f"Grp{tag}Itc", p_fmt(t.loc[ds, "group_itc_p"])); put(f"Grp{tag}Gfp", p_fmt(t.loc[ds, "group_gfp_p"]))


def event_checks():
    a = pd.read_csv(R / "event_onset_thresholds.csv").set_index("threshold")
    for th, tag in ((0.1, "Ten"), (0.3, "Thirty")):
        put(f"Onset{tag}Median", a.loc[th, "median_onset_ms"], "{:.0f}"); put(f"Onset{tag}Late", a.loc[th, "pct_after_300ms"], "{:.0f}")
    b = pd.read_csv(R / "event_type_within_affec.csv").set_index("component")
    for c, tag in (("P300", "Pthree"), ("LPP", "LPP")):
        put(f"Evt{tag}T", b.loc[c, "t"], "{:.2f}"); put(f"Evt{tag}P", p_fmt(b.loc[c, "p"]))
        put(f"Evt{tag}Dz", r_fmt(b.loc[c, "dz"])); put(f"Evt{tag}Df", int(b.loc[c, "n"]) - 1)
        put(f"Evt{tag}Vid", f"{b.loc[c, 'mean_video']:.2f}".replace("-", "\ensuremath{-}"))
        put(f"Evt{tag}Expr", f"{b.loc[c, 'mean_expression']:.2f}".replace("-", "\ensuremath{-}"))


def late_lpp():
    d = pd.read_csv(R / "late_lpp.csv").set_index("window")
    for w, tag in (("sustained", "Sus"), ("peak-locked", "Peak")):
        put(f"Late{tag}", f"{d.loc[w, 'beta']:.3f}".replace("-", "\ensuremath{-}"))
        put(f"Late{tag}Lo", f"{d.loc[w, 'ci_lo']:.3f}".replace("-", "\ensuremath{-}")); put(f"Late{tag}Hi", f"{d.loc[w, 'ci_hi']:.3f}")
        put(f"Late{tag}P", p_fmt(d.loc[w, "p"]))


def main():
    for fn in (data_quality, timing, stimulus, montage, kinematics, cross_dataset, calibration, affec_internal,
               sensitivity, event_checks, late_lpp):
        safe(fn)
    lines = ["% Auto-generated by analysis/16_paper_numbers.py - do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(len(M), "macros written;", sum(v == "??" for v in M.values()), "missing")


if __name__ == "__main__":
    main()
