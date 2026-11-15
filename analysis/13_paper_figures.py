"""Step 13: publication figures for the Cognitive Robotics submission (validity-audit framing, V20).

Journal rules (submission/journal_reference/期刊要求汇总.md): one file per figure, vector PDF
(fonts embedded) plus TIFF at 600 dpi, Arial, colour-blind-safe palette (Okabe-Ito), captions in
a separate file. All panels are drawn from result files, so they follow any re-run.

Fig 1  audit design: databases -> planned comparison -> validity checks C1-C7 -> AFFEC analysis
Fig 2  C1 event timing (Miko marker spacing, ITC real vs surrogate, grand-average ERPs)
Fig 3  C2-C4 event type, speech overlap, 16-channel measurement validity
Fig 4  C5 kinematic cross-domain validity and the objective eye-opening measure
Fig 5  C6-C7 preprocessing dependence, model calibration, apparent effects by event type
Fig 6  AFFEC: motion amplitude, subjective contagion and LPP
Output: ../submission/figures/FigN.pdf and FigN.tif
"""
import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import config as cfg
from common import log

OUT = cfg.ANALYSIS_DIR.parent / "submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = cfg.RESULTS_DIR
PHYS, VIRT, AUTH = "#D55E00", "#0072B2", "#E69F00"
GREEN, PINK, GREY = "#009E73", "#CC79A7", "#7F7F7F"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
MM = 1 / 25.4
W2 = 190 * MM


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    log("saved", name)


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------- Fig 1
def audit_status():
    """Pass/fail per check from the result files (used by Fig 1 and the manuscript)."""
    st = {}
    tim = read("event_timing_summary.csv")
    if tim is not None:
        t = tim.set_index("dataset")
        from scipy import stats as _st
        if "miko" in t.index:        # pass: more time-locked participants than chance (binomial, alpha .05)
            k = int(round(t.loc["miko", "frac_itc_p05"] * t.loc["miko", "n"]))
            st["C1"] = bool(_st.binom.sf(k - 1, int(t.loc["miko", "n"]), .05) < .05) if k else False
    st["C2"] = False            # video- vs expression-locked effects reverse (V2/V18)
    dec = read("affec_stim_features.csv")
    st["C3"] = None if dec is None else bool((dec.speech_onset < 0.2).mean() < 0.1)
    val = read("erp_stats_validity.csv")
    st["C4"] = None if val is None else dict(zip(val.component, val.valid_16ch))
    st["C5"] = False            # X2C fails on AFFEC/Miko (V16); eye geometry used instead (V17)
    cs = read("conn_stats.csv")
    st["C6"] = None
    if cs is not None:
        e = cs[(cs.effect == "embodiment") & (cs.metric == "geff") & (cs.band == "beta")]
        pp = e.set_index("analysis").p
        st["C6"] = bool((pp.get("primary", 1) < .05) == (pp.get("miko_author", 1) < .05))
    cal = read("model_calibration_summary.csv")
    st["C7"] = None if cal is None else dict(zip(cal.model + ":" + cal.component, cal.false_positive_rate))
    return st


def fig1():
    fig, ax = plt.subplots(figsize=(W2, 92 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(-1, 52); ax.axis("off")

    def box(x, y, w, h, title, body, color, fs=6.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    fc=color, ec="0.35", lw=0.6))
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=7.3, fontweight="bold")
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=fs, linespacing=1.25)

    box(0.5, 34, 27, 14.5, "Physical robot: Miko I", "co-present robot, screen eyes\n10 participants, 16 ch, 128 Hz\n"
        "neutral → expression change\nmarkers from a co-registered camera", "#FBE3D6")
    box(0.5, 17.5, 27, 15, "Screen-mediated faces: AFFEC", "CREMA-D actor videos with speech\n71 participants with EEG, 63 ch\n"
        "fixation → face onset\nfelt and perceived ratings", "#D6E6F4")
    box(0.5, 2, 27, 14, "Expression parameters: X2C", "100,000 humanoid faces\n30 control values per image\n"
        "image → control regressor", "#E8E8E8")
    box(32, 17.5, 22, 31, "Planned comparison", "unified preprocessing\n16 common channels\n\nRoute 1: ERP\n(N170, P300, LPP)\n"
        "Route 2: dwPLI networks\nRoute 3: kinematics – EEG\n\nembodied emotional\ndistance (EED)", "#F2F2F2")
    checks = [("C1", "event timing", "markers time-locked to the EEG?"),
              ("C2", "event type", "face onset vs expression change"),
              ("C3", "auditory overlap", "speech within ERP windows?"),
              ("C4", "montage validity", "16-ch ROI tracks the full montage?"),
              ("C5", "kinematic transfer", "parameters valid across face types?"),
              ("C6", "preprocessing dependence", "effect survives alternative cleaning?"),
              ("C7", "model calibration", "false-positive rate under the null")]
    st = audit_status()
    for i, (cid, name, q) in enumerate(checks):
        y = 44.5 - i * 6.1
        v = st.get(cid)
        if isinstance(v, dict):
            vals = [bool(x) for x in v.values()] if cid != "C7" else []
            v = (True if all(vals) else (False if not any(vals) else None)) if vals else None
        v = None if v is None else bool(v)
        col = "#E2F0D9" if v is True else ("#F8D7D3" if v is False else "#FFF4CC")
        ax.add_patch(FancyBboxPatch((58, y), 22, 5, boxstyle="round,pad=0.25,rounding_size=0.8", fc=col, ec="0.4", lw=0.5))
        ax.text(59, y + 3.4, f"{cid}  {name}", fontsize=6.6, fontweight="bold", va="center")
        ax.text(59, y + 1.3, q, fontsize=5.8, va="center")
    ax.text(69, 50.4, "Validity audit", ha="center", fontsize=7.5, fontweight="bold")
    box(84, 26, 15.5, 22.5, "Outcome", "cross-dataset\nembodiment effects\ninterpreted only\nwhere C1–C7 hold\n\n"
        "AFFEC-internal\nkinematics –\ncontagion – LPP", "#E2F0D9", fs=6.0)
    ax.text(84, 20, "fill: green = passed\nred = failed\nyellow = condition-\ndependent", fontsize=5.8, va="top")
    kw = dict(arrowstyle="-|>", color="0.3", lw=0.7, mutation_scale=7)
    for y in (41, 25):
        ax.annotate("", xy=(31.5, y), xytext=(28, y), arrowprops=kw)
    ax.annotate("", xy=(31.5, 20), xytext=(28, 9), arrowprops=kw)
    ax.annotate("", xy=(57.5, 33), xytext=(54.5, 33), arrowprops=kw)
    ax.annotate("", xy=(83.5, 37), xytext=(80.5, 37), arrowprops=kw)
    save(fig, "Fig1")


# ---------------------------------------------------------------------------- Fig 2
def grand_waves():
    R12 = importlib.import_module("12_robustness")
    mk, af = R12.build("pipeline", "video")
    t = next(iter(mk.values()))[1]
    sel = lambda d: np.stack([z for z, _, n in d.values() if n >= cfg.MIN_TRIALS_SHARED])
    mka, _ = R12.build("author", "video")
    return t, sel(mk), sel(mka), sel(af)


def fig2(waves):
    iv = read("miko_marker_intervals.csv")
    tim = read("event_timing_check.csv")
    sens = read("timing_sensitivity_summary.csv")
    fig = plt.figure(figsize=(W2, 118 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.55, wspace=0.35)
    d_ = fig.add_subplot(gs[1, 1]); label(d_, "d", x=-0.12)
    if sens is not None:
        x = sens.scale.values
        d_.plot(x, 100 * sens.detect_itc, "o-", color=PHYS, ms=4, label="ITC, individual participants")
        d_.plot(x, 100 * sens.detect_gfp, "s--", color=AUTH, ms=4, label="GFP, individual participants")
        for _, r in sens.iterrows():
            if "group_gfp_p" in sens and min(r.group_itc_p, r.group_gfp_p) < .05:
                d_.text(r.scale, 104, "*", ha="center", fontsize=10)
        d_.set(xticks=x, xlabel="injected response (× AFFEC expression-onset response)", ylabel="detected (%)",
               ylim=(0, 112))
        d_.legend(frameon=False, fontsize=6, loc="upper left")
        d_.text(.98, .03, "* group-level p < .05", transform=d_.transAxes, ha="right", fontsize=6)
    d_.set_title("sensitivity for Miko's event type", pad=8)
    gs = [gs[0, 0], gs[0, 1], gs[1, 0]]
    a = fig.add_subplot(gs[0]); label(a, "a")
    s = iv[iv.order == "stored order"].interval_s
    lo, hi = np.floor(s.min() / 5) * 5, np.ceil(s.max() / 5) * 5
    a.hist(s, np.arange(lo, hi + 1.5, 1.5), color=PHYS, alpha=.8)
    a.axvline(8, color="k", lw=.8, ls="--")
    a.axvline(0, color=GREY, lw=.5)
    a.text(10, .97, "design:\n8 s", fontsize=6.3, va="top", transform=a.get_xaxis_transform())
    a.text(.03, .97, f"{100 * (s < 0).mean():.0f}% negative", fontsize=6.3, va="top", transform=a.transAxes)
    a.set(xlabel="interval between successive markers (s)", ylabel="count (10 participants)")
    a.set_title("Miko marker spacing", pad=8)
    b = fig.add_subplot(gs[1]); label(b, "b")
    for ds, col, mk, nm in (("affec", VIRT, "o", "AFFEC"), ("miko", PHYS, "s", "Miko (pipeline)"),
                            ("miko_author", AUTH, "D", "Miko (author-cleaned)")):
        d = tim[tim.dataset == ds]
        sig = d.itc_p < .05
        b.scatter(d.itc_surr_median[~sig], d.itc_real[~sig], s=12, marker=mk, facecolor="none", edgecolor=col, lw=.7,
                  label=f"{nm}: {sig.sum()}/{len(d)} p < .05")
        b.scatter(d.itc_surr_median[sig], d.itc_real[sig], s=12, marker=mk, color=col, lw=0)
    lim = max(tim.itc_real.max(), tim.itc_surr_median.max()) * 1.05
    b.plot([0, lim], [0, lim], color=GREY, lw=.6)
    b.set(xlim=(0, lim), ylim=(0, lim), xlabel="ITC at shifted markers (median)", ylabel="ITC at recorded markers")
    b.set_title("time-locking (ITC, 2–8 Hz)", pad=8)
    b.legend(frameon=False, fontsize=5.8, loc="upper left")
    c = fig.add_subplot(gs[2]); label(c, "c", x=-0.12)
    t, mk, mka, af = waves
    roi = [cfg.COMMON_CHANNELS.index(ch) for ch in ("Pz", "POz")]
    for x, col, nm in ((af, VIRT, "AFFEC (video onset)"), (mk, PHYS, "Miko (pipeline)"), (mka, AUTH, "Miko (author-cleaned)")):
        y = x[:, roi].mean(1)
        mu, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(len(y))
        c.plot(t * 1000, mu, color=col, lw=1, label=f"{nm}, n = {len(y)}")
        c.fill_between(t * 1000, mu - se, mu + se, color=col, alpha=.2, lw=0)
    c.axhline(0, color=GREY, lw=.4); c.axvline(0, color=GREY, lw=.4)
    c.set(xlim=(-200, 1000), xlabel="time (ms)", ylabel="Pz/POz (baseline-SD units)")
    c.set_title("grand averages", pad=8)
    c.legend(frameon=False, fontsize=6, loc="upper right")
    save(fig, "Fig2")


# ---------------------------------------------------------------------------- Fig 3
def affec_lock_waves():
    R12 = importlib.import_module("12_robustness")
    out = {}
    for lock in ("video", "expression"):
        _, af = R12.build("pipeline", lock)
        t = next(iter(af.values()))[1]
        out[lock] = (t, np.stack([z for z, _, n in af.values() if n >= cfg.MIN_TRIALS_SHARED]))
    return out


def fig3(lockw):
    stim = read("affec_stim_features.csv")
    tr = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    fig = plt.figure(figsize=(W2, 110 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)
    a = fig.add_subplot(gs[0, :2]); label(a, "a", x=-0.08)
    roi = [cfg.COMMON_CHANNELS.index(ch) for ch in ("Pz", "POz")]
    for lock, col in (("video", VIRT), ("expression", GREEN)):
        t, x = lockw[lock]
        y = x[:, roi].mean(1)
        mu, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(len(y))
        a.plot(t * 1000, mu, color=col, lw=1, label=f"locked to {lock} onset (n = {len(y)})")
        a.fill_between(t * 1000, mu - se, mu + se, color=col, alpha=.2, lw=0)
    for comp, c in (("P300", "#FFF4CC"), ("LPP", "#E2F0D9")):
        t0, t1 = cfg.COMPONENTS[comp][:2]
        a.axvspan(t0 * 1000, t1 * 1000, color=c, zorder=0, lw=0)
        a.text((t0 + t1) / 2 * 1000, 1.0, comp, ha="center", va="bottom", fontsize=6.5, transform=a.get_xaxis_transform())
    a.axhline(0, color=GREY, lw=.4); a.axvline(0, color=GREY, lw=.4)
    a.set(xlim=(-200, 1000), xlabel="time (ms)", ylabel="Pz/POz (baseline-SD units)")
    a.set_title("AFFEC: same trials, two time-locking events", pad=12)
    a.legend(frameon=False, loc="lower right")
    b = fig.add_subplot(gs[0, 2]); label(b, "b")
    bins = np.arange(0, 1.25, 0.05)
    b.hist(stim.expr_onset, bins, color=GREEN, alpha=.7, label="expression-motion onset")
    b.hist(stim.speech_onset, bins, color=PINK, alpha=.6, label="speech onset")
    b.axvspan(0.13, 0.20, color="0.85", zorder=0, lw=0)
    b.set(xlabel="time after video onset (s)", ylabel=f"clips (n = {len(stim)})")
    b.legend(frameon=False, fontsize=6)
    b.text(0.165, 1.0, "N170", ha="center", va="bottom", fontsize=6, transform=b.get_xaxis_transform())
    pairs = [("N170", "pipeline", "N170", "full62", "N170_std", "16-ch N170 (T7/P3/T8/P4)", "62-ch N170 (P7/PO7/P8/PO8)"),
             ("P300", "pipeline", "P300", "full62", "P300", "16-ch P300", "62-ch P300"),
             ("LPP", "pipeline", "LPP", "full62", "LPP", "16-ch LPP", "62-ch LPP")]
    vs = read("erp_validation16_subjects.csv")          # same table as the statistics (step 06)
    for j, (comp, v1, c1, v2, c2, xl, yl) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, j]); label(ax, "cde"[j])
        d = vs[(vs.component == comp) & (vs.reference == f"{v2}:{c2}")].rename(columns={"amp_16ch": "u", "amp_full": "v"})
        r = d.u.corr(d.v)
        ax.scatter(d.u, d.v, s=9, color=VIRT if r >= .7 else PHYS, alpha=.8, lw=0)
        ax.set(xlabel=xl, ylabel=yl, title=f"r = {r:.2f}, n = {len(d)} ({'passes' if r >= .7 else 'fails'} r ≥ .70)")
    save(fig, "Fig3")


# ---------------------------------------------------------------------------- Fig 4
def fig4():
    val = read("x2c_crossdomain_validity.csv")
    geo = read("eye_geometry_kinematics.csv")
    mk = read("miko_eye_geometry.csv")
    fig = plt.figure(figsize=(W2, 64 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.3], wspace=0.5)
    a = fig.add_subplot(gs[0]); label(a, "a", x=-0.55)
    pr = val[val.test == "a priori pair"].copy()
    pr["lab"] = pr.component.str.replace(" ~ ", " ↔ ")
    y = np.arange(len(pr))
    a.barh(y, pr.r, color=[VIRT if r > .3 else PHYS for r in pr.r])
    a.axvline(0, color="k", lw=.5); a.axvline(.3, color=GREY, ls="--", lw=.6)
    a.set(yticks=y, xlabel="frame-wise r (AFFEC, X2C vs MediaPipe)", title="X2C on human faces")
    a.set_yticklabels(pr.lab, fontsize=5.6)
    cc = val[val.test == "CCA out-of-sample"].r.iloc[0]
    a.text(0.98, 0.02, f"held-out CCA r1 = {cc:.2f}", transform=a.transAxes, ha="right", fontsize=6)
    b = fig.add_subplot(gs[1]); b.axis("off"); label(b, "b", x=0.0)
    for i, r in enumerate(mk.itertuples()):
        im = Image.open(cfg.ANALYSIS_DIR / "stimuli" / "miko" / f"{r.emotion}.png")
        sub = b.inset_axes([(i % 3) * 0.34, 0.5 - (i // 3) * 0.52, 0.31, 0.44])
        sub.imshow(im); sub.axis("off")
        sub.set_title(f"{r.emotion}\n{r.eye_logratio_peak:+.2f}", fontsize=6.3, pad=1)
    b.set_title("Miko: log(eye area / neutral)", fontsize=8, loc="center", pad=12)
    c = fig.add_subplot(gs[2]); label(c, "c")
    emos = ["angry", "happy", "sad", "fear", "disgust", "neutral", "surprised"]
    rng = np.random.default_rng(1)
    for i, e in enumerate(emos):
        d = geo[(geo.dataset == "affec") & (geo.emotion == e)]
        c.scatter(i + rng.uniform(-.17, .17, len(d)), d.eye_logratio_peak, s=7, color=VIRT, alpha=.55, lw=0)
        m = geo[(geo.dataset == "miko") & (geo.emotion == e)]
        c.scatter(np.full(len(m), i), m.eye_logratio_peak, marker="*", s=70, color=PHYS, zorder=3, lw=.3, ec="k")
    c.axhline(0, color=GREY, lw=.5)
    c.set(xticks=range(len(emos)), ylabel="log(eye area / neutral) at peak", title="shared eye-opening scale")
    c.set_xticklabels(emos, rotation=35, ha="right")
    c.scatter([], [], s=7, color=VIRT, label="AFFEC clip"); c.scatter([], [], marker="*", s=50, color=PHYS, label="Miko face")
    c.legend(frameon=False, loc="lower left")
    save(fig, "Fig4")


# ---------------------------------------------------------------------------- Fig 5
def fig5():
    top = read("conn_topology.csv")
    cal = read("model_calibration_summary.csv")
    spec = read("robust_spec_curve.csv")
    fig = plt.figure(figsize=(W2, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 1.3], wspace=0.45)
    a = fig.add_subplot(gs[0]); label(a, "a")
    bands = list(cfg.BANDS)
    for k, (ds, ver, col, nm) in enumerate([("miko", "pipeline", PHYS, "Miko, pipeline"),
                                           ("miko", "author", AUTH, "Miko, author-cleaned"),
                                           ("affec", "pipeline", VIRT, "AFFEC")]):
        d = top[(top.dataset == ds) & (top.version == ver)].groupby(["subject", "band"]).geff.mean().unstack()
        mu, se = d[bands].mean(), d[bands].std(ddof=1) / np.sqrt(len(d))
        a.bar(np.arange(3) + (k - 1) * .26, mu, .24, yerr=se, color=col, label=f"{nm} (n = {len(d)})",
              error_kw=dict(lw=.6, capsize=1.5))
    a.axhline(0, color="k", lw=.5)
    a.set(xticks=range(3), ylabel="Δ global efficiency (post − base)", title="C6: cleaning dependence")
    a.set_xticklabels(bands)
    a.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=1)
    b = fig.add_subplot(gs[1]); label(b, "b", x=-0.3)
    if cal is not None:
        for k, (model, col, nm) in enumerate((("single_trial_crossed", PHYS, "single-trial,\ncrossed RE"),
                                              ("cell_mean", VIRT, "cell-mean\nLMM"))):
            d = cal[cal.model == model].reset_index(drop=True)
            x = np.arange(len(d)) + (k - .5) * .3
            b.errorbar(x, d.false_positive_rate, yerr=[d.false_positive_rate - d.ci_lo, d.ci_hi - d.false_positive_rate],
                       fmt="o", color=col, ms=4, capsize=2, lw=.8, label=nm)
        b.set_xticks(range(len(d))); b.set_xticklabels(d.component)
        b.axhline(.05, color="k", ls="--", lw=.6)
    b.set(ylabel="false-positive rate (null splits)", ylim=(0, 1), title="C7: model calibration")
    b.legend(frameon=False, fontsize=6, loc="upper right")
    c = fig.add_subplot(gs[2]); label(c, "c")
    for comp, mk in (("P300", "o"), ("LPP", "^")):
        for lock, col in (("video", VIRT), ("expression", GREEN)):
            d = spec[(spec.component == comp) & (spec.affec_lock == lock)].sort_values("hedges_g")
            x = np.linspace(0, 1, len(d)) + (0 if comp == "P300" else 1.15)
            c.scatter(x, d.hedges_g, s=6, marker=mk, color=col, alpha=.8, lw=0,
                      label=f"{lock} onset" if comp == "P300" else None)
    c.axhline(0, color="k", lw=.5)
    c.set(xticks=[.5, 1.65], ylabel="Hedges' g (Miko − AFFEC)", title="C2: apparent effects by event")
    n_ = spec.groupby(["component", "affec_lock"]).size()
    c.set_xticklabels([k + "\n" + f"{n_[(k, 'video')]} + {n_[(k, 'expression')]} specs" for k in ("P300", "LPP")])
    c.legend(frameon=False, fontsize=6, loc="lower right")
    save(fig, "Fig5")


# ---------------------------------------------------------------------------- Fig 6
def fig6():
    tr = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    stim = read("affec_stim_features.csv")
    h4 = read("bb_h4.csv")
    a_ = tr[(tr.dataset == "affec") & (tr.version == "pipeline") & (tr.component == "LPP") &
            (tr.min_shared_trials >= cfg.MIN_TRIALS_SHARED)].copy()
    fv, pv = a_.f_emotion_v - 5, a_.p_emotion_v - 5
    a_["SCI"] = a_.f_emotion_a * ((np.sign(fv) == np.sign(pv)) & ((a_.f_emotion_v - a_.p_emotion_v).abs() <= 2))
    clip = a_.groupby("stim").agg(LPP=("amp_z", "mean"), SCI=("SCI", "mean"), emotion=("emotion", "first")).join(
        stim.set_index("stim")[["bs_amp"]])
    clip["zA"] = (clip.bs_amp - stim.bs_amp.mean()) / stim.bs_amp.std()
    fig = plt.figure(figsize=(W2, 60 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, .8], wspace=0.45)
    for j, (col, yl, c) in enumerate((("SCI", "subjective contagion (clip mean)", GREEN),
                                      ("LPP", "LPP, Pz/POz (clip mean, SD units)", VIRT))):
        ax = fig.add_subplot(gs[j]); label(ax, "ab"[j])
        ax.scatter(clip.zA, clip[col], s=10, color=c, alpha=.8, lw=0)
        b = np.polyfit(clip.zA, clip[col], 1)
        xs = np.linspace(clip.zA.min(), clip.zA.max(), 10)
        ax.plot(xs, np.polyval(b, xs), color="k", lw=.8)
        r = clip.zA.corr(clip[col])
        ax.set(xlabel="facial motion amplitude (z, MediaPipe)", ylabel=yl, title=f"{len(clip)} clips, 6 emotions, r = {r:.2f}")
    c = fig.add_subplot(gs[2]); label(c, "c", x=-0.4)
    r = h4[(h4.kinematics == "bs") & (h4.dataset == "affec")].iloc[0]
    y = [r.beta_subjective, r.beta_neural_LPP]
    ylm = [r.lmm_sci_beta, r.lmm_lpp_beta]
    c.errorbar([-.12, .88], ylm, yerr=[[ylm[0] - r.lmm_sci_lo, ylm[1] - r.lmm_lpp_lo], [r.lmm_sci_hi - ylm[0], r.lmm_lpp_hi - ylm[1]]], fmt="s", color=GREEN, ms=4, capsize=2, lw=.8, label="trial-level LMM")
    c.errorbar([.12, 1.12], y, yerr=[[y[0] - r.bs_ci_lo, y[1] - r.bn_ci_lo], [r.bs_ci_hi - y[0], r.bn_ci_hi - y[1]]],
               fmt="o", color="k", ms=4, capsize=2, lw=.8, label="participant bootstrap")
    c.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.2))
    c.axhline(0, color=GREY, lw=.5)
    c.set(xticks=[0, 1], xlim=(-.5, 1.5), ylabel="slope per SD of amplitude (95% CI)", title="slopes")
    c.set_xticklabels(["contagion", "LPP"])
    save(fig, "Fig6")


def main():
    fig1()
    fig2(grand_waves())
    fig3(affec_lock_waves())
    fig4(); fig5(); fig6()


if __name__ == "__main__":
    main()
