"""Step 22: data figures 2-8 of the framework-structured manuscript in the card layout of Fig. 1.

Fig 2 route 1: amplitude, latency, topography          Fig 6 C5: kinematic descriptors across face types
Fig 3 C1: event timing                                  Fig 7 route 3: motion, contagion and LPP
Fig 4 C2-C4: event type, speech, montage                Fig 8 embodied emotional distance
Fig 5 C6-C7: calibration, network cleaning, specifications
All panels read the result files written by the analysis steps (same tables as the statistics).
Output: submission/figures_v3/FigN.png (600 dpi) and FigN.pdf
"""
import importlib
import json
import numpy as np
import pandas as pd
import mne
from PIL import Image

import config as cfg
import figkit as fk
from figkit import (INK, GREY, LG, TEAL, TERRA, GOLD, PURP, GREEN, TINT, PHYS, SCREEN, AUTH)

R = cfg.RESULTS_DIR
OUT = cfg.ANALYSIS_DIR.parent / "submission" / "figures_v3"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(3)
COMPS = ["N170", "P300", "LPP"]
ROI = [cfg.COMMON_CHANNELS.index(c) for c in ("Pz", "POz")]


def read(n):
    return pd.read_csv(R / n)


def ax_mm(x, y, w, h):
    a = fk.FIG.add_axes([x / fk.W, y / fk.H, w / fk.W, h / fk.H], zorder=5)
    a.set_facecolor("none")
    return a


def fmt_p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def save(n):
    fk.save(OUT / f"{n}.png", OUT / f"{n}.pdf")
    print("saved", n)


def mean_se(y):
    return y.mean(0), y.std(0, ddof=1) / np.sqrt(len(y))


# ================================================================ Fig 2
def fig2():
    trials = read("erp_trials.csv") if False else pd.read_csv(R / "erp_trials.csv", low_memory=False)
    ok = trials[trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED]
    inc = {ds: set(ok[(ok.dataset == ds) & (ok.version == "pipeline")].subject) for ds in ("miko", "affec")}
    t_ = ok[(ok.version == "pipeline") & ok.emotion.isin(cfg.SHARED_EMOTIONS) & ok.component.isin(COMPS)]
    amp = t_.groupby(["component", "dataset", "subject"]).amp_z.mean()
    eq = read("erp_stats_equivalence.csv").set_index("component")
    tests = read("erp_stats_tests.csv"); tests = tests[tests.analysis == "primary"].set_index(["component", "measure", "effect"])
    lat = read("erp_subject_latency.csv")
    lat = lat[(lat.version == "pipeline") & lat.emotion.isin(cfg.SHARED_EMOTIONS)]
    lat = lat[[s in inc[d] for d, s in zip(lat.dataset, lat.subject)]]
    lsub = lat.groupby(["component", "dataset", "subject"]).latency.mean() * 1000
    tan = read("erp_stats_tanova.csv").set_index("component")

    fk.canvas(180, 124)
    for k, (letter, title, data, ylab, key, icon) in enumerate(
            [("a", "Amplitude", amp, "Mean amplitude (baseline SD)", "amplitude", fk.ic_wave),
             ("b", "Peak timing", lsub, "50% fractional-area latency (ms)", "latency", fk.ic_stopwatch)]):
        a = fk.card(2 + k * 74, 63, 72, 59, letter, title, TEAL, icon, pad=(11, 7.5, 2.5, 9.5))
        for i, c in enumerate(COMPS):
            for j, (ds, col) in enumerate((("miko", PHYS), ("affec", SCREEN))):
                v = data.loc[(c, ds)].values
                x = i + (j - 0.5) * 0.38
                a.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=5, color=col, alpha=0.35, lw=0, zorder=3)
                m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                a.errorbar(x, m, yerr=se, fmt="o", ms=4.2, color=col, mec="white", mew=0.5, capsize=2, lw=0.9, zorder=4)
            p = tests.loc[(c, key, "embodiment"), "p"]; pf = tests.loc[(c, key, "embodiment"), "p_fdr"]
            s = (f"g = {eq.loc[c, 'hedges_g']:.2f}\n" if key == "amplitude" else "") + fmt_p(p) + f"\nFDR {pf:.2f}".replace("0.", ".")
            a.text(i, 1.01, s.replace("-", "−"), transform=a.get_xaxis_transform(), ha="center", va="bottom",
                   fontsize=5.6, color=GREY if c == "N170" else INK)
        a.set_xticks(range(3)); a.set_xticklabels(["N170†", "P300", "LPP"]); a.set_xlim(-0.6, 2.6)
        a.set_ylabel(ylab)
        if key == "amplitude":
            a.axhline(0, color=GREY, lw=0.5)
        fk.tidy(a)
    # sample card
    fk.rbox(150, 63, 28, 59, fc="#F7F9FA", ec=LG, lw=0.6, r=1.8)
    fk.t(164, 118.6, "Embodiment form", 6.4, "bold")
    fk.robot(164, 103.5, 0.95)
    fk.t(164, 93.4, "Physical · Miko", 6.0, "bold", PHYS)
    fk.t(164, 90.2, "co-present robot, n = 10", 5.4)
    fk.monitor(164, 80.0, 0.9)
    fk.t(164, 73.2, "Screen-mediated · AFFEC", 6.0, "bold", SCREEN)
    fk.t(164, 70.0, "actor videos, n = 49", 5.4)
    fk.t(164, 65.6, "† not montage-valid (C4)", 5.0, c=GREY, style="italic")

    c_ = fk.card(2, 2, 176, 58, "c", "Scalp distribution (GFP-normalized, TANOVA)", TEAL, fk.ic_head, axes=False)
    maps = {}
    for ds in ("miko", "affec"):
        ev = []
        for s in sorted(inc[ds]):
            f = cfg.DERIV_DIR / "erp" / f"{ds}_pipeline_{s}-ave.fif"
            if f.exists():
                e = [x for x in mne.read_evokeds(f, verbose=False) if x.comment in cfg.SHARED_EMOTIONS]
                ev.append(mne.combine_evoked(e, "equal").pick(cfg.COMMON_CHANNELS))
        maps[ds] = ev
    info = maps["affec"][0].info
    for i, c in enumerate(COMPS):
        t0, t1 = cfg.COMPONENTS[c][:2]
        x0 = 16 + i * 52
        fk.rbox(x0 - 1.5, 4.0, 48, 48.5, fc="#FAFBFC", ec=LG, lw=0.4, r=1.2, z=2)
        fk.t(x0 + 22.5, 49.0, f"{c}  {t0 * 1000:.0f}–{t1 * 1000:.0f} ms", 6.2, "bold")
        fk.chip(x0 + 22.5, 7.6, "TANOVA " + fmt_p(tan.loc[c, "p_perm"]), TEAL if tan.loc[c, "p_perm"] < .05 else GREY, size=5.6, w=28)
        for j, ds in enumerate(("miko", "affec")):
            X = []
            for e in maps[ds]:
                m = e.copy().crop(t0, t1).data.mean(1); m = m - m.mean(); X.append(m / np.sqrt(np.mean(m ** 2)))
            m = np.mean(X, 0); m = m / np.sqrt(np.mean(m ** 2))
            a = ax_mm(x0 + j * 23.5, 12.5, 21.5, 31)
            im, _ = mne.viz.plot_topomap(m, info, axes=a, show=False, cmap="RdBu_r", vlim=(-2, 2), contours=4,
                                         sensors=True, sphere="auto")
            fk.t(x0 + 10.75 + j * 23.5, 44.6, ["Physical", "Screen"][j], 5.8, "bold", [PHYS, SCREEN][j])
    cax = ax_mm(168.5, 12, 2.4, 34)
    cb = fk.FIG.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
    cb.set_label("normalized amplitude", fontsize=5.6); cb.ax.yaxis.set_label_position("left")
    save("Fig2")


# ================================================================ Fig 3
def grand_waves():
    R12 = importlib.import_module("12_robustness")
    mk, af = R12.build("pipeline", "video")
    t = next(iter(mk.values()))[1]
    sel = lambda d: np.stack([z for z, _, n in d.values() if n >= cfg.MIN_TRIALS_SHARED])
    mka, _ = R12.build("author", "video")
    return t, sel(mk), sel(mka), sel(af)


def fig3(waves):
    iv = read("miko_marker_intervals.csv"); tim = read("event_timing_check.csv"); sens = read("timing_sensitivity_summary.csv")
    summ = read("event_timing_summary.csv").set_index("dataset")
    fk.canvas(180, 122)
    a = fk.card(2, 62, 86, 58, "a", "Recorded marker spacing", TERRA, fk.ic_clock, pad=(11, 7.5, 3, 2))
    s = iv[iv.order == "stored order"].interval_s
    a.hist(s, np.arange(np.floor(s.min()), np.ceil(s.max()) + 1.5, 1.5), color=PHYS, alpha=0.85, lw=0)
    a.axvline(8, color=INK, lw=0.8, ls="--"); a.axvline(0, color=GREY, lw=0.5)
    a.set_xlabel("Interval between successive markers (s)"); a.set_ylabel("Count (10 participants)")
    fk.tidy(a)
    fk.chip(31, 106, f"{100 * (s < 0).mean():.0f}% negative intervals", TERRA, size=5.8, w=29)
    fk.chip(70, 106, "designed interval: 8 s", INK, size=5.8, w=24)

    b = fk.card(92, 62, 86, 58, "b", "Time-locking at the markers (ITC, 2–8 Hz)", TERRA, fk.ic_target, pad=(17, 7.5, 16, 2))
    rows = [("miko", PHYS, "Miko\npipeline"), ("miko_author", AUTH, "Miko\nauthor"), ("affec", SCREEN, "AFFEC")]
    for k, (ds, col, lab) in enumerate(rows):
        d = tim[tim.dataset == ds]
        g = d.itc_real - d.itc_surr_median
        sig = d.itc_p < .05
        y = 2 - k + rng.uniform(-0.18, 0.18, len(d))
        b.scatter(g[~sig], y[~sig], s=9, facecolor="white", edgecolor=col, lw=0.6, zorder=3)
        b.scatter(g[sig], y[sig], s=9, color=col, lw=0, zorder=3)
        b.text(1.02, (2 - k + 0.5) / 3.0, f"{sig.sum()}/{len(d)}", transform=b.transAxes, color=col, fontsize=6.4,
               fontweight="bold", va="center")
        b.text(1.02, (2 - k + 0.5) / 3.0 - 0.09, "detected", transform=b.transAxes, color=col, fontsize=5.2, va="center")
    b.axvline(0, color=GREY, lw=0.6)
    b.set_yticks([2, 1, 0]); b.set_yticklabels([r[2] for r in rows]); b.set_ylim(-0.5, 2.5)
    b.set_xlabel("ITC at markers − median ITC at shifted markers")
    fk.tidy(b, grid=False)
    # schematic: recorded vs circularly shifted markers
    fk.t(100.5, 112.3, "test", 4.8, c=GREY, style="italic")
    for k_, (yy, col, off) in enumerate(((112.8, INK, 0.0), (110.6, GREY, 2.2))):
        fk.OV.plot([104, 120], [yy, yy], color=LG, lw=0.5, zorder=9)
        for xx in (105.5, 109.0, 113.5, 117.0):
            x_ = 104 + ((xx + off - 104) % 16)
            fk.OV.plot([x_, x_], [yy - 0.6, yy + 0.6], color=col, lw=0.7, zorder=10)
    fk.t(121.0, 112.8, "recorded", 4.6, ha="left"); fk.t(121.0, 110.6, "shifted ×500", 4.6, ha="left", c=GREY)
    fk.chip(154, 111.6, f"group ITC {fmt_p(summ.loc['miko', 'group_itc_p'])} (Miko)", PHYS, size=5.3, w=34)

    c = fk.card(2, 2, 86, 57, "c", "Grand-average response at Pz/POz", TEAL, fk.ic_wave, pad=(11, 7.5, 3, 2))
    tt, mk, mka, af = waves
    for x, col in ((af, SCREEN), (mk, PHYS), (mka, AUTH)):
        mu, se = mean_se(x[:, ROI].mean(1))
        c.plot(tt * 1000, mu, color=col, lw=0.9)
        c.fill_between(tt * 1000, mu - se, mu + se, color=col, alpha=0.18, lw=0)
    c.axhline(0, color=GREY, lw=0.4); c.axvline(0, color=GREY, lw=0.4)
    c.set_xlim(-200, 1000); c.set_xlabel("Time from recorded marker (ms)"); c.set_ylabel("Amplitude (baseline SD)")
    fk.tidy(c)
    fk.legend_item(55, 49.5, "line", SCREEN, f"AFFEC video onset (n = {len(af)})", 5.3)
    fk.legend_item(55, 46.4, "line", PHYS, f"Miko pipeline (n = {len(mk)})", 5.3)
    fk.legend_item(55, 43.3, "line", AUTH, f"Miko author-cleaned (n = {len(mka)})", 5.3)

    d = fk.card(92, 2, 86, 57, "d", "Sensitivity: injected AFFEC response", TERRA, fk.ic_bell, pad=(11, 7.5, 3, 5))
    x = sens.scale.values
    d.plot(x, 100 * sens.detect_itc, "o-", color=PHYS, ms=3.5, lw=0.9, label="ITC")
    d.plot(x, 100 * sens.detect_gfp, "s--", color=AUTH, ms=3.5, lw=0.9, label="GFP")
    for _, r in sens.iterrows():
        if min(r.group_itc_p, r.group_gfp_p) < .05:
            d.text(r.scale, 103, "*", ha="center", fontsize=9, color=INK)
    d.set_xticks(x); d.set_ylim(-4, 112)
    d.set_xlabel("Injected amplitude (× AFFEC expression response)"); d.set_ylabel("Participants detected (%)")
    d.legend(frameon=False, fontsize=5.8, loc="center left")
    fk.tidy(d)
    fk.t(170, 12.5, "* group-level p < .05", 5.2, c=GREY, ha="right")
    save("Fig3")


# ================================================================ Fig 4
def affec_lock_waves():
    R12 = importlib.import_module("12_robustness")
    out = {}
    for lock in ("video", "expression"):
        _, af = R12.build("pipeline", lock)
        t = next(iter(af.values()))[1]
        out[lock] = (t, np.stack([z for z, _, n in af.values() if n >= cfg.MIN_TRIALS_SHARED]))
    return out


def fig4(lockw):
    stim = read("affec_stim_features.csv"); vs = read("erp_validation16_subjects.csv")
    et = read("event_type_within_affec.csv").set_index("component")
    fk.canvas(180, 122)
    a = fk.card(2, 62, 108, 58, "a", "C2  Event type: same AFFEC trials, two time-locking events", GOLD, fk.ic_swap,
                pad=(11, 7.5, 3, 2))
    for comp, col in (("P300", "#F6EBCB"), ("LPP", "#DDEDE3")):
        t0, t1 = cfg.COMPONENTS[comp][:2]
        a.axvspan(t0 * 1000, t1 * 1000, color=col, lw=0, zorder=0, alpha=0.8)
        a.text((t0 + t1) / 2 * 1000, 0.03, comp, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8)
    for lock, col in (("video", SCREEN), ("expression", PURP)):
        tt, x = lockw[lock]
        mu, se = mean_se(x[:, ROI].mean(1))
        a.plot(tt * 1000, mu, color=col, lw=1.0)
        a.fill_between(tt * 1000, mu - se, mu + se, color=col, alpha=0.18, lw=0)
    a.axhline(0, color=GREY, lw=0.4); a.axvline(0, color=GREY, lw=0.4)
    a.set_xlim(-200, 1000); a.set_xlabel("Time from alignment event (ms)"); a.set_ylabel("Pz/POz (baseline SD)")
    fk.tidy(a)
    fk.legend_item(18, 75.0, "line", SCREEN, f"video onset (face appears), n = {len(lockw['video'][1])}", 5.4)
    fk.legend_item(18, 71.8, "line", PURP, f"expression-motion onset, n = {len(lockw['expression'][1])}", 5.4)
    r = et.loc["P300"]
    a.text(0.985, 0.95, f"P300 video vs expression: t({int(r.n) - 1}) = {r.t:.2f}, {fmt_p(r.p)}", transform=a.transAxes,
           ha="right", va="top", fontsize=5.4, color=INK, bbox=dict(boxstyle="round,pad=0.35", fc=TINT[GOLD], ec=GOLD, lw=0.45))

    b = fk.card(114, 62, 64, 58, "b", "C3  Speech onset", GOLD, fk.ic_speaker, pad=(11, 7.5, 3, 2))
    for col, lab, v in ((PURP, "expression motion", stim.expr_onset), (TERRA, "speech", stim.speech_onset)):
        xs = np.sort(v.dropna().values)
        b.step(np.r_[0, xs], np.r_[0, np.arange(1, len(xs) + 1) / len(xs) * 100], where="post", color=col, lw=1.0, label=lab)
    b.axvspan(0.13, 0.20, color="#E1E6EA", lw=0, zorder=0)
    b.text(0.165, 0.97, "N170", transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=5.4, color=GREY)
    b.set_xlim(0, 1.25); b.set_ylim(0, 103)
    b.set_xlabel("Time from video onset (s)"); b.set_ylabel("Clips with onset (%)")
    b.legend(frameon=False, fontsize=5.6, loc="lower right")
    fk.tidy(b)
    fk.chip(160, 84, f"{100 * (stim.speech_onset < 0.2).mean():.0f}% speech < 200 ms", TERRA, size=5.4, w=25)

    for j, (comp, ref) in enumerate((("N170", "full62:N170_std"), ("P300", "full62:P300"), ("LPP", "full62:LPP"))):
        d = vs[(vs.component == comp) & (vs.reference == ref)]
        r_ = d.amp_16ch.corr(d.amp_full); okv = r_ >= 0.7
        ax = fk.card(2 + j * 59, 2, 57, 57, "cde"[j], f"C4  {comp} montage validity", GOLD,
                     lambda cx, cy, rr, cc: fk.ic_head(cx, cy, rr * 0.8, cc, ring=True), pad=(11, 7.5, 3, 2))
        col = SCREEN if okv else PHYS
        ax.scatter(d.amp_16ch, d.amp_full, s=7, color=col, alpha=0.75, lw=0, zorder=3)
        bb = np.polyfit(d.amp_16ch, d.amp_full, 1); xs = np.linspace(d.amp_16ch.min(), d.amp_16ch.max(), 5)
        ax.plot(xs, np.polyval(bb, xs), color=col, lw=0.9)
        ax.set_xlabel("16-channel amplitude (SD)"); ax.set_ylabel("62-channel amplitude (SD)")
        fk.tidy(ax)
        fk.chip(2 + j * 59 + 39, 13.0, f"r = {r_:.2f}  n = {len(d)}", col, size=5.6, w=22)
        fk.check(2 + j * 59 + 53.0, 13.0, 1.7, okv)
    save("Fig4")


# ================================================================ Fig 5
def fig5():
    top = read("conn_topology.csv"); cal = read("model_calibration_summary.csv"); spec = read("robust_spec_curve.csv")
    cs = read("conn_stats.csv")
    fk.canvas(180, 122)
    a = fk.card(2, 62, 104, 58, "a", "C7  False-positive rate under null data", TERRA, fk.ic_bell, pad=(30, 7.5, 14, 3))
    rows = []
    for model, col, nm in (("single_trial_crossed", TERRA, "Trial-level crossed"), ("cell_mean", TEAL, "Cell-mean LMM")):
        for comp in ("P300", "LPP"):
            r = cal[(cal.model == model) & (cal.component == comp)].iloc[0]
            rows.append((f"{nm} · {comp}", r, col))
    for i, (lab, r, col) in enumerate(rows):
        y = len(rows) - 1 - i
        a.plot([100 * r.ci_lo, 100 * r.ci_hi], [y, y], color=col, lw=1.1)
        a.plot(100 * r.false_positive_rate, y, "o", color=col, ms=4.5, mec="white", mew=0.5)
        a.text(100 * r.ci_hi + 1.2, y, f"{100 * r.false_positive_rate:.0f}%", va="center", fontsize=6.2, color=col, fontweight="bold")
    a.axvline(5, color=INK, ls="--", lw=0.7)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1]); a.set_xlim(0, 45); a.set_ylim(-0.6, 3.6)
    a.set_xlabel("False-positive rate (%), exact 95% CI")
    fk.tidy(a, grid=False)
    # design card
    fk.rbox(110, 62, 68, 58, fc="#F7F9FA", ec=LG, lw=0.6, r=1.8)
    fk.t(144, 116.2, "Null simulation design", 6.4, "bold")
    for i in range(20):
        cx, cy = 116 + (i % 10) * 5.0, 106.5 - (i // 10) * 5.0
        fk.face(cx, cy, 1.5, z=8, smile=0.3)
        fk.OV.add_patch(fk.Circle((cx, cy), 2.1, fc="none", ec=TERRA if i < 10 else TEAL, lw=0.6, zorder=9))
    fk.t(144, 97.2, "AFFEC participants only: 10 relabeled “physical”", 5.4)
    fk.t(144, 93.8, "Miko structure imposed: one stimulus per emotion", 5.4)
    fk.t(144, 90.4, "100 random splits per model and component", 5.4)
    fk.arr(144, 87.6, 144, 84.6, c=GREY, ms=3.5)
    fk.rbox(114, 70.5, 60, 12.5, fc="white", ec=TERRA, lw=0.5, r=1.2)
    fk.t(144, 79.6, "stimulus intercept ≡ emotion effect", 5.6, "bold", TERRA)
    fk.t(144, 75.6, "crossed model counts trials, not participants", 5.3)
    fk.t(144, 72.6, "cell-mean model keeps the nominal 5%", 5.3, c=TEAL)
    fk.t(144, 66.2, "dashed line: nominal α = 5%", 5.0, c=GREY, style="italic")

    b = fk.card(2, 2, 104, 57, "b", "C6  Network efficiency under two cleaning variants", PURP, fk.ic_network,
                pad=(12, 13, 3, 2))
    bands = list(cfg.BANDS)
    sets = [("miko", "pipeline", PHYS, "Miko pipeline"), ("miko", "author", AUTH, "Miko author-cleaned"), ("affec", "pipeline", SCREEN, "AFFEC")]
    for k, (ds, ver, col, nm) in enumerate(sets):
        d = top[(top.dataset == ds) & (top.version == ver)].groupby(["subject", "band"]).geff.mean().unstack()[bands]
        for i, bd in enumerate(bands):
            x = i + (k - 1) * 0.27
            v = d[bd].values
            b.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=4, color=col, alpha=0.3, lw=0, zorder=3)
            b.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=3.8, color=col, capsize=1.8,
                       lw=0.9, mec="white", mew=0.4, zorder=4)
        fk.legend_item(16 + k * 30, 6.0, "dot", col, f"{nm} (n = {len(d)})", 5.3)
    b.axhline(0, color=GREY, lw=0.5)
    b.set_xticks(range(3)); b.set_xticklabels([x.capitalize() for x in bands])
    b.set_ylabel("Δ global efficiency (post − baseline)")
    fk.tidy(b)
    pb = cs[(cs.analysis == "primary") & (cs.band == "beta") & (cs.metric == "geff") & (cs.effect == "embodiment")].p.iloc[0]
    pa = cs[(cs.analysis == "miko_author") & (cs.band == "beta") & (cs.metric == "geff") & (cs.effect == "embodiment")].p.iloc[0]
    fk.chip(84, 50.5, f"beta: pipeline {fmt_p(pb)}", PHYS, size=5.3, w=29)
    fk.chip(84, 46.2, f"beta: author {fmt_p(pa)}", AUTH, size=5.3, w=29)

    c = fk.card(110, 2, 68, 57, "c", "Specification curves", GOLD, fk.ic_sliders, pad=(11, 13, 3, 2))
    for comp, off in (("P300", 0.0), ("LPP", 1.2)):
        for lock, col in (("video", SCREEN), ("expression", PURP)):
            d = spec[(spec.component == comp) & (spec.affec_lock == lock)].sort_values("hedges_g")
            x = np.linspace(0, 1, len(d)) + off
            c.plot(x, d.hedges_g, "-", color=col, lw=0.6)
            c.scatter(x, d.hedges_g, s=2.5, color=col, lw=0, zorder=3)
    c.axhline(0, color=GREY, lw=0.5)
    c.set_xticks([0.5, 1.7]); c.set_xticklabels(["P300", "LPP"]); c.set_ylabel("Hedges' g (Miko − AFFEC)")
    fk.tidy(c)
    n_ = spec.groupby(["component", "affec_lock"]).size().iloc[0]
    fk.legend_item(116, 6.0, "line", SCREEN, "video onset", 5.3)
    fk.legend_item(140, 6.0, "line", PURP, "expression onset", 5.3)
    fk.t(174, 48.0, f"{n_} specifications per curve", 5.0, c=GREY, ha="right")
    save("Fig5")


# ================================================================ Fig 6
def fig6():
    val = read("x2c_crossdomain_validity.csv"); geo = read("eye_geometry_kinematics.csv"); mk = read("miko_eye_geometry.csv")
    fk.canvas(180, 116)
    a = fk.card(2, 50, 92, 64, "a", "X2C humanoid controls on human faces", GOLD, fk.ic_transfer, pad=(46, 7.5, 4, 2))
    pr = val[val.test == "a priori pair"].reset_index(drop=True)
    y = np.arange(len(pr))[::-1]
    for yy, (_, r) in zip(y, pr.iterrows()):
        col = SCREEN if r.r >= 0.3 else PHYS
        a.plot([0, r.r], [yy, yy], color=LG, lw=1.6, solid_capstyle="round", zorder=2)
        a.plot(r.r, yy, "o", color=col, ms=4.2, mec="white", mew=0.5, zorder=3)
    a.axvline(0, color=GREY, lw=0.5); a.axvline(0.3, color=INK, lw=0.6, ls="--")
    a.set_yticks(y); a.set_yticklabels([s.replace(" ~ ", " ↔ ") for s in pr.component], fontsize=5.4)
    a.set_xlim(-0.5, 0.8); a.set_xlabel("Frame-wise correlation r (AFFEC)")
    fk.tidy(a, grid=False)
    cc = val[val.test == "CCA out-of-sample"].r.iloc[0]
    fk.t(90.5, 104.8, "criterion r ≥ .30", 5.2, c=INK, style="italic", ha="right")
    fk.t(90.5, 101.6, f"held-out CCA r₁ = {cc:.2f}", 5.2, c=GOLD, ha="right", w="bold")

    fk.card(98, 50, 80, 64, "b", "Miko screen expressions", PHYS, fk.ic_robot, axes=False)
    order = ["neutral", "angry", "happy", "sad", "surprised"]
    mk = mk.set_index("emotion")
    for i, e in enumerate(order):
        col, row = i % 3, i // 3
        x0, y0 = 101 + col * 25.5, 82 - row * 29
        im = Image.open(cfg.ANALYSIS_DIR / "stimuli" / "miko" / f"{e}.png")
        a_ = ax_mm(x0, y0, 22, 20); a_.imshow(im); a_.axis("off")
        v = mk.loc[e, "eye_logratio_peak"]
        fk.chip(x0 + 11, y0 + 23.6, f"{e} {v:+.2f}".replace("-", "−"), PHYS if e != "neutral" else GREY, size=5.4, w=20)
    fk.rbox(152, 55, 23, 22, fc="#FAFBFC", ec=LG, lw=0.4, r=1.0, z=2)
    fk.ic_eye(163.5, 70.5, 3.2, PHYS)
    fk.t(163.5, 63.5, "log(eye area /", 5.0); fk.t(163.5, 60.7, "neutral)", 5.0)
    fk.t(163.5, 57.4, "documented faces", 4.6, c=GREY, style="italic")

    c = fk.card(2, 2, 176, 45, "c", "A shared eye-opening scale for robot and human faces", GOLD, fk.ic_eye, pad=(14, 7.5, 40, 2))
    emos = ["angry", "happy", "sad", "fear", "disgust", "neutral", "surprised"]
    for i, e in enumerate(emos):
        d = geo[(geo.dataset == "affec") & (geo.emotion == e)]
        c.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d.eye_logratio_peak, s=7, color=SCREEN, alpha=0.55, lw=0, zorder=3)
        m = geo[(geo.dataset == "miko") & (geo.emotion == e)]
        c.scatter(np.full(len(m), i), m.eye_logratio_peak, marker="*", s=60, color=PHYS, zorder=4, lw=0.3, ec=INK)
    c.axhline(0, color=GREY, lw=0.5)
    c.set_xticks(range(len(emos))); c.set_xticklabels([e.capitalize() for e in emos])
    c.set_ylabel("Peak log eye-area ratio")
    fk.tidy(c)
    fk.legend_item(146, 32, "dot", SCREEN, "AFFEC clip (n = 84)", 5.6)
    fk.legend_item(146, 27.5, "star", PHYS, "Miko face (n = 4)", 5.6)
    fk.t(146, 19.5, "Miko expressions fall inside", 5.2, ha="left", c=GREY)
    fk.t(146, 16.5, "the range of human clips", 5.2, ha="left", c=GREY)
    save("Fig6")


# ================================================================ Fig 7
def fig7():
    tr = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    stim = read("affec_stim_features.csv"); h4 = read("bb_h4.csv")
    a_ = tr[(tr.dataset == "affec") & (tr.version == "pipeline") & (tr.component == "LPP") &
            (tr.min_shared_trials >= cfg.MIN_TRIALS_SHARED)].copy()
    fv, pv = a_.f_emotion_v - 5, a_.p_emotion_v - 5
    a_["SCI"] = a_.f_emotion_a * ((np.sign(fv) == np.sign(pv)) & ((a_.f_emotion_v - a_.p_emotion_v).abs() <= 2))
    clip = a_.groupby("stim").agg(LPP=("amp_z", "mean"), SCI=("SCI", "mean")).join(stim.set_index("stim")[["bs_amp"]])
    clip["zA"] = (clip.bs_amp - stim.bs_amp.mean()) / stim.bs_amp.std()
    r = h4[(h4.kinematics == "bs") & (h4.dataset == "affec")].iloc[0]
    fk.canvas(180, 116)
    for j, (col_, title, yl, c, icon) in enumerate(
            (("SCI", "Motion amplitude and subjective contagion", "Subjective contagion (clip mean)", GREEN, fk.ic_faceflow),
             ("LPP", "Motion amplitude and LPP", "LPP at Pz/POz (clip mean, SD)", TEAL, fk.ic_wave))):
        ax = fk.card(2 + j * 89, 52, 87, 62, "ab"[j], title, c, icon, pad=(11, 7.5, 3, 2))
        ax.scatter(clip.zA, clip[col_], s=9, color=c, alpha=0.7, lw=0.3, edgecolor="white", zorder=3)
        bb = np.polyfit(clip.zA, clip[col_], 1); xs = np.linspace(clip.zA.min(), clip.zA.max(), 10)
        ax.plot(xs, np.polyval(bb, xs), color=c, lw=1.1)
        ax.set_xlabel("Facial motion amplitude (z, MediaPipe)"); ax.set_ylabel(yl)
        fk.tidy(ax)
        fk.chip(2 + j * 89 + 68, 103.5 if j == 0 else 64.0, f"84 clips · r = {clip.zA.corr(clip[col_]):.2f}", c, size=5.6, w=24)
    for j, (title, lmm, boot, c, unit) in enumerate(
            (("Contagion slope", (r.lmm_sci_beta, r.lmm_sci_lo, r.lmm_sci_hi), (r.beta_subjective, r.bs_ci_lo, r.bs_ci_hi), GREEN,
              "Contagion units per SD of amplitude"),
             ("LPP slope", (r.lmm_lpp_beta, r.lmm_lpp_lo, r.lmm_lpp_hi), (r.beta_neural_LPP, r.bn_ci_lo, r.bn_ci_hi), TEAL,
              "Baseline SD per SD of amplitude"))):
        ax = fk.card(2 + j * 89, 2, 87, 46, "cd"[j], title, c, fk.ic_scale, pad=(28, 7.5, 4, 2))
        for k, (lab, (m, lo, hi), mk_) in enumerate((("Trial-level LMM", lmm, "s"), ("Participant bootstrap", boot, "o"))):
            y = 1 - k
            ax.plot([lo, hi], [y, y], color=c, lw=1.2)
            ax.plot(m, y, mk_, color=c, ms=4.8, mec="white", mew=0.5)
            ax.text(m, y - 0.3, f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "−"), ha="center", va="top", fontsize=5.4, color=c)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["Trial-level LMM", "Participant bootstrap"]); ax.set_ylim(-0.75, 1.5)
        ax.set_xlim(-0.05, 0.42) if j == 0 else ax.set_xlim(-0.11, 0.11)
        ax.set_xlabel(unit)
        fk.tidy(ax, grid=False)
    fk.chip(90.0, 25, "dissociation", INK, size=5.6, w=17, fill=False, bold=True)
    save("Fig7")


# ================================================================ Fig 8
def fig8():
    e = read("eed_summary.csv").set_index("analysis"); con = read("eed_contributions.csv")
    prim = "EED (primary, eye-geometry corrected)"
    fk.canvas(180, 74)
    a = fk.card(2, 2, 82, 70, "a", "Distance and permutation null", GOLD, fk.ic_ellipses, pad=(20, 7.5, 13, 7))
    rows = [("Primary", prim, TEAL), ("Profile", "EED profile (within-participant centred)", PURP),
            ("No beta", "EED without beta-band features", GOLD), ("No correction", "EED (no kinematic correction)", GREY)]
    for i, (lab, key, col) in enumerate(rows):
        r = e.loc[key]; y = len(rows) - 1 - i
        a.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.2)
        a.plot(r.EED, y, "o", color=col, ms=5, mec="white", mew=0.5)
        a.plot(r.null_median, y, marker="|", ms=9, color=INK, mew=1.1)
        a.text(3.05, y, fmt_p(r.p_perm), va="center", fontsize=5.8)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1])
    a.set_xlim(0.8, 3.0); a.set_ylim(-0.6, 3.6); a.set_xlabel("EED (Mahalanobis distance)")
    fk.tidy(a, grid=False)
    fk.legend_item(25, 60.8, "dot", TEAL, "EED, 95% bootstrap interval", 5.3)
    fk.OV.plot([56.5, 56.5], [59.6, 62.0], color=INK, lw=1.1, zorder=12); fk.t(58.4, 60.8, "null median", 5.3, ha="left")
    b = fk.card(88, 2, 90, 70, "b", "What drives the distance", GOLD, fk.ic_bars, pad=(13, 7.5, 30, 3))
    c = con[con.analysis == prim].set_index("feature")[["angry", "happy", "sad"]]
    groups = {"ERP (P300, LPP)": [f for f in c.index if f.startswith("erp")],
              "Module dwPLI": [f for f in c.index if f.startswith("conn")],
              "Theta efficiency": ["geff_theta"], "Alpha efficiency": ["geff_alpha"], "Beta efficiency": ["geff_beta"]}
    cols = [TEAL, PURP, GREEN, GOLD, TERRA]
    left = np.zeros(3)
    for (g, fs), col in zip(groups.items(), cols):
        v = c.loc[fs].sum().values
        b.barh([2, 1, 0], v, left=left, color=col, height=0.58, edgecolor="white", lw=0.4)
        left += v
    for i, emo in enumerate(["angry", "happy", "sad"]):
        b.text(left[i] + 0.06, 2 - i, f"EED = {e.loc[prim, 'EED_' + emo]:.2f}", va="center", fontsize=5.8)
    b.set_yticks([2, 1, 0]); b.set_yticklabels(["Angry", "Happy", "Sad"]); b.set_xlim(0, max(left) * 1.32)
    b.set_xlabel("Contribution to squared EED")
    fk.tidy(b, grid=False)
    for k, (g, col) in enumerate(zip(groups, cols)):
        fk.legend_item(151.5, 58 - k * 4.4, "square", col, g, 5.3)
    share = c.loc[["geff_theta", "geff_beta", "geff_alpha"]].sum().sum() / c.sum().sum()
    fk.chip(163.5, 30, f"efficiency: {100 * share:.0f}% of EED²", TERRA, size=5.3, w=26)
    save("Fig8")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["2", "3", "4", "5", "6", "7", "8"]
    if "2" in which: fig2()
    if "3" in which: fig3(grand_waves())
    if "4" in which: fig4(affec_lock_waves())
    if "5" in which: fig5()
    if "6" in which: fig6()
    if "7" in which: fig7()
    if "8" in which: fig8()
