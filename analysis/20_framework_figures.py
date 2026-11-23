"""Step 20: figures and numbers for the framework-structured manuscript (v3).

Fig 1  research framework: physical vs screen-mediated embodiment, three routes, EED, validity layer
Fig 2  route 1: participant amplitudes, 50 % fractional-area latencies and GFP-normalised topographies
Fig 8  embodied emotional distance (EED) per emotion and feature contributions
Also results/framework_numbers.json with the values the text quotes that are not in numbers.tex.
Style matches the co-author's redrawn figures (Times New Roman, same palette, no in-figure titles).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mne
from scipy import stats

import config as cfg

R = cfg.RESULTS_DIR
OUT = cfg.ROOT / "submission" / "figures_v3" if hasattr(cfg, "ROOT") else None
OUT = (R.parent.parent / "submission" / "figures_v3")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#5b6770",
                     "xtick.color": "#2b333a", "ytick.color": "#2b333a", "axes.labelcolor": "#2b333a",
                     "mathtext.fontset": "stix"})
TEAL, TERRA, GOLD, PURP, GREEN, INK, GREY = "#1F7A8C", "#C0694E", "#B8954A", "#6E6AA6", "#3F7F5F", "#2B333A", "#8A949C"
MM = 1 / 25.4
W2 = 174 * MM
DPI = 600
NUM = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", name)


def label(ax, s, x=-0.14, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)


# ----------------------------------------------------------------------------------------- Fig 1
def box(ax, x, y, w, h, text, fc, ec, fs=7.5, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8))
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, c=GREY):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=0, shrinkB=0))


def fig1():
    fig = plt.figure(figsize=(W2, 104 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.02, 0.86, 0.96, 0.11,
        "Core question: does the embodiment form of a robot (physical body vs screen-mediated, virtual presentation)\n"
        "modulate EEG responses to its emotional expressions, and do subjective contagion and neural response strength diverge?",
        "#EEF2F5", "#9AA7B0", fs=8)
    # data layer
    y = 0.68; h = 0.13
    box(ax, 0.02, y, 0.30, h, "Physical embodiment\nMiko: co-present robot, screen eyes\n10 participants, 16-channel EEG",
        "#F6E3DC", TERRA)
    box(ax, 0.35, y, 0.30, h, "Screen-mediated (virtual) embodiment\nAFFEC: actor videos on a monitor\n71 participants, 63-channel EEG, ratings",
        "#DCEEF1", TEAL)
    box(ax, 0.68, y, 0.30, h, "Behavioral annotation\nX2C: 30 humanoid expression controls\nMediaPipe: facial motion of AFFEC clips",
        "#EFEAD9", GOLD)
    for x in (0.17, 0.50, 0.83):
        arrow(ax, x, 0.86, x, y + h + 0.005)
    # unified preprocessing
    box(ax, 0.02, 0.555, 0.96, 0.075, "Unified preprocessing: shared 16-channel montage, 128 Hz, ICA blink removal, autoreject, "
        "amplitudes in baseline-SD units", "#F4F6F7", "#9AA7B0", fs=7.5)
    for x in (0.17, 0.50, 0.83):
        arrow(ax, x, y, x, 0.63)
    # routes
    y2, h2 = 0.29, 0.22
    routes = [("Route 1  Neural response", "N170 (face encoding)\nP300 (motivated attention)\nLPP (sustained affect)\n"
               "amplitude · latency · topography", "#DCEEF1", TEAL),
              ("Route 2  Brain network", "frontal–parietal–temporal\ndwPLI (theta, alpha, beta)\n"
               "efficiency · clustering · modularity\nnetwork-based statistic", "#E6E5F2", PURP),
              ("Route 3  Behavior–EEG coupling", "motion amplitude · speed · symmetry\n→ ERP, connectivity, contagion\n"
               "subjective contagion vs LPP\nneural markers of contagion", "#E2EFE7", GREEN)]
    for i, (t, b, fc, ec) in enumerate(routes):
        x = 0.02 + i * 0.33
        box(ax, x, y2, 0.30, h2, "", fc, ec)
        ax.text(x + 0.15, y2 + h2 - 0.035, t, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + 0.15, y2 + 0.085, b, ha="center", va="center", fontsize=7, color=INK, linespacing=1.3)
        arrow(ax, x + 0.15, 0.555, x + 0.15, y2 + h2 + 0.005)
    # EED
    box(ax, 0.02, 0.135, 0.63, 0.105, "Embodied emotional distance (EED): Mahalanobis distance between physical and virtual\n"
        "neural profiles (P300, LPP, module dwPLI, global efficiency), corrected for eye-opening amplitude",
        "#F6EEDA", GOLD, fs=7.5)
    for x in (0.17, 0.50):
        arrow(ax, x, y2, x, 0.24)
    box(ax, 0.68, 0.135, 0.30, 0.105, "Neural markers of embodied\nemotional contagion\n(slope, stability, physical gain)",
        "#E2EFE7", GREEN, fs=7.5)
    arrow(ax, 0.83, y2, 0.83, 0.24)
    # validity layer
    box(ax, 0.02, 0.015, 0.96, 0.085,
        "Validity checks before cross-form inference\nC1 event timing · C2 event type · C3 auditory overlap · C4 montage · "
        "C5 kinematic transfer · C6 preprocessing · C7 model calibration", "#FBF3F0", TERRA, fs=7.5)
    for x in (0.33, 0.83):
        arrow(ax, x, 0.135, x, 0.10)
    save(fig, "Fig1")


# ----------------------------------------------------------------------------------------- Fig 2
def included(trials):
    ok = trials[trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED]
    return {ds: set(ok[(ok.dataset == ds) & (ok.version == "pipeline")].subject) for ds in ("miko", "affec")}


def fig2():
    trials = pd.read_csv(R / "erp_trials.csv", low_memory=False)
    inc = included(trials)
    t = trials[(trials.version == "pipeline") & trials.emotion.isin(cfg.SHARED_EMOTIONS) &
               (trials.min_shared_trials >= cfg.MIN_TRIALS_SHARED) & trials.component.isin(["N170", "P300", "LPP"])]
    subj = t.groupby(["component", "dataset", "subject"]).amp_z.mean()
    eq = pd.read_csv(R / "erp_stats_equivalence.csv").set_index("component")
    tests = pd.read_csv(R / "erp_stats_tests.csv")
    tests = tests[tests.analysis == "primary"].set_index(["component", "measure", "effect"])
    lat = pd.read_csv(R / "erp_subject_latency.csv")
    lat = lat[(lat.version == "pipeline") & lat.emotion.isin(cfg.SHARED_EMOTIONS)]
    lat = lat[lat.apply(lambda r: r.subject in inc[r.dataset], axis=1)]
    lsub = lat.groupby(["component", "dataset", "subject"]).latency.mean() * 1000
    tan = pd.read_csv(R / "erp_stats_tanova.csv").set_index("component")

    fig = plt.figure(figsize=(W2, 128 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.32)
    rng = np.random.default_rng(3)
    comps = ["N170", "P300", "LPP"]
    for k, (title, data, ylab, key) in enumerate(
            [("Amplitude", subj, "Mean amplitude (baseline SD)", "amplitude"),
             ("Latency", lsub, "50% fractional-area latency (ms)", "latency")]):
        ax = fig.add_subplot(gs[0, k]); label(ax, "ab"[k])
        for i, c in enumerate(comps):
            for j, (ds, col) in enumerate((("miko", TERRA), ("affec", TEAL))):
                v = data.loc[(c, ds)].values
                x = i + (j - 0.5) * 0.36
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=7, color=col, alpha=0.35, lw=0)
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=4.5, color=col,
                            capsize=2.5, lw=1, mec="white", mew=0.5)
            p = tests.loc[(c, key, "embodiment"), "p"]; pf = tests.loc[(c, key, "embodiment"), "p_fdr"]
            txt = f"p = {p:.3f}".replace("0.", ".") + f"\nFDR {pf:.2f}".replace("0.", ".")
            if key == "amplitude":
                txt = f"g = {eq.loc[c, 'hedges_g']:.2f}\n" + txt
            ax.text(i, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5,
                    color=GREY if c == "N170" else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([c + ("*" if c == "N170" else "") for c in comps])
        ax.set_ylabel(ylab)
        if key == "amplitude":
            ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xlim(-0.6, 2.6)
    h = [plt.Line2D([], [], marker="o", ls="", color=TERRA, label="Physical (Miko, n = 10)"),
         plt.Line2D([], [], marker="o", ls="", color=TEAL, label="Screen-mediated (AFFEC, n = 49)")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)

    # topographies
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
    sub = gs[1, :].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.15, hspace=0.25)
    for i, c in enumerate(comps):
        t0, t1 = cfg.COMPONENTS[c][:2]
        for j, ds in enumerate(("miko", "affec")):
            X = []
            for e in maps[ds]:
                m = e.copy().crop(t0, t1).data.mean(1); m = m - m.mean(); X.append(m / np.sqrt(np.mean(m ** 2)))
            m = np.mean(X, 0); m = m / np.sqrt(np.mean(m ** 2))
            ax = fig.add_subplot(sub[j, i])
            im, _ = mne.viz.plot_topomap(m, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-2, 2), contours=4,
                                         sensors=True, sphere="auto")
            if j == 0:
                ax.set_title(f"{c} ({t0*1000:.0f}–{t1*1000:.0f} ms)\nTANOVA p = {tan.loc[c, 'p_perm']:.3f}".replace("0.", "."),
                             fontsize=7.5, color=INK)
            if i == 0:
                ax.text(-0.25, 0.5, ["Physical", "Screen"][j], transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=7.5, color=[TERRA, TEAL][j], fontweight="bold")
            if i == 0 and j == 0:
                label(ax, "c", x=-0.45, y=1.12)
    cax = fig.add_subplot(sub[:, 3]); cb = fig.colorbar(im, cax=cax); cb.set_label("GFP-normalized amplitude", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "Fig2")
    # numbers for the text
    for c in comps:
        for ds in ("miko", "affec"):
            NUM[f"lat_{c}_{ds}"] = round(float(lsub.loc[(c, ds)].mean()), 0)
        NUM[f"lat_{c}_p"] = float(tests.loc[(c, "latency", "embodiment"), "p"])
        NUM[f"lat_{c}_pfdr"] = float(tests.loc[(c, "latency", "embodiment"), "p_fdr"])
        NUM[f"lat_{c}_chi2"] = float(tests.loc[(c, "latency", "embodiment"), "chi2"])
        NUM[f"amp_{c}_p"] = float(tests.loc[(c, "amplitude", "embodiment"), "p"])
        NUM[f"amp_{c}_pfdr"] = float(tests.loc[(c, "amplitude", "embodiment"), "p_fdr"])
        NUM[f"amp_{c}_int_p"] = float(tests.loc[(c, "amplitude", "embodiment x emotion"), "p"])
        NUM[f"tanova_{c}"] = float(tan.loc[c, "p_perm"])
        NUM[f"g_{c}"] = float(eq.loc[c, "hedges_g"]); NUM[f"bf01_{c}"] = float(eq.loc[c, "BF01"])
        NUM[f"tost_{c}"] = float(eq.loc[c, "tost_p"])


# ----------------------------------------------------------------------------------------- Fig 8
def fig8():
    e = pd.read_csv(R / "eed_summary.csv").set_index("analysis")
    con = pd.read_csv(R / "eed_contributions.csv")
    prim = "EED (primary, eye-geometry corrected)"
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.45))
    label(a, "a"); label(b, "b")
    rows = [("Primary", prim, TEAL), ("Profile", "EED profile (within-participant centred)", PURP),
            ("No beta", "EED without beta-band features", GOLD), ("No correction", "EED (no kinematic correction)", GREY)]
    for i, (lab, key, col) in enumerate(rows):
        r = e.loc[key]
        a.errorbar(r.EED, i, xerr=[[r.EED - r.ci_lo], [r.ci_hi - r.EED]], fmt="o", color=col, capsize=2.5, ms=5, lw=1)
        a.plot(r.null_median, i, marker="|", ms=10, color=INK, mew=1.2)
        a.text(3.05, i, f"p = {r.p_perm:.3f}".replace("0.", "."), va="center", fontsize=7, color=INK)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows]); a.invert_yaxis()
    a.set_xlim(0.8, 3.0); a.set_xlabel("EED (Mahalanobis distance)")
    a.plot([], [], marker="|", ls="", color=INK, ms=8, label="permutation null median")
    a.plot([], [], marker="o", ls="-", color=GREY, ms=4, label="EED, 95% bootstrap interval")
    a.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.55, 1.04), ncol=2)
    c = con[con.analysis == prim].set_index("feature")[["angry", "happy", "sad"]]
    groups = {"ERP (P300, LPP)": [f for f in c.index if f.startswith("erp")],
              "Module dwPLI": [f for f in c.index if f.startswith("conn")],
              "Theta efficiency": ["geff_theta"], "Alpha efficiency": ["geff_alpha"], "Beta efficiency": ["geff_beta"]}
    cols = [TEAL, PURP, GREEN, GOLD, TERRA]
    left = np.zeros(3)
    for (g, fs), col in zip(groups.items(), cols):
        v = c.loc[fs].sum().values
        b.barh(range(3), v, left=left, color=col, height=0.55, label=g, edgecolor="white", lw=0.4)
        left += v
    for i, emo in enumerate(["angry", "happy", "sad"]):
        b.text(left[i] + 0.05, i, f"EED = {e.loc[prim, 'EED_' + emo]:.2f}", va="center", fontsize=7)
    b.set_yticks(range(3)); b.set_yticklabels(["Angry", "Happy", "Sad"]); b.invert_yaxis()
    b.set_xlabel("Contribution to squared EED"); b.set_xlim(0, max(left) * 1.3)
    b.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1)
    save(fig, "Fig8")
    NUM["eed_contrib_share_geff"] = float(c.loc[["geff_theta", "geff_beta", "geff_alpha"]].sum().sum() / c.sum().sum())


def extra_numbers():
    cs = pd.read_csv(R / "conn_stats.csv")
    def g(an, band, metric, eff):
        r = cs[(cs.analysis == an) & (cs.band == band) & (cs.metric == metric) & (cs.effect == eff)].iloc[0]
        return dict(chi2=float(r.chi2), df=int(r.df), p=float(r.p), p_fdr=None if pd.isna(r.p_fdr) else float(r.p_fdr))
    NUM["theta_geff_int_primary"] = g("primary", "theta", "geff", "embodiment x emotion")
    NUM["theta_geff_int_author"] = g("miko_author", "theta", "geff", "embodiment x emotion")
    NUM["beta_geff_primary"] = g("primary", "beta", "geff", "embodiment")
    NUM["beta_geff_author"] = g("miko_author", "beta", "geff", "embodiment")
    top = pd.read_csv(R / "conn_topology.csv")
    th = top[(top.band == "theta") & top.version.isin(["pipeline", "author"])]
    NUM["theta_geff_means"] = {f"{d}_{v}_{e}": round(float(x), 4) for (d, v, e), x in
                               th.groupby(["dataset", "version", "emotion"]).geff.mean().items()}
    nbs = pd.read_csv(R / "conn_nbs.csv"); NUM["nbs_min_p"] = float(nbs.p.min()); NUM["nbs_edges_max"] = int(nbs.n_edges.max())
    sl = pd.read_csv(R / "bb_slopes.csv")
    s = sl[sl.kinematics == "bs"]
    NUM["route3_n_slopes"] = int(len(s)); NUM["route3_min_pfdr"] = float(s.p_fdr.min())
    NUM["route3_min_p"] = float(s.p.min())
    NUM["route3_min_p_term"] = s.loc[s.p.idxmin(), ["feature", "predictor"]].tolist()
    for f in ("SCI", "LPP", "P300"):
        for pr in ("zA", "zV", "zS"):
            r = s[(s.feature == f) & (s.predictor == pr)].iloc[0]
            NUM[f"slope_{f}_{pr}"] = dict(beta=float(r.beta), se=float(r.se), p=float(r.p), p_fdr=float(r.p_fdr))
    pg_ = pd.read_csv(R / "bb_physical_gain.csv").set_index("feature")
    for f in ("P300", "LPP"):
        r = pg_.loc[f]
        NUM[f"gain_{f}"] = dict(gain=float(r.physical_gain), lo=float(r.ci_lo), hi=float(r.ci_hi), p=float(r.p_boot),
                                slope=float(r.slope_A_geo))
    mk = pd.read_csv(R / "bb_markers.csv"); NUM["n_markers"] = int(mk.is_marker.sum())
    lo = pd.read_csv(R / "bb_loso.csv"); NUM["loso"] = {f"{k}_{f}": float(v) for k, f, v in lo.values}
    m = pd.read_csv(R / "erp_stats_matched.csv").set_index("component")
    NUM["matched"] = {c: dict(g=float(m.loc[c, "g_median"]), lo=float(m.loc[c, "g_p2_5"]), hi=float(m.loc[c, "g_p97_5"]))
                      for c in m.index}


if __name__ == "__main__":
    fig1(); fig2(); fig8(); extra_numbers()
    (R / "framework_numbers.json").write_text(json.dumps(NUM, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(NUM, indent=1, ensure_ascii=False))
