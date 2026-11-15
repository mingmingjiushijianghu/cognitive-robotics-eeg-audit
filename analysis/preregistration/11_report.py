"""Step 11: figures (design §7 Fig. 2-5) and results/REPORT.md.

Fig 2  grand-average ERPs at the component ROIs (physical vs virtual, shared emotions) + topomaps
Fig 3  Delta dwPLI matrices per band (physical, virtual, difference)
Fig 4  kinematic space (A_eye vs A per clip, Miko points) and the A -> LPP slope
Fig 5  EED per emotion and feature contributions
Each figure is skipped if its inputs are missing, so the report can be rebuilt at any stage.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

import config as C
from common import log

R, F = C.RESULTS_DIR, C.FIG_DIR
COL = {"physical": "#c0392b", "virtual": "#2c7fb8"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def grand(dataset, version):
    ev = []
    for f in sorted((C.DERIV_DIR / "erp").glob(f"{dataset}_{version}_*-ave.fif")):
        e = [x for x in mne.read_evokeds(f, verbose=False) if x.comment in C.SHARED_EMOTIONS]
        if e:
            ev.append(mne.combine_evoked(e, "equal").pick(C.COMMON_CHANNELS))
    return ev


def fig2(av):
    g = {"physical": grand("miko", "pipeline"), "virtual": grand("affec", av)}
    if not all(g.values()):
        return None
    comps = ["N170", "P300", "LPP"]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5))
    for j, comp in enumerate(comps):
        t0, t1, rois = C.COMPONENTS[comp][:3]
        chans = sum(rois.values(), [])
        for k, evs in g.items():
            # baseline-SD units (V4): divide each participant's waveform by its baseline SD
            ys = []
            for e in evs:
                d = e.copy().pick(chans).data
                sd = e.copy().crop(*C.BASELINE).pick(chans).data.std()
                ys.append(d.mean(0) / sd)
            ys = np.array(ys)
            m, se = ys.mean(0), ys.std(0, ddof=1) / np.sqrt(len(ys))
            ax[0, j].plot(evs[0].times * 1000, m, color=COL[k], label=f"{k} (n={len(ys)})")
            ax[0, j].fill_between(evs[0].times * 1000, m - se, m + se, color=COL[k], alpha=.2, lw=0)
        ax[0, j].axvspan(t0 * 1000, t1 * 1000, color="0.9", zorder=0)
        ax[0, j].set(title=f"{comp} ({'/'.join(chans)})", xlabel="ms", ylabel="baseline-SD units", xlim=(-200, 1000))
        for i, (k, evs) in enumerate(g.items()):
            pass
    ax[0, 0].legend(frameon=False)
    for j, comp in enumerate(comps):
        t0, t1 = C.COMPONENTS[comp][:2]
        diff = []
        for k in ("physical", "virtual"):
            e = mne.grand_average(g[k]).crop(t0, t1)
            v = e.data.mean(1)
            diff.append(v / np.sqrt(np.mean((v - v.mean()) ** 2)))
        info = g["physical"][0].info
        mne.viz.plot_topomap(diff[0] - diff[1], info, axes=ax[1, j], show=False, cmap="RdBu_r")
        ax[1, j].set_title(f"{comp}: GFP-normalised map, physical − virtual", fontsize=8)
    fig.tight_layout()
    fig.savefig(F / "fig2_erp.png", dpi=200); plt.close(fig)
    return "fig2_erp.png"


def fig3():
    files = sorted((C.DERIV_DIR / "conn").glob("*_pipeline_*.npz"))
    if not files:
        return None
    acc = {}
    for f in files:
        ds = f.name.split("_")[0]
        z = np.load(f)
        for b in C.BANDS:
            keys = [k for k in z.files if f"|{b}|" in k]
            if not keys:
                continue
            emos = {k.split("|")[0] for k in keys}
            d = np.mean([z[f"{e}|{b}|post"] - z[f"{e}|{b}|base"] for e in emos], 0)
            acc.setdefault((ds, b), []).append(d)
    fig, ax = plt.subplots(3, 3, figsize=(10, 9.5))
    for i, b in enumerate(C.BANDS):
        p, v = np.mean(acc.get(("miko", b), [np.zeros((16, 16))]), 0), np.mean(acc.get(("affec", b), [np.zeros((16, 16))]), 0)
        lim = np.nanmax(np.abs([p, v])) or 1
        for j, (m, t) in enumerate(((p, "physical"), (v, "virtual"), (p - v, "physical − virtual"))):
            im = ax[i, j].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i, j].set(title=f"{b}: {t}", xticks=range(16), yticks=range(16))
            ax[i, j].set_xticklabels(C.COMMON_CHANNELS, rotation=90, fontsize=6)
            ax[i, j].set_yticklabels(C.COMMON_CHANNELS, fontsize=6)
        fig.colorbar(im, ax=ax[i, :].tolist(), shrink=.7, label="Δ dwPLI (post − base)")
    fig.savefig(F / "fig3_connectivity.png", dpi=200); plt.close(fig)
    return "fig3_connectivity.png"


def fig4():
    p = R / "stim_kinematics.csv"
    if not p.exists():
        return None
    k = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    a, m = k[k.dataset == "affec"], k[k.dataset == "miko"]
    for emo, d in a.groupby("emotion"):
        ax[0].scatter(d.A, d.A_eye, s=14, label=emo, alpha=.8)
    ax[0].scatter(m.A, m.A_eye, marker="*", s=120, color="k", label="Miko")
    for _, r in m.iterrows():
        ax[0].annotate(r.emotion, (r.A, r.A_eye), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set(xlabel="motion amplitude A (23 expressive controls)", ylabel="A_eye (10 eye-region controls)")
    ax[0].legend(fontsize=6, frameon=False, ncol=2)
    t = R / "erp_trials.csv"
    if t.exists():
        tr = pd.read_csv(t, low_memory=False)
        tr = tr[(tr.dataset == "affec") & (tr.version == "pipeline") & (tr.component == "LPP")]
        cm = tr.groupby("stim").amp_z.mean().rename("LPP").to_frame().join(a.set_index("stim")[["A", "emotion"]]).dropna()
        ax[1].scatter(cm.A, cm.LPP, s=14, color=COL["virtual"])
        b = np.polyfit(cm.A, cm.LPP, 1)
        xs = np.linspace(cm.A.min(), cm.A.max(), 10)
        ax[1].plot(xs, np.polyval(b, xs), color=COL["virtual"])
        ax[1].set(xlabel="motion amplitude A", ylabel="LPP, clip mean (baseline-SD units)",
                  title=f"AFFEC clips: r = {np.corrcoef(cm.A, cm.LPP)[0, 1]:.2f}")
    fig.tight_layout(); fig.savefig(F / "fig4_kinematics.png", dpi=200); plt.close(fig)
    return "fig4_kinematics.png"


def fig5():
    p, c = R / "eed_summary.csv", R / "eed_contributions.csv"
    if not p.exists():
        return None
    s, con = pd.read_csv(p), pd.read_csv(c)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    row = s.iloc[0]
    ax[0].bar(C.SHARED_EMOTIONS, [row[f"EED_{e}"] for e in C.SHARED_EMOTIONS], color="0.4")
    ax[0].axhline(row.null_median, ls="--", color="k", lw=.8, label="permutation null median")
    ax[0].set(ylabel="EED (Mahalanobis)", title=f"EED = {row.EED:.2f}, p_perm = {row.p_perm:.4f}")
    ax[0].legend(frameon=False)
    cc = con[con.analysis == row.analysis].set_index("feature")[C.SHARED_EMOTIONS]
    cc.plot.barh(ax=ax[1], width=.8)
    ax[1].set(xlabel="contribution to EED² (d_k·(Σ⁻¹d)_k)")
    fig.tight_layout(); fig.savefig(F / "fig5_eed.png", dpi=200); plt.close(fig)
    return "fig5_eed.png"


def main():
    dec = R / "expr_lock_decision.json"
    lock = json.loads(dec.read_text()) if dec.exists() else {}
    av = "expr" if lock.get("primary_lock") == "expression" else "pipeline"
    figs = [f for f in (fig2(av), fig3(), fig4(), fig5()) if f]
    parts = ["# 实验结果汇总（自动生成）\n", f"AFFEC 主分析锁定方式：`{av}`；判定依据：{lock}\n"]
    for name in ("miko_preprocessing_qc.csv", "affec_preprocessing_qc.csv"):
        p = R / name
        if p.exists():
            q = pd.read_csv(p)
            num = q.select_dtypes("number")
            parts.append(f"## 预处理 QC：{name}\n\n{num.describe().T[['mean', 'min', 'max']].round(2).to_markdown()}\n")
    for md in ("erp_stats_summary.md", "bb_summary.md", "eed_summary.md"):
        p = R / md
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    p = R / "conn_stats.csv"
    if p.exists():
        cs = pd.read_csv(p)
        parts.append("# 路线 2：连通性（主分析，按 p 排序前 15）\n\n" +
                     cs[cs.analysis == "primary"].sort_values("p").head(15).round(4).to_markdown(index=False) + "\n")
    parts.append("## 图\n\n" + "\n".join(f"![{f}](../figures/{f})" for f in figs) + "\n")
    (R / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    log("report written with figures:", figs)


if __name__ == "__main__":
    main()
