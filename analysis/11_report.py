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

import config as cfg
from common import log

R, F = cfg.RESULTS_DIR, cfg.FIG_DIR
COL = {"physical": "#c0392b", "virtual": "#2c7fb8"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def grand(dataset, version):
    ev = []
    for f in sorted((cfg.DERIV_DIR / "erp").glob(f"{dataset}_{version}_*-ave.fif")):
        e = [x for x in mne.read_evokeds(f, verbose=False) if x.comment in cfg.SHARED_EMOTIONS]
        if e:
            ev.append(mne.combine_evoked(e, "equal").pick(cfg.COMMON_CHANNELS))
    return ev


def fig2(av):
    g = {"physical": grand("miko", "pipeline"), "virtual": grand("affec", av)}
    if not all(g.values()):
        return None
    comps = ["N170", "P300", "LPP"]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5))
    for j, comp in enumerate(comps):
        t0, t1, rois = cfg.COMPONENTS[comp][:3]
        chans = sum(rois.values(), [])
        for k, evs in g.items():
            # baseline-SD units (V4): divide each participant's waveform by its baseline SD
            ys = []
            for e in evs:
                d = e.copy().pick(chans).data
                sd = e.copy().crop(*cfg.BASELINE).pick(chans).data.std()
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
        t0, t1 = cfg.COMPONENTS[comp][:2]
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
    files = sorted((cfg.DERIV_DIR / "conn").glob("*_pipeline_*.npz"))
    if not files:
        return None
    acc = {}
    for f in files:
        ds = f.name.split("_")[0]
        z = np.load(f)
        for b in cfg.BANDS:
            keys = [k for k in z.files if f"|{b}|" in k]
            if not keys:
                continue
            emos = {k.split("|")[0] for k in keys}
            d = np.mean([z[f"{e}|{b}|post"] - z[f"{e}|{b}|base"] for e in emos], 0)
            acc.setdefault((ds, b), []).append(d)
    fig, ax = plt.subplots(3, 3, figsize=(10, 9.5))
    for i, b in enumerate(cfg.BANDS):
        p, v = np.mean(acc.get(("miko", b), [np.zeros((16, 16))]), 0), np.mean(acc.get(("affec", b), [np.zeros((16, 16))]), 0)
        lim = np.nanmax(np.abs([p, v])) or 1
        for j, (m, t) in enumerate(((p, "physical"), (v, "virtual"), (p - v, "physical − virtual"))):
            im = ax[i, j].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[i, j].set(title=f"{b}: {t}", xticks=range(16), yticks=range(16))
            ax[i, j].set_xticklabels(cfg.COMMON_CHANNELS, rotation=90, fontsize=6)
            ax[i, j].set_yticklabels(cfg.COMMON_CHANNELS, fontsize=6)
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
    # V17: objective eye-opening change (signed log ratio to neutral), AFFEC clips vs Miko faces
    g = pd.read_csv(R / "eye_geometry_kinematics.csv")
    emos = sorted(g.emotion.unique())
    for i, emo in enumerate(emos):
        d = g[(g.dataset == "affec") & (g.emotion == emo)]
        ax[0].scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-.15, .15, len(d)),
                      d.eye_logratio_peak, s=12, color=COL["virtual"], alpha=.6)
        dm = g[(g.dataset == "miko") & (g.emotion == emo)]
        ax[0].scatter(np.full(len(dm), i), dm.eye_logratio_peak, marker="*", s=160, color=COL["physical"])
    ax[0].axhline(0, color="0.6", lw=.6)
    ax[0].set(xticks=range(len(emos)), xticklabels=emos, ylabel="eye opening, log(area / neutral)",
              title="dots: AFFEC clips; stars: Miko faces")
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
    ax[0].bar(cfg.SHARED_EMOTIONS, [row[f"EED_{e}"] for e in cfg.SHARED_EMOTIONS], color="0.4")
    ax[0].axhline(row.null_median, ls="--", color="k", lw=.8, label="permutation null median")
    ax[0].set(ylabel="EED (Mahalanobis)", title=f"EED = {row.EED:.2f}, p_perm = {row.p_perm:.4f}")
    ax[0].legend(frameon=False)
    cc = con[con.analysis == row.analysis].set_index("feature")[cfg.SHARED_EMOTIONS]
    cc.plot.barh(ax=ax[1], width=.8)
    ax[1].set(xlabel="contribution to EED² (d_k·(Σ⁻¹d)_k)")
    fig.tight_layout(); fig.savefig(F / "fig5_eed.png", dpi=200); plt.close(fig)
    return "fig5_eed.png"


def _read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def _fmt_p(p):
    return "—" if p is None or not np.isfinite(p) else (f"{p:.3f}" if p >= .001 else "<.001")


def framework_gap():
    """V18: hypothesis-by-hypothesis comparison with the framework, robustness and literature."""
    rows = []
    tests, val, eq = _read("erp_stats_tests.csv"), _read("erp_stats_validity.csv"), _read("erp_stats_equivalence.csv")
    rob, clus = _read("robust_summary.csv"), _read("robust_clusters.csv")
    if tests is not None:
        def tp(analysis, comp, eff="embodiment"):
            r = tests[(tests.analysis == analysis) & (tests.component == comp) & (tests.measure == "amplitude") & (tests.effect == eff)]
            return float(r.p.iloc[0]) if len(r) else np.nan
        ok = val.set_index("component").valid_16ch.to_dict() if val is not None else {}
        rows.append(dict(hyp="H1a N170 不受具身形式影响", expected="无差异（CogSci 2024：真人/虚拟脸 N170 无差异）",
                         observed=f"16 导 N170 未通过 §3.3 验证（valid={ok.get('N170')}）",
                         robustness="—", status="无法检验（V15）"))
        for comp, lit in (("P300", "iCub 2026：实体机器人 P3 受表情调制"), ("LPP", "Schindler 2017/CogSci 2024：真人脸 LPP > 虚拟脸")):
            p = tp("primary", comp)
            e = eq[eq.component == comp].iloc[0] if eq is not None else None
            direction = "实体 > 虚拟" if e is not None and e.mean_phys > e.mean_virt else "实体 < 虚拟"
            sens = [tp(a, comp) for a in ("miko_author", "min_trials_3", "min_trials_8", "affec_expr_locked", "affec_speech_after_200ms")]
            r = rob[rob.component == comp].iloc[0] if rob is not None and (rob.component == comp).any() else None
            rob_txt = (f"敏感性分析 p<.05 的比例 {np.mean([x < .05 for x in sens if np.isfinite(x)]):.0%}"
                       + (f"；多元分析 {r.n_specs} 种组合中 {r.frac_p05:.0%} 显著，联合置换 p = {_fmt_p(r.p_joint_perm)}" if r is not None else ""))
            supported = p < .05 and direction == "实体 > 虚拟" and (r is None or r.p_joint_perm < .05)
            rows.append(dict(hyp=f"H1 {comp} 实体端更强", expected=f"实体 > 虚拟（{lit}）",
                             observed=f"{direction}，g = {e.hedges_g:.2f}，p = {_fmt_p(p)}；TOST p = {_fmt_p(e.tost_p)}，BF01 = {e.BF01:.2f}" if e is not None else _fmt_p(p),
                             robustness=rob_txt, status="支持" if supported else ("方向一致但不显著" if direction == "实体 > 虚拟" else "不支持")))
    cs = _read("conn_stats.csv")
    if cs is not None:
        e = cs[(cs.effect == "embodiment") & cs.metric.isin(["F-P", "geff"])]
        prim = e[e.analysis == "primary"].sort_values("p")
        best = prim.iloc[0]
        auth = e[(e.analysis == "miko_author") & (e.band == best.band) & (e.metric == best.metric)]
        robust = len(auth) and auth.p.iloc[0] < .05 and np.sign(auth.beta_virtual_minus_physical.iloc[0]) == np.sign(best.beta_virtual_minus_physical)
        rows.append(dict(hyp="H2 实体端额–顶连接更强、全局效率更高", expected="实体 > 虚拟",
                         observed=f"最强：{best.band} {best.metric}，虚拟−实体 β = {best.beta_virtual_minus_physical:.4f}，p = {_fmt_p(best.p)}（FDR {_fmt_p(best.p_fdr)}）",
                         robustness=f"作者清洗版 Miko：p = {_fmt_p(auth.p.iloc[0]) if len(auth) else '—'}",
                         status="支持" if (best.p_fdr < .05 and best.beta_virtual_minus_physical < 0 and robust) else "不稳健/不支持"))
    g = _read("bb_physical_gain.csv")
    if g is not None and "physical_gain" in g:
        r = g[g.feature == "LPP"].iloc[0]
        rows.append(dict(hyp="H3 运动学–神经耦合受具身调制（实体增益）", expected="实体增益 > 0",
                         observed=f"LPP 实体增益 = {r.physical_gain:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]，p = {_fmt_p(r.p_boot)}；AFFEC 内斜率 = {r.slope_A_geo:.2f}",
                         robustness=f"Miko 取值在 AFFEC 范围内：{r.miko_inside_affec_range}",
                         status="支持" if r.p_boot < .05 and r.physical_gain > 0 else "不支持"))
    h4 = _read("bb_h4.csv")
    if h4 is not None:
        r = h4[(h4.kinematics == "bs") & (h4.dataset == "affec")].iloc[0]
        dissoc = (r.bs_ci_lo > 0 or r.bs_ci_hi < 0) and (r.bn_ci_lo < 0 < r.bn_ci_hi)
        rows.append(dict(hyp="H4 主观感染力与神经响应分离", expected="两者斜率之比因具身形式而异",
                         observed=f"AFFEC：主观斜率 {r.beta_subjective:.2f} [{r.bs_ci_lo:.2f}, {r.bs_ci_hi:.2f}]，LPP 斜率 {r.beta_neural_LPP:.3f} [{r.bn_ci_lo:.3f}, {r.bn_ci_hi:.3f}]",
                         robustness="Miko 无逐试次评分（V11）", status="虚拟端内部出现分离；跨形态比较无法检验" if dissoc else "未见分离"))
    ed = _read("eed_summary.csv")
    if ed is not None:
        pr = ed.iloc[0]
        others = "；".join(f"{a.split('(')[0].strip() or a}: p = {_fmt_p(p)}" for a, p in zip(ed.analysis[1:], ed.p_perm[1:]))
        rows.append(dict(hyp="创新点 EED > 0", expected="具身情感距离显著大于置换零分布",
                         observed=f"EED = {pr.EED:.2f}，置换 p = {_fmt_p(pr.p_perm)}", robustness=others,
                         status="支持" if pr.p_perm < .05 and (ed.p_perm < .05).all() else ("主分析显著但不稳健" if pr.p_perm < .05 else "不支持")))
    if clus is not None and "affec_lock" in clus:
        note = {"video": "面孔出现 vs 表情变化，存在事件类型混淆（V2），不能解释为具身效应",
                "expression": "两端事件对齐（预先规划的敏感性分析）；额极簇可能含眼动差异"}
        for lock, d in clus.groupby("affec_lock"):
            sig = d[d.p < .05].sort_values("p")
            obs = "；".join(f"{c.t_start*1000:.0f}–{c.t_end*1000:.0f} ms {c.channels} {c.direction}（p = {_fmt_p(c.p)}）"
                           for c in sig.itertuples()) or "无显著簇"
            rows.append(dict(hyp=f"补充：时空簇检验（AFFEC 按{'视频' if lock == 'video' else '表情运动'}起点锁定）",
                             expected="（无预设窗口）", observed=obs,
                             robustness="2,000 次置换，已控制全时程多重比较；" + note[lock],
                             status="有显著簇" if len(sig) else "无显著簇"))
    df = pd.DataFrame(rows)
    df.to_csv(R / "framework_gap.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    dec = R / "expr_lock_decision.json"
    lock = json.loads(dec.read_text()) if dec.exists() else {}
    av = "expr" if lock.get("primary_lock") == "expression" else "pipeline"
    figs = [f for f in (fig2(av), fig3(), fig4(), fig5()) if f]
    if (F / "fig6_spec_curve.png").exists():
        figs.append("fig6_spec_curve.png")
    parts = ["# 实验结果汇总（自动生成）\n", f"AFFEC 主分析锁定方式：`{av}`；判定依据：{lock}\n"]
    gap = framework_gap()
    if len(gap):
        gap.columns = ["假设", "框架预期（文献）", "实际结果", "稳健性", "判定"]
        parts.append("## 与框架预期对照（V18）\n\n" + gap.to_markdown(index=False) + "\n")
    p = R / "robust_summary.csv"
    if p.exists():
        parts.append("## 多元分析汇总（12 步）\n\n" + pd.read_csv(p).round(4).to_markdown(index=False) + "\n")
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
