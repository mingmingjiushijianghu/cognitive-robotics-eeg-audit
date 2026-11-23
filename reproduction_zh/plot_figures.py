"""从已保存的统计汇总表重绘中文图（Python 3.11，numpy/pandas/matplotlib）。"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BLUE = "#14637a"
OCHRE = "#9e4d24"
INK = "#202e35"
GREY = "#65747b"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path(__file__).parent / "summary_data")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--song-font", type=Path, default=Path("C:/Windows/Fonts/simsun.ttc"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if args.song_font.exists():
    font_manager.fontManager.addfont(args.song_font)
song = font_manager.FontProperties(fname=args.song_font).get_name() if args.song_font.exists() else "Noto Serif CJK SC"
plt.rcParams.update({
    "font.family": ["Times New Roman", song],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def read(file):
    return pd.read_csv(args.data / (file + ".csv"))

def save(fig, name):
    for ext in ("png", "pdf", "eps"):
        output = args.output / f"{name}.{ext}"
        if output.exists():
            raise FileExistsError(output)
        fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.055, facecolor="white")
    plt.close(fig)

def panels(count, height=2.05, width=3.14):
    fig, axes = plt.subplots(count, 1, figsize=(width, height * count), layout="constrained", squeeze=False)
    return fig, axes.ravel()

def effect_intervals(ax, centers, lower, upper, labels, color=BLUE):
    centers, lower, upper = [np.asarray(x) for x in (centers, lower, upper)]
    yy = np.arange(len(centers))
    ax.errorbar(centers, yy, xerr=[centers-lower, upper-centers], marker="o", linestyle="none",
                markersize=3.5, color=color, ecolor=color, linewidth=0.9, capsize=2.5)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.65, zorder=0)

def workflow():
    fig, ax = plt.subplots(figsize=(3.14, 2.30))
    ax.axis("off")
    entries = [
        (0.50, 0.91, "社交机器人表情评估", INK),
        (0.50, 0.72, "Miko、AFFEC脑电记录与X2C表情参数", INK),
        (0.25, 0.49, "跨数据集脑电比较", BLUE),
        (0.76, 0.49, "屏幕面孔组内分析", OCHRE),
        (0.25, 0.28, "ERP为主，网络与距离辅助", BLUE),
        (0.76, 0.28, "动作幅度、评分与LPP", OCHRE),
        (0.25, 0.08, "锁时、布局与模型校准", BLUE),
        (0.76, 0.08, "主观唤醒度关联", OCHRE),
    ]
    for x, y, label, color in entries:
        ax.text(x, y, label, color=color, ha="center", va="center", transform=ax.transAxes,
                fontsize=7.9, bbox={"boxstyle":"round,pad=0.28", "fc":"white", "ec":color, "lw":0.62})
    connections = [
        (.5,.86,.5,.78), (.47,.67,.25,.55), (.53,.67,.76,.55),
        (.25,.44,.25,.34), (.76,.44,.76,.34), (.25,.23,.25,.14), (.76,.23,.76,.14),
    ]
    for x1,y1,x2,y2 in connections:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), transform=ax.transAxes, mutation_scale=8,
                                     arrowstyle="-|>", color=GREY, linewidth=0.65))
    save(fig, "fig1")

def calibration():
    data = read("model_calibration_summary")
    fig, (ax,) = panels(1, 1.92)
    for model, shift, label, color in [
        ("single_trial_crossed", -0.16, "单试次交叉模型", OCHRE),
        ("cell_mean", 0.16, "单元均值模型", BLUE),
    ]:
        subset = data[data.model == model].set_index("component").loc[["P300", "LPP"]]
        rates = subset.false_positive_rate.to_numpy()
        xpos = np.arange(2) + shift
        ax.errorbar(xpos, rates, yerr=[rates-subset.ci_lo, subset.ci_hi-rates], fmt="o",
                    color=color, capsize=3, linewidth=0.85, label=label, markersize=4)
        for x, rate in zip(xpos, rates):
            ax.text(x+0.069, rate, f"{rate:.0%}", fontsize=7.5, va="center", color=color)
    ax.axhline(0.05, color=GREY, linestyle="--", linewidth=0.65, label="名义水平")
    ax.set(xticks=[0,1], xticklabels=["P300","LPP"], ylabel="假阳性率", xlim=(-0.42, 1.48), ylim=(0, 0.43))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper right", labelspacing=0.22)
    save(fig, "fig2")

def comparability():
    events = read("event_type_within_affec").set_index("component")
    validity = read("erp_stats_validation16")
    validity = validity[validity.reference != "full62:N170"].set_index("component")
    fig, (a,b) = panels(2, 1.39)
    positions = np.arange(2)
    a.axhline(0, color=GREY, linewidth=0.65)
    for attr, shift, label, color in [
        ("mean_video", -.14, "面孔出现", BLUE), ("mean_expression", .14, "表情起始", OCHRE)
    ]:
        a.scatter(positions+shift, events.loc[["P300","LPP"],attr], color=color, s=15, label=label)
    a.set(xticks=positions, xticklabels=["P300","LPP"], xlim=(-.32,1.42), ylim=(-.21,.24),
          ylabel="波幅／基线标准差", title="(a)　锁时事件（49名被试）")
    a.legend(loc="upper right", ncol=2, columnspacing=0.6, handletextpad=0.22)
    b.axhline(.70, color=GREY, linestyle="--", linewidth=.65)
    b.scatter(np.arange(3), validity.loc[["N170","P300","LPP"],"r_subjects"], s=16, color=BLUE)
    for x, value in enumerate(validity.loc[["N170","P300","LPP"],"r_subjects"]):
        b.text(x, value+.07, f"{value:.2f}", ha="center", fontsize=7.5)
    b.set(xticks=np.arange(3), xticklabels=["N170","P300","LPP"], ylabel="相关系数",
          xlim=(-.50,2.5), ylim=(0,1.05), title="(b)　16导与62导测量（43名被试）")
    save(fig, "fig3")

def timing():
    summary = read("event_timing_summary").set_index("dataset")
    sensitivity = read("timing_sensitivity_summary")
    fig, (a,b) = panels(2, 1.55)
    ids = ["miko","miko_author","affec"]
    tested = summary.loc[ids, "n"]
    rates = summary.loc[ids, "frac_itc_p05"]
    a.barh([0,1,2], rates, color=[OCHRE,OCHRE,BLUE], height=.49)
    for idx, (value, count) in enumerate(zip(rates, tested)):
        a.text(value+.025, idx, f"{round(value*count)}/{count}", va="center", fontsize=8)
    a.set(yticks=[0,1,2], yticklabels=["Miko统一清理","Miko原作者清理","AFFEC"],
          xlim=(0,1.02), xlabel="检出比例", title="(a)　标记处相位一致性检验")
    a.invert_yaxis()
    a.xaxis.set_major_formatter(PercentFormatter(1))
    for key, label, color, symbol in [
        ("group_itc_p", "试次间相位一致性", BLUE, "o"), ("group_gfp_p", "全局场功率", OCHRE, "s")
    ]:
        b.plot(sensitivity.scale, sensitivity[key], marker=symbol, ms=3.2, color=color, label=label, lw=.8)
    b.axhline(.05, color=GREY, linestyle="--", linewidth=.65)
    b.set(yscale="log", ylim=(1e-10,2.5), xticks=[0,.25,.5,1],
          xlabel="反应注入幅度／原始表情反应", ylabel="组水平 p 值", title="(b)　注入反应的检出灵敏度")
    b.legend(loc="lower left", labelspacing=.20, borderaxespad=.3, handlelength=1.4)
    save(fig, "fig4")

def ratings():
    comps = read("bb_sci_components_bs").set_index("outcome")
    h4 = read("bb_h4").query("kinematics == 'bs' and dataset == 'affec'").iloc[0]
    late = read("late_lpp").set_index("window")
    fig, (a,b) = panels(2, 1.60)
    arousal = comps.loc[["felt arousal", "perceived arousal"]]
    effect_intervals(a, list(arousal.beta)+[h4.lmm_sci_beta], list(arousal.ci_lo)+[h4.lmm_sci_lo],
                     list(arousal.ci_hi)+[h4.lmm_sci_hi], ["自我感受唤醒度","感知唤醒度","主观情绪感染"])
    a.set(xlim=(-.05,.9), xlabel="回归斜率／量表点", title="(a)　主观评分")
    late = late.loc[["peak-locked","sustained"]]
    effect_intervals(b, [h4.lmm_lpp_beta]+list(late.beta), [h4.lmm_lpp_lo]+list(late.ci_lo),
                     [h4.lmm_lpp_hi]+list(late.ci_hi), ["视频起始后\n400～800 ms","动作峰值后\n400～800 ms","视频起始后\n1.0～2.5 s"], OCHRE)
    b.set(xlim=(-.11,.135), xlabel="回归斜率／基线标准差", title="(b)　晚期正电位与持续正波")
    save(fig, "fig5")

def specification():
    specs = read("robust_spec_curve")
    fig, axes = panels(2, 1.40)
    for ax, component in zip(axes, ("P300","LPP")):
        for key, label, color in [("video","面孔出现",BLUE), ("expression","表情起始",OCHRE)]:
            values = np.sort(specs.loc[(specs.component == component) & (specs.affec_lock == key),"hedges_g"])
            ax.plot(np.arange(1, len(values)+1), values, label=label, color=color, linewidth=1.0)
        ax.axhline(0, color=GREY, linestyle="--", linewidth=.65)
        ax.set(ylabel="效应量 g", title=component)
        ax.legend(loc="best", ncol=2, columnspacing=.64)
    axes[-1].set_xlabel("各锁时方式内按效应量排序的设定")
    save(fig, "figs1")

def supplementary_distance():
    distances = read("eed_summary")
    fig, (ax,) = panels(1, 1.9)
    effect_intervals(ax, distances.EED, distances.ci_lo, distances.ci_hi,
                     ["主分析", "被试内中心化", "去除β特征", "不作运动学校正"])
    ax.set(xlim=(0.85, 3.05), xlabel="综合脑电特征距离")
    for y,p_value in enumerate(distances.p_perm):
        ax.text(3.03, y-.15, f"置换 p = {p_value:.3f}", ha="right", fontsize=7.1)
    save(fig, "figs2")

if __name__ == "__main__":
    for render in (workflow, calibration, comparability, timing, ratings, specification, supplementary_distance):
        render()

