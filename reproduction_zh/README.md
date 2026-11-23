# 社交机器人表情评估的中文图复现

`summary_data/` 保存投稿稿件用到的汇总统计。Python脚本以汇总表为输入，可重绘正文图1～5和补充图S1～S2。原始数据及完整流水线见仓库首页。

| 图 | 数据文件 |
|---|---|
| 图1 | 研究流程示意 |
| 图2 | `model_calibration_summary.csv` |
| 图3 | `event_type_within_affec.csv`、`erp_stats_validation16.csv` |
| 图4 | `event_timing_summary.csv`、`timing_sensitivity_summary.csv` |
| 图5 | `bb_h4.csv`、`bb_sci_components_bs.csv`、`late_lpp.csv` |
| 图S1 | `robust_spec_curve.csv` |
| 图S2 | `eed_summary.csv` |

在仓库根目录执行：

```text
python reproduction_zh/plot_figures.py --output figures_zh --song-font C:/Windows/Fonts/simsun.ttc
```

脚本依赖 NumPy、pandas、Matplotlib，输出 EPS、PDF 和 600 dpi PNG。换用其他平台时，将字体参数指向本机的宋体文件。汇总表保留原分析输出的字段名。用于分析的第三方原始数据请从各数据源取得。
