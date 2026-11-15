# Analysis code: auditing cross-dataset EEG comparisons (Miko, AFFEC, X2C)

This folder regenerates every result, figure and number of the manuscript from the three public
databases. Environment: Python 3.11 (`../experiment/requirements-lock.txt`); PyTorch 2.5.1 with
CUDA 12.4 for step 03.

## Data

| Database | Source | Expected location (`config.py`) |
|---|---|---|
| Miko | Zenodo, doi:10.5281/zenodo.5784856 | `DATA_ROOT/80_Miko_robot_expression_EEG` |
| AFFEC | Zenodo, doi:10.5281/zenodo.19408373 (`gsr.zip` from 14794876) | `DATA_ROOT/81_AFFEC_virtual_agent_EEG_GSR_eye` |
| X2C | Hugging Face `Peizhen/X2C` | `DATA_ROOT/82_X2C_humanoid_robot_expressions_30ctrl` |

The MediaPipe face landmarker model (`models/face_landmarker.task`) is downloaded from
`storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/`.

## Running

```
python run_pipeline.py            # run all steps whose inputs are ready
python run_pipeline.py --status   # state of each step
python run_pipeline.py --force 06 # rerun a step and everything downstream
```

`run_pipeline.py` records each script's SHA-256 and reruns a step and its dependants when the
script changes. Logs: `logs/`; results: `results/`; figures: `../submission/figures/`; manuscript
numbers: `../submission/manuscript/numbers.tex`.

## Steps

| Step | Script | Purpose |
|---|---|---|
| 00 | `00_freeze_prereg.py` | freezes the analysis plan (SHA-256) before ERP results exist |
| 01 | `01_miko_preprocess.py` | Miko preprocessing (common pipeline and authors' cleaned data) |
| 02 | `02_affec_preprocess.py` | AFFEC preprocessing (16-channel, long and 62-channel epochs) |
| 03 | `03_x2c_train.py` | X2C image-to-control regressor |
| 04 | `04_stim_features.py` | speech onset, MediaPipe kinematics, expression onset, eye geometry |
| 05 | `05_erp_features.py` | single-trial ERP features |
| 06 | `06_erp_stats.py` | route-1 statistics, 16-channel validity (C4) |
| 07 | `07_x2c_kinematics.py` | X2C predictions and cross-domain validity (C5) |
| 08 | `08_connectivity.py` | dwPLI networks, topology, NBS (C6 by Miko version) |
| 09 | `09_brain_behavior.py` | AFFEC kinematics, contagion and LPP |
| 10 | `10_eed.py` | embodied emotional distance |
| 11 | `11_report.py` | `results/REPORT.md` and framework comparison |
| 12 | `12_robustness.py` | specification curves and cluster tests |
| 13 | `13_paper_figures.py` | Figures 1-6 (PDF and 600-dpi TIFF) |
| 14 | `14_event_timing.py` | event-timing check (C1) |
| 15 | `15_model_calibration.py` | false-positive rates of the two models (C7) |
| 16 | `16_paper_numbers.py` | LaTeX macros for every number in the manuscript |
| 17 | `17_timing_sensitivity.py` | sensitivity of C1 for Miko's event type |
| 18 | `18_event_checks.py` | onset-threshold robustness and event-type effect (C2) |
| 19 | `19_late_lpp.py` | later LPP windows relative to the motion peak |

`common.py` holds the preprocessing shared by both datasets; `config.py` holds all parameters.
