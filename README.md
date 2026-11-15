# Cross-dataset EEG validity audit

Analysis code accompanying **Auditing cross-dataset electroencephalography comparisons between a physically present robot and screen-mediated emotional faces**.

This repository contains analysis scripts, dependency specifications and analysis-plan documentation. Raw EEG, participant-level results, videos, model weights and manuscript files are obtained or generated separately.

## Data and setup

Use Python 3.11 and a separate environment. The original analysis used PyTorch 2.5.1 and CUDA 12.4 for regressor training. Install the dependencies in `experiment/requirements-lock.txt` (the descriptive list is `experiment/requirements.txt`). GPU packages may require the matching official PyTorch index.

Download the original datasets under their providers' access conditions:

* Miko: https://doi.org/10.5281/zenodo.5784856
* AFFEC: https://doi.org/10.5281/zenodo.19408373 (GSR archive: https://doi.org/10.5281/zenodo.14794876)
* X2C: https://huggingface.co/datasets/Peizhen/X2C

Edit **your clone's** `analysis/config.py` to set `DATA_ROOT` and `X2C_IMG_ROOT`. The checked-in defaults are the original Windows data locations; they are not portable defaults. Arrange the extracted AFFEC BIDS EEG and stimuli under `_bids_eeg` and `_stim/stim` as documented in `analysis/README.md`. Download the MediaPipe face landmarker model to `analysis/models/face_landmarker.task` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task .

Run from `analysis/`:

```text
python run_pipeline.py --status
python run_pipeline.py
```

The pipeline creates outputs under `analysis/derivatives`, `analysis/results`, `analysis/models`, `analysis/logs`, and `submission/figures` / `submission/manuscript`. Existing outputs can be regenerated; use a fresh clone and separate data copies. Full EEG processing and GPU training are computationally intensive. See `analysis/README.md` for all 20 steps and dependencies.

## Analysis-plan provenance

`analysis/preregistration/` preserves the local frozen analysis-plan snapshot and its original `MANIFEST.json`. It is a local timestamped record, not an independently registered public preregistration. The current scripts include later audit extensions; the snapshot deliberately retains the earlier versions. Do not replace the original manifest with hashes of current scripts. The runner skips the freezing step when the preserved manifest exists.

## Release validation

The release was checked for Python syntax, file inventory, credential patterns and preservation of the original snapshot hashes. The full numerical analysis was not rerun as part of repository publication. The numerical results remain those of the original analysis.
