"""Shared settings for the embodied-emotion EEG study (design: ../实验设计方案.md, plan A, revision v2).

Every parameter here is referenced by section number in the design document so that
the Miko and AFFEC pipelines stay identical. "V<n>" refers to the revision table §0.2.
"""
from pathlib import Path

# ---- paths -----------------------------------------------------------------
DATA_ROOT = Path("E:/data")
MIKO_DIR = DATA_ROOT / "80_Miko_robot_expression_EEG"
AFFEC_DIR = DATA_ROOT / "81_AFFEC_virtual_agent_EEG_GSR_eye"
AFFEC_BIDS = AFFEC_DIR / "_bids_eeg"        # eeg.zip extracted here (02_affec_preprocess.py)
AFFEC_STIM = AFFEC_DIR / "_stim" / "stim"   # stim.zip extracted here
X2C_DIR = DATA_ROOT / "82_X2C_humanoid_robot_expressions_30ctrl"
# E: is a USB HDD; random reads of 100k small JPEGs starve the GPU -> images are mirrored to SSD
X2C_IMG_ROOT = Path("D:/data_cache/x2c")

ANALYSIS_DIR = Path(__file__).resolve().parent
DERIV_DIR = ANALYSIS_DIR / "derivatives"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIG_DIR = ANALYSIS_DIR / "figures"
MODEL_DIR = ANALYSIS_DIR / "models"
LOG_DIR = ANALYSIS_DIR / "logs"
for _d in (DERIV_DIR, RESULTS_DIR, FIG_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261115

# ---- channels (§3.1): the 16 Miko channels are the common set -------------
COMMON_CHANNELS = ["Fp1", "Fpz", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4",
                   "T7", "T8", "P3", "Pz", "P4", "POz", "Oz"]
MONTAGE = "standard_1020"
EOG_PROXY = ["Fp1", "Fp2"]           # no EOG channels in Miko -> frontal proxies
AFFEC_DROP = ["A1"]                  # earlobe channel (reference side), not scalp EEG

# ---- preprocessing (§3.1) --------------------------------------------------
SFREQ = 128.0                        # AFFEC is resampled to this
BAND = (0.5, 60.0)                   # matches Miko's provided filtering
NOTCH = 50.0                         # both datasets recorded on 50 Hz mains
ICA_HIGHPASS = 1.0                   # ICA is fitted on a 1 Hz high-passed copy
ICA_METHOD = "infomax"
ICA_N_COMPONENTS = 14                # 16 ch, average reference -> rank 15; 15 was unstable
ICA_SEED = 97
ICA_FIT_REJECT_UV = 400.0            # skip segments above this peak-to-peak when fitting ICA (V12)
EOG_Z_THRESHOLD = 3.0
EOG_R_THRESHOLD = 0.7                # ...or |r| >= 0.7 with Fp1/Fp2 (Miko blink ICs: r = .86-.99)
EOG_MAX_COMPONENTS = 2                # mne find_bads_eog (adaptive z-scoring)
ERP_LOWPASS = 30.0
TMIN, TMAX = -0.5, 1.0               # epoch span; -0.5..0 is the connectivity baseline (§4.2)
BASELINE = (-0.2, 0.0)               # ERP baseline (§3.1)
REJECT_ABS_UV = 100.0                # drop epoch if any channel exceeds +-100 uV
OVERLAP_WINDOW = (-0.5, 1.0)         # drop trial if another onset falls in this window (§2.1)
MIN_TRIALS_PER_EMOTION = 20          # Miko subject-average threshold (§3.1)
AFFEC_LONG_TMAX = 3.5                # long AFFEC epochs for expression-onset re-locking (V2)
AFFEC_RUNS = [1, 2, 3]               # run-0 = 4 practice trials (V7)

# ---- emotions (§3.4) -------------------------------------------------------
MIKO_EMOTIONS = {1: "angry", 2: "happy", 3: "sad", 4: "surprised"}
AFFEC_EMOTIONS = {"anger": "angry", "angry": "angry", "happy": "happy", "sad": "sad",
                  "fear": "fear", "disgust": "disgust", "neutral": "neutral"}
SHARED_EMOTIONS = ["angry", "happy", "sad"]   # present in both Miko and AFFEC

# ---- ERP components (§3.2 + V1) --------------------------------------------
# name: (tmin, tmax, {roi_label: [channels]}, polarity, cross_dataset)
# V1: AFFEC clips have speech audio -> vertex (Cz) ROIs are excluded from the
#     cross-dataset comparison; the *_Cz variants are sensitivity analyses.
COMPONENTS = {
    "N170": (0.130, 0.200, {"L": ["T7", "P3"], "R": ["T8", "P4"]}, -1, True),
    "P300": (0.250, 0.500, {"mid": ["Pz", "POz"]}, +1, True),
    "LPP": (0.400, 0.800, {"mid": ["Pz", "POz"]}, +1, True),
    "VPP": (0.140, 0.200, {"mid": ["Cz"]}, +1, False),
    "P300_Cz": (0.250, 0.500, {"mid": ["Cz", "Pz", "POz"]}, +1, False),
    "LPP_Cz": (0.400, 0.800, {"mid": ["Cz", "Pz", "POz"]}, +1, False),
}
# §3.3: standard N170 sites, only measurable in AFFEC's 62-channel montage
N170_STANDARD = {"L": ["P7", "PO7"], "R": ["P8", "PO8"]}

# ---- connectivity (§4) -----------------------------------------------------
MODULES = {
    "frontal": ["Fp1", "Fpz", "Fp2", "F3", "Fz", "F4"],
    "parietal": ["P3", "Pz", "P4", "POz"],
    "temporal": ["T7", "T8"],
    "aux": ["C3", "Cz", "C4", "Oz"],
}
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
CONN_POST = (0.0, 1.0)
CONN_BASE = (-0.5, 0.0)
CONN_REPEATS = 100                   # trial-count equalisation repeats (§4.2)
DENSITIES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_PERM = 5000

# ---- X2C (§2.3, §5) --------------------------------------------------------
# Order of ctrl_value in metadata.jsonl. Table 2 of the paper lists the names in a
# different order; this mapping was inferred from the empirical ranges and from the
# left/right pair correlations (r = .79-.93 for brows, lids, lip-corner raise, lip-top
# raise), see results/x2c_control_mapping.csv.
X2C_CONTROLS = [
    "BrowInnerL", "BrowInnerR", "BrowOuterL", "BrowOuterR",
    "EyelidLowerL", "EyelidLowerR", "EyelidUpperL", "EyelidUpperR",
    "GazePhi", "GazeTheta",
    "HeadPitch", "HeadRoll", "HeadYaw",
    "JawPitch", "JawYaw",
    "LipBottomCurl", "LipBottomDepressL", "LipBottomDepressM", "LipBottomDepressR",
    "LipCornerRaiseL", "LipCornerRaiseR", "LipCornerStretchL", "LipCornerStretchR",
    "LipTopCurl", "LipTopRaiseL", "LipTopRaiseM", "LipTopRaiseR",
    "NeckPitch", "NeckRoll", "NoseWrinkle",
]
X2C_PAIRS = [("BrowInnerL", "BrowInnerR"), ("BrowOuterL", "BrowOuterR"),
             ("EyelidUpperL", "EyelidUpperR"), ("EyelidLowerL", "EyelidLowerR"),
             ("LipBottomDepressL", "LipBottomDepressR"), ("LipCornerRaiseL", "LipCornerRaiseR"),
             ("LipCornerStretchL", "LipCornerStretchR"), ("LipTopRaiseL", "LipTopRaiseR")]
X2C_POSE = ["GazePhi", "GazeTheta", "HeadPitch", "HeadRoll", "HeadYaw", "NeckPitch", "NeckRoll"]
X2C_EYE_SUBSPACE = ["BrowInnerL", "BrowInnerR", "BrowOuterL", "BrowOuterR",
                    "EyelidLowerL", "EyelidLowerR", "EyelidUpperL", "EyelidUpperR",
                    "GazePhi", "GazeTheta"]          # V5: Miko I's face shows eyes only
X2C_BLOCK = 1000                     # contiguous frames per split block (no adjacent-frame leakage)
X2C_VAL_FRAC = 0.1
X2C_IMG = 224
X2C_EPOCHS = 12
X2C_BATCH = 96

# ---- stimulus kinematics (§5.2, V1, V2) ------------------------------------
EXPR_ONSET_FRAC = 0.20               # motion energy first exceeds 20 % of its peak
EXPR_ONSET_RULE = (0.70, 0.300)      # >=70 % of clips with onset > 300 ms -> expression-locked primary
SPEECH_RMS_FRAC = 0.10               # speech onset: audio RMS first exceeds 10 % of its peak
