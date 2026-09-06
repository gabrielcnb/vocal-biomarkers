"""Central configuration for the Vocal Biomarkers project."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "static" / "plots"

# Dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
DATASET_PATH = DATA_RAW_DIR / "parkinsons.data"

# Features to drop (not predictive)
DROP_COLUMNS = ["name", "status"]

# Target column
TARGET_COLUMN = "status"

# Model filenames
SVM_MODEL = MODELS_DIR / "svm_model.joblib"
RF_MODEL = MODELS_DIR / "rf_model.joblib"
XGB_MODEL = MODELS_DIR / "xgb_model.joblib"
BEST_MODEL = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# Flask
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# Feature groups for UI
FEATURE_GROUPS = {
    "Fundamental Frequency": [
        ("MDVP:Fo(Hz)", "Average fundamental frequency"),
        ("MDVP:Fhi(Hz)", "Maximum fundamental frequency"),
        ("MDVP:Flo(Hz)", "Minimum fundamental frequency"),
    ],
    "Jitter (Frequency Variation)": [
        ("MDVP:Jitter(%)", "Jitter, percentage"),
        ("MDVP:Jitter(Abs)", "Jitter, absolute"),
        ("MDVP:RAP", "Relative average perturbation"),
        ("MDVP:PPQ", "Period perturbation quotient"),
        ("Jitter:DDP", "Difference of period differences"),
    ],
    "Shimmer (Amplitude Variation)": [
        ("MDVP:Shimmer", "Local shimmer"),
        ("MDVP:Shimmer(dB)", "Shimmer in dB"),
        ("Shimmer:APQ3", "Amplitude perturbation quotient 3"),
        ("Shimmer:APQ5", "Amplitude perturbation quotient 5"),
        ("MDVP:APQ", "Amplitude perturbation quotient"),
        ("Shimmer:DDA", "Difference of amplitude differences"),
    ],
    "Harmonics-to-Noise Ratio": [
        ("NHR", "Noise-to-harmonics ratio"),
        ("HNR", "Harmonics-to-noise ratio"),
    ],
    "Non-Linear Measures": [
        ("RPDE", "Recurrence period density entropy"),
        ("DFA", "Detrended fluctuation analysis"),
        ("D2", "Correlation dimension"),
        ("spread1", "Fundamental frequency variation measure"),
        ("spread2", "Frequency variation measure"),
        ("PPE", "Pitch period entropy"),
    ],
}
