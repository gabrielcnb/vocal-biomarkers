"""Configuração central do projeto Vocal Biomarkers."""
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
    "Frequência Fundamental": [
        ("MDVP:Fo(Hz)", "Frequência fundamental média"),
        ("MDVP:Fhi(Hz)", "Frequência fundamental máxima"),
        ("MDVP:Flo(Hz)", "Frequência fundamental mínima"),
    ],
    "Jitter (Variação de Frequência)": [
        ("MDVP:Jitter(%)", "Jitter percentual"),
        ("MDVP:Jitter(Abs)", "Jitter absoluto"),
        ("MDVP:RAP", "Perturbação relativa média"),
        ("MDVP:PPQ", "Quociente de perturbação de período"),
        ("Jitter:DDP", "Diferença de diferenças de período"),
    ],
    "Shimmer (Variação de Amplitude)": [
        ("MDVP:Shimmer", "Shimmer local"),
        ("MDVP:Shimmer(dB)", "Shimmer em dB"),
        ("Shimmer:APQ3", "Quociente de perturbação de amplitude 3"),
        ("Shimmer:APQ5", "Quociente de perturbação de amplitude 5"),
        ("MDVP:APQ", "Quociente de perturbação de amplitude"),
        ("Shimmer:DDA", "Diferença de diferenças de amplitude"),
    ],
    "Razão Harmônico/Ruído": [
        ("NHR", "Razão ruído-harmônico"),
        ("HNR", "Razão harmônico-ruído"),
    ],
    "Medidas Não-Lineares": [
        ("RPDE", "Entropia de recorrência"),
        ("DFA", "Análise de flutuação destendenciada"),
        ("D2", "Dimensão de correlação"),
        ("spread1", "Medida fundamental de variação de frequência"),
        ("spread2", "Medida de variação de frequência"),
        ("PPE", "Entropia de período de pitch"),
    ],
}
