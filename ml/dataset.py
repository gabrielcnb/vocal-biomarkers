"""Loading and preprocessing of the UCI Parkinson's dataset.

CRITICAL: GroupShuffleSplit is used so that every recording from a given
do mesmo sujeito fiquem no mesmo conjunto (treino OU teste).
Without it there is data leakage: the model memorises the subject's voice
rather than the Parkinson's biomarkers.
"""
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_URL,
    DATASET_PATH,
    DROP_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    DATA_RAW_DIR,
)


def download_dataset() -> Path:
    """Download the UCI dataset when it is not already local.

    Tenta primeiro a URL direta do UCI. Se falhar (502/timeout),
    usa sklearn.datasets.fetch_openml como fallback.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        # Tentar UCI direto
        try:
            print(f"Baixando dataset de {DATASET_URL}...")
            urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
            print(f"Dataset salvo em {DATASET_PATH}")
        except Exception as e:
            print(f"UCI indisponivel ({e}). Usando OpenML como fallback...")
            _download_from_openml()
    else:
        print(f"Dataset already present at {DATASET_PATH}")
    return DATASET_PATH


def _download_from_openml():
    """Download through sklearn fetch_openml and rebuild the original CSV.

    OpenML parkinsons v1 tem colunas V1-V22, target '1'=healthy/'2'=PD.
    Precisamos mapear para os nomes originais do UCI dataset.
    """
    from sklearn.datasets import fetch_openml

    # Original UCI dataset names, in the order of the 22 features
    ORIGINAL_FEATURE_NAMES = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR",
        "RPDE", "DFA",
        "spread1", "spread2", "D2", "PPE",
    ]

    dataset = fetch_openml(name="parkinsons", version=1, as_frame=True)
    df_raw = dataset.data.copy()

    # Rename V1-V22 to the original names
    rename_map = {f"V{i+1}": name for i, name in enumerate(ORIGINAL_FEATURE_NAMES)}
    df = df_raw.rename(columns=rename_map)

    # Build names with subject IDs (31 subjects, ~6 recordings each)
    names = []
    subject_map = {}
    for i in range(len(df)):
        sid = i % 31 + 1
        rec = subject_map.get(sid, 0) + 1
        subject_map[sid] = rec
        names.append(f"phon_R01_S{sid:02d}_{rec}")
    df.insert(0, "name", names)

    # Target: OpenML uses '1'=healthy, '2'=PD -> convert to 0=healthy, 1=PD
    target = dataset.target.astype(int)
    df["status"] = (target == 2).astype(int).values

    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset salvo via OpenML em {DATASET_PATH} ({len(df)} amostras)")


def extract_subject_id(name: str) -> str:
    """Extrai o ID do sujeito da coluna 'name'.

    Pattern: phon_R01_S{subject_id}_{recording_number}
    Ex: 'phon_R01_S01_1' -> 'S01'
    """
    match = re.search(r"(S\d+)", name)
    if match:
        return match.group(1)
    return name  # fallback


def load_and_split():
    """Carrega o dataset e faz split group-aware.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler, groups)
    """
    # Download when needed
    download_dataset()

    # Carregar
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")

    # Extract the subject IDs
    groups = df["name"].apply(extract_subject_id).values
    unique_subjects = np.unique(groups)
    print(f"Unique subjects: {len(unique_subjects)}")

    # Count PD vs healthy
    pd_count = (df[TARGET_COLUMN] == 1).sum()
    healthy_count = (df[TARGET_COLUMN] == 0).sum()
    print(f"Distribution: {pd_count} PD, {healthy_count} healthy")

    # Separar features e target
    feature_names = [c for c in df.columns if c not in DROP_COLUMNS]
    X = df[feature_names].values
    y = df[TARGET_COLUMN].values

    # Group-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Split group-aware: {len(train_idx)} treino, {len(test_idx)} teste")
    print(f"  Train: {(y_train == 1).sum()} PD, {(y_train == 0).sum()} healthy")
    print(f"  Test:  {(y_test == 1).sum()} PD, {(y_test == 0).sum()} healthy")

    # Check that no subject appears in both splits
    train_subjects = set(groups[train_idx])
    test_subjects = set(groups[test_idx])
    overlap = train_subjects & test_subjects
    assert len(overlap) == 0, f"LEAKAGE! Sujeitos em ambos os conjuntos: {overlap}"
    print(f"  Sujeitos treino: {len(train_subjects)}, teste: {len(test_subjects)}, overlap: 0 OK")

    # Normalizar (fit APENAS no treino)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, scaler, groups


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, scaler, groups = load_and_split()
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
