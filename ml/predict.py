"""Predição usando o melhor modelo treinado."""
import numpy as np
import joblib

from config import BEST_MODEL, SCALER_PATH, FEATURE_NAMES_PATH


def load_model():
    """Carrega o melhor modelo, scaler e nomes das features."""
    model = joblib.load(BEST_MODEL)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, scaler, feature_names


def predict_from_features(feature_dict: dict) -> dict:
    """Prediz a partir de um dicionário de features.

    Args:
        feature_dict: {feature_name: value}

    Returns:
        dict com prediction, confidence, probabilities
    """
    model, scaler, feature_names = load_model()

    # Montar array na ordem correta
    values = []
    missing = []
    for fname in feature_names:
        if fname in feature_dict:
            values.append(float(feature_dict[fname]))
        else:
            missing.append(fname)
            values.append(0.0)  # fallback

    if missing:
        print(f"AVISO: Features ausentes (usando 0.0): {missing}")

    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    prediction = int(model.predict(X_scaled)[0])

    return {
        "prediction": prediction,
        "label": "Parkinson" if prediction == 1 else "Saudável",
        "confidence": float(max(proba)),
        "prob_healthy": float(proba[0]),
        "prob_parkinson": float(proba[1]),
        "missing_features": missing,
    }
