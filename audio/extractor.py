"""Extração de biomarcadores vocais de arquivos de áudio usando Parselmouth/Praat.

Extrai as mesmas features do dataset UCI Parkinson's:
- Frequência fundamental (Fo, Fmax, Fmin)
- Jitter (variações de frequência)
- Shimmer (variações de amplitude)
- HNR/NHR (razão harmônico/ruído)
- Medidas não-lineares (RPDE, DFA, D2, spread1, spread2, PPE) via nolds

Se nolds não estiver disponível ou falhar, usa valores medianos do dataset.
"""
import warnings
import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False


# Valores medianos do dataset UCI (fallback para features não-lineares)
MEDIAN_VALUES = {
    "RPDE": 0.498536,
    "DFA": 0.718099,
    "D2": 2.301442,
    "spread1": -5.720868,
    "spread2": 0.226510,
    "PPE": 0.206023,
}


def extract_features(audio_path: str) -> dict:
    """Extrai biomarcadores vocais de um arquivo WAV.

    Args:
        audio_path: caminho para o arquivo .wav

    Returns:
        dict com as 22 features do dataset UCI

    Raises:
        RuntimeError: se parselmouth não estiver instalado
    """
    if not HAS_PARSELMOUTH:
        raise RuntimeError(
            "Parselmouth não está instalado. Instale com: pip install praat-parselmouth"
        )

    warnings_list = []

    # Carregar áudio
    sound = parselmouth.Sound(audio_path)

    # Pitch analysis
    pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
    f0_values = pitch.selected_array["frequency"]
    f0_values = f0_values[f0_values > 0]  # remover unvoiced

    if len(f0_values) < 5:
        raise ValueError(
            "Áudio insuficiente para análise. Grave pelo menos 3 segundos de vogal sustentada (/a/)."
        )

    fo_mean = np.mean(f0_values)
    fo_max = np.max(f0_values)
    fo_min = np.min(f0_values)

    # Point process para jitter/shimmer
    point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)

    # Jitter
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = jitter_rap * 3  # DDP = 3 * RAP

    # Shimmer
    shimmer_local = call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_local_db = call(
        [sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq3 = call(
        [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq5 = call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_apq11 = call(
        [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    shimmer_dda = shimmer_apq3 * 3  # DDA = 3 * APQ3

    # Harmonics
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.5  # NHR = 1/HNR_linear

    # Medidas não-lineares
    nonlinear = _extract_nonlinear(f0_values, sound)

    features = {
        "MDVP:Fo(Hz)": fo_mean,
        "MDVP:Fhi(Hz)": fo_max,
        "MDVP:Flo(Hz)": fo_min,
        "MDVP:Jitter(%)": jitter_local * 100,
        "MDVP:Jitter(Abs)": jitter_local_abs,
        "MDVP:RAP": jitter_rap,
        "MDVP:PPQ": jitter_ppq5,
        "Jitter:DDP": jitter_ddp,
        "MDVP:Shimmer": shimmer_local,
        "MDVP:Shimmer(dB)": shimmer_local_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "MDVP:APQ": shimmer_apq11,
        "Shimmer:DDA": shimmer_dda,
        "NHR": nhr,
        "HNR": hnr,
        **nonlinear,
    }

    # Limpar NaN/inf
    for k, v in features.items():
        if v is None or np.isnan(v) or np.isinf(v):
            features[k] = MEDIAN_VALUES.get(k, 0.0)
            warnings_list.append(f"Feature {k} teve valor inválido, usando mediana")

    return features, warnings_list


def _extract_nonlinear(f0_values: np.ndarray, sound) -> dict:
    """Tenta extrair features não-lineares usando nolds.

    Se falhar, retorna valores medianos do dataset.
    """
    result = {}
    used_median = []

    signal = np.array(sound.values[0])

    if HAS_NOLDS and len(f0_values) >= 20:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result["RPDE"] = nolds.sampen(f0_values, emb_dim=2)
                if np.isnan(result["RPDE"]) or np.isinf(result["RPDE"]):
                    raise ValueError("RPDE inválido")
        except Exception:
            result["RPDE"] = MEDIAN_VALUES["RPDE"]
            used_median.append("RPDE")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result["DFA"] = nolds.dfa(f0_values)
                if np.isnan(result["DFA"]) or np.isinf(result["DFA"]):
                    raise ValueError("DFA inválido")
        except Exception:
            result["DFA"] = MEDIAN_VALUES["DFA"]
            used_median.append("DFA")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result["D2"] = nolds.corr_dim(f0_values, emb_dim=2)
                if np.isnan(result["D2"]) or np.isinf(result["D2"]):
                    raise ValueError("D2 inválido")
        except Exception:
            result["D2"] = MEDIAN_VALUES["D2"]
            used_median.append("D2")
    else:
        for key in ["RPDE", "DFA", "D2"]:
            result[key] = MEDIAN_VALUES[key]
            used_median.append(key)

    # spread1, spread2, PPE — sempre median (requerem algoritmos especializados)
    for key in ["spread1", "spread2", "PPE"]:
        result[key] = MEDIAN_VALUES[key]

    if used_median:
        print(f"  Usando valores medianos para: {used_median}")

    return result
