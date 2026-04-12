"""Flask web server para Vocal Biomarkers for Parkinson's Detection."""
import sys
import os
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, BEST_MODEL, FEATURE_GROUPS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload


@app.route("/")
def index():
    """Página principal com formulário de predição."""
    model_exists = BEST_MODEL.exists()
    return render_template("index.html", feature_groups=FEATURE_GROUPS, model_exists=model_exists)


@app.route("/evaluation")
def evaluation():
    """Dashboard de avaliação dos modelos."""
    plots_dir = Path(__file__).parent / "static" / "plots"
    plots = {
        "roc": (plots_dir / "roc_curves.png").exists(),
        "confusion": (plots_dir / "confusion_matrices.png").exists(),
        "importance": (plots_dir / "feature_importance.png").exists(),
        "comparison": (plots_dir / "metrics_comparison.png").exists(),
    }
    return render_template("evaluation.html", plots=plots)


@app.route("/api/predict/manual", methods=["POST"])
def predict_manual():
    """Predição a partir de features inseridas manualmente."""
    try:
        from ml.predict import predict_from_features

        data = request.get_json()
        if not data:
            return jsonify({"error": "Nenhum dado recebido"}), 400

        result = predict_from_features(data)
        return jsonify(result)

    except FileNotFoundError:
        return jsonify({"error": "Modelo não treinado. Execute train.py primeiro."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/upload", methods=["POST"])
def predict_upload():
    """Predição a partir de arquivo de áudio enviado."""
    try:
        from audio.extractor import extract_features
        from ml.predict import predict_from_features

        if "audio" not in request.files:
            return jsonify({"error": "Nenhum arquivo de áudio enviado"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400

        # Salvar temporariamente
        suffix = Path(audio_file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            features, warnings = extract_features(tmp_path)
            result = predict_from_features(features)
            result["extracted_features"] = features
            result["warnings"] = warnings
            return jsonify(result)
        finally:
            os.unlink(tmp_path)

    except FileNotFoundError:
        return jsonify({"error": "Modelo não treinado. Execute train.py primeiro."}), 503
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Erro na extração de features: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"Servidor iniciando em http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
