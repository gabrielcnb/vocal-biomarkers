"""Main training and evaluation pipeline.

Executa:
1. Download e carregamento do dataset (group-aware split)
2. Treinamento dos 3 modelos (SVM, RF, XGBoost)
3. Full evaluation with plots
4. Final summary
"""
import sys
import time
from pathlib import Path

# Adicionar pasta do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml.dataset import load_and_split
from ml.train_models import train_all
from ml.evaluate import evaluate_all


def main():
    print("=" * 70)
    print("  VOCAL BIOMARKERS FOR PARKINSON'S DETECTION")
    print("  Training and Evaluation Pipeline")
    print("=" * 70)

    start_time = time.time()

    # 1. Dataset
    print("\n[1/3] Carregando dataset UCI Parkinson's...")
    X_train, X_test, y_train, y_test, feature_names, scaler, groups = load_and_split()

    # 2. Treinamento
    print("\n[2/3] Treinando modelos...")
    results = train_all(X_train, y_train, scaler, feature_names)

    # 3. Evaluation
    print("\n[3/3] Avaliando modelos no conjunto de teste...")
    metrics = evaluate_all(results, X_test, y_test, feature_names)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  RESULTADOS FINAIS (Conjunto de Teste - Group-Aware Split)")
    print("=" * 70)
    print(f"{'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'CV AUC':>10}")
    print("-" * 78)
    for name, m in metrics.items():
        print(
            f"{name:<18} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f} {m['cv_auc']:>10.4f}"
        )
    print("-" * 78)

    best = max(metrics.items(), key=lambda x: x[1]["auc"])
    print(f"\nMelhor modelo (por AUC no teste): {best[0]}, AUC = {best[1]['auc']:.4f}")
    print(f"Tempo total: {elapsed:.1f}s")
    print(f"\nPlots salvos em: static/plots/")
    print(f"Modelos salvos em: models/")


if __name__ == "__main__":
    main()
