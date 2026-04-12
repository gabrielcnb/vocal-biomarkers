"""Avaliação completa dos modelos treinados.

Gera:
- Curvas ROC sobrepostas com AUC
- Matrizes de confusão (1x3 heatmaps)
- Importância de features (RF, XGBoost, SVM permutation)
- Comparação de métricas em barra
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.inspection import permutation_importance

from config import PLOTS_DIR


def evaluate_all(models_dict, X_test, y_test, feature_names):
    """Avalia todos os modelos e gera plots.

    Args:
        models_dict: {nome: (modelo, cv_score)}
        X_test: dados de teste normalizados
        y_test: labels de teste
        feature_names: lista de nomes das features

    Returns:
        dict: métricas de cada modelo
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    metrics = {}

    for name, (model, cv_score) in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
            "cv_auc": cv_score,
        }

        print(f"\n{'=' * 50}")
        print(f"Modelo: {name}")
        print(f"{'=' * 50}")
        print(classification_report(y_test, y_pred, target_names=["Saudável", "Parkinson"]))

    # --- 1. Curvas ROC ---
    _plot_roc_curves(models_dict, X_test, y_test)

    # --- 2. Matrizes de Confusão ---
    _plot_confusion_matrices(models_dict, X_test, y_test)

    # --- 3. Importância de Features ---
    _plot_feature_importance(models_dict, X_test, y_test, feature_names)

    # --- 4. Comparação de Métricas ---
    _plot_metrics_comparison(metrics)

    return metrics


def _plot_roc_curves(models_dict, X_test, y_test):
    """Curvas ROC sobrepostas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for (name, (model, _)), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Taxa de Falso Positivo", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiro Positivo", fontsize=12)
    ax.set_title("Curvas ROC - Detecção de Parkinson por Biomarcadores Vocais", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot salvo: roc_curves.png")


def _plot_confusion_matrices(models_dict, X_test, y_test):
    """Matrizes de confusão lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["Saudável", "Parkinson"]

    for ax, (name, (model, _)) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False,
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Real" if ax == axes[0] else "")
        ax.set_xlabel("Previsto")

    plt.suptitle("Matrizes de Confusão", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot salvo: confusion_matrices.png")


def _plot_feature_importance(models_dict, X_test, y_test, feature_names):
    """Importância de features para cada modelo."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (name, (model, _)) in zip(axes, models_dict.items()):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # SVM: usar permutation importance
            perm = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = perm.importances_mean

        # Top 15
        indices = np.argsort(importances)[-15:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(indices)))
        ax.barh(range(len(indices)), top_importances, color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importância")

    plt.suptitle("Top 15 Features Mais Importantes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot salvo: feature_importance.png")


def _plot_metrics_comparison(metrics):
    """Gráfico de barras comparando métricas."""
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_labels = ["Acurácia", "Precisão", "Recall", "F1-Score", "AUC"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, (name, m) in enumerate(metrics.items()):
        values = [m[mn] for mn in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparação de Modelos - Métricas de Desempenho", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.15])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot salvo: metrics_comparison.png")
