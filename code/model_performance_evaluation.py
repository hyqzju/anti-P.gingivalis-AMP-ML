"""Evaluate machine-learning model performance for antimicrobial peptide data.

This script performs five-fold stratified cross-validation for three models:
Random Forest, Logistic Regression, and XGBoost. It reports Average Precision
(AP), accuracy, recall, precision, F1 score, and mean ROC curves.

Outputs:
    - Figure 1 as PNG and PDF
    - Mean model performance metrics as CSV
    - Per-fold model metrics as CSV
    - Mean ROC curve coordinates as CSV

Example:
    python model_performance_evaluation.py --input-csv Pg_AMP_data.csv --output-dir results
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


FEATURE_COLUMNS = [
    "cLogP",
    "Net_charge",
    "Length",
    "Cationic_comp",
    "Hydrophobic_comp",
    "Has_neutral",
]

LABEL_COLUMN = "Class_gingivalis"

MODEL_NAMES = [
    "Random Forest",
    "Logistic Regression",
    "XGBoost",
]

MODEL_COLORS = {
    "Random Forest": "#E76F61",
    "Logistic Regression": "#6BB7D6",
    "XGBoost": "#3BAE9F",
}


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducible model evaluation."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Random Forest, Logistic Regression, and XGBoost models "
            "for antimicrobial peptide classification."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV file containing descriptor columns and Class_gingivalis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figure1_model_performance"),
        help="Directory where figures and CSV result files will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible cross-validation.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    return parser.parse_args()


def load_and_clean_data(input_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and apply numeric cleaning used for model evaluation."""
    data = pd.read_csv(input_csv, encoding="utf-8")

    print(f"Loaded rows: {len(data)}")
    print(f"Available columns: {data.columns.tolist()}")

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{missing_columns}. Available columns: {data.columns.tolist()}"
        )

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    clean_data = data.dropna(subset=required_columns).copy()
    clean_data = clean_data[clean_data[LABEL_COLUMN].isin([0, 1])].copy()

    features = clean_data[FEATURE_COLUMNS].copy()
    labels = clean_data[LABEL_COLUMN].astype(int).copy()

    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "Both classes must be present after cleaning. "
            "Class_gingivalis must contain both 0 and 1."
        )

    print(f"Rows after cleaning: {len(clean_data)}")
    print(f"Rows removed during cleaning: {len(data) - len(clean_data)}")
    print(f"Class 1 count: {n_positive}")
    print(f"Class 0 count: {n_negative}")
    print(f"cLogP minimum: {features['cLogP'].min()}")
    print(f"cLogP maximum: {features['cLogP'].max()}")

    return features, labels


def build_models(seed: int, labels: pd.Series) -> dict[str, object]:
    """Create the three machine-learning models used in the comparison."""
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            criterion="entropy",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=n_negative / n_positive,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
    }


def evaluate_models(
    features: pd.DataFrame,
    labels: pd.Series,
    models: dict[str, object],
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate models with cross-validation and return summary tables."""
    summary_rows = []
    fold_rows = []

    for model_name, model in models.items():
        ap_values = []
        accuracy_values = []
        recall_values = []
        precision_values = []
        f1_values = []

        for fold_id, (train_idx, val_idx) in enumerate(cv.split(features, labels), start=1):
            x_train = features.iloc[train_idx]
            x_val = features.iloc[val_idx]
            y_train = labels.iloc[train_idx]
            y_val = labels.iloc[val_idx]

            classifier = clone(model)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_val)
            y_prob = classifier.predict_proba(x_val)[:, 1]

            ap = average_precision_score(y_val, y_prob)
            accuracy = accuracy_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            ap_values.append(ap)
            accuracy_values.append(accuracy)
            recall_values.append(recall)
            precision_values.append(precision)
            f1_values.append(f1)

            fold_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold_id,
                    "AP": ap,
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1": f1,
                }
            )

        summary_rows.append(
            {
                "Model": model_name,
                "AP_mean": np.mean(ap_values),
                "AP_std": np.std(ap_values, ddof=1),
                "Accuracy_mean": np.mean(accuracy_values),
                "Accuracy_std": np.std(accuracy_values, ddof=1),
                "Recall_mean": np.mean(recall_values),
                "Recall_std": np.std(recall_values, ddof=1),
                "Precision_mean": np.mean(precision_values),
                "Precision_std": np.std(precision_values, ddof=1),
                "F1_mean": np.mean(f1_values),
                "F1_std": np.std(f1_values, ddof=1),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def calculate_mean_roc_curves(
    features: pd.DataFrame,
    labels: pd.Series,
    models: dict[str, object],
    cv: StratifiedKFold,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray | float]], pd.DataFrame]:
    """Calculate interpolated mean ROC curves across cross-validation folds."""
    mean_fpr = np.linspace(0, 1, 100)
    roc_summary = {}
    roc_rows = []

    for model_name, model in models.items():
        tprs = []
        auc_values = []

        for fold_id, (train_idx, val_idx) in enumerate(cv.split(features, labels), start=1):
            del fold_id
            x_train = features.iloc[train_idx]
            x_val = features.iloc[val_idx]
            y_train = labels.iloc[train_idx]
            y_val = labels.iloc[val_idx]

            classifier = clone(model)
            classifier.fit(x_train, y_train)
            y_prob = classifier.predict_proba(x_val)[:, 1]

            fpr, tpr, _ = roc_curve(y_val, y_prob)
            auc_values.append(auc(fpr, tpr))

            interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
            interpolated_tpr[0] = 0.0
            tprs.append(interpolated_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        std_tpr = np.std(tprs, axis=0, ddof=1)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(auc_values, ddof=1)

        roc_summary[model_name] = {
            "mean_tpr": mean_tpr,
            "std_tpr": std_tpr,
            "tpr_upper": tpr_upper,
            "tpr_lower": tpr_lower,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        }

        for fpr_value, mean_tpr_value, lower_value, upper_value in zip(
            mean_fpr,
            mean_tpr,
            tpr_lower,
            tpr_upper,
        ):
            roc_rows.append(
                {
                    "Model": model_name,
                    "FPR": fpr_value,
                    "Mean_TPR": mean_tpr_value,
                    "TPR_lower": lower_value,
                    "TPR_upper": upper_value,
                    "Mean_AUC": mean_auc,
                    "AUC_std": std_auc,
                }
            )

    return mean_fpr, roc_summary, pd.DataFrame(roc_rows)


def configure_figure_style() -> None:
    """Apply publication-oriented Matplotlib style settings."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def plot_model_performance(
    fold_metrics: pd.DataFrame,
    mean_fpr: np.ndarray,
    roc_summary: dict[str, dict[str, np.ndarray | float]],
    output_png: Path,
    output_pdf: Path,
) -> None:
    """Create the AP boxplot and three-model ROC comparison figure."""
    configure_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)
    ax_ap, ax_roc = axes

    ap_data = [
        fold_metrics[fold_metrics["Model"] == model_name]["AP"].values
        for model_name in MODEL_NAMES
    ]

    boxplot = ax_ap.boxplot(
        ap_data,
        labels=MODEL_NAMES,
        patch_artist=True,
        widths=0.42,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 0.9},
        capprops={"color": "black", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "color": "black"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3.5,
            "alpha": 0.85,
        },
    )

    for patch, model_name in zip(boxplot["boxes"], MODEL_NAMES):
        patch.set_facecolor(MODEL_COLORS[model_name])
        patch.set_alpha(0.88)

    for index, model_name in enumerate(MODEL_NAMES):
        mean_ap = fold_metrics[fold_metrics["Model"] == model_name]["AP"].mean()
        text_y = min(mean_ap + 0.010, 1.015)

        ax_ap.scatter(
            index + 1,
            mean_ap,
            color="red",
            s=45,
            zorder=5,
            label="Mean AP" if index == 0 else "",
        )
        ax_ap.text(
            index + 1,
            text_y,
            f"{mean_ap:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax_ap.set_ylim(0.60, 1.03)
    ax_ap.set_ylabel("Average Precision (AP)", fontsize=8, fontweight="bold")
    ax_ap.set_xlabel("Machine Learning Models", fontsize=8, fontweight="bold")
    ax_ap.set_title("Average Precision Across Models", fontsize=8, fontweight="bold", pad=8)
    ax_ap.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax_ap.legend(
        loc="upper right",
        fontsize=7,
        frameon=True,
        borderpad=0.4,
        handletextpad=0.4,
    )
    ax_ap.tick_params(axis="both", labelsize=8, width=0.8)
    ax_ap.text(
        -0.06,
        1.04,
        "a",
        transform=ax_ap.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    for model_name in MODEL_NAMES:
        mean_auc = float(roc_summary[model_name]["mean_auc"])
        std_auc = float(roc_summary[model_name]["std_auc"])

        ax_roc.plot(
            mean_fpr,
            roc_summary[model_name]["mean_tpr"],
            color=MODEL_COLORS[model_name],
            linewidth=1.8,
            label=f"{model_name} AUC = {mean_auc:.3f} +/- {std_auc:.3f}",
        )
        ax_roc.fill_between(
            mean_fpr,
            roc_summary[model_name]["tpr_lower"],
            roc_summary[model_name]["tpr_upper"],
            color=MODEL_COLORS[model_name],
            alpha=0.12,
            linewidth=0,
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.set_xlabel("False Positive Rate", fontsize=8, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontsize=8, fontweight="bold")
    ax_roc.set_title("ROC Curves Across Models", fontsize=8, fontweight="bold", pad=8)
    ax_roc.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.25)
    ax_roc.legend(
        loc="lower right",
        fontsize=7,
        frameon=True,
        borderpad=0.4,
        handlelength=2.0,
    )
    ax_roc.tick_params(axis="both", labelsize=8, width=0.8)
    ax_roc.text(
        -0.06,
        1.04,
        "b",
        transform=ax_roc.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.88, wspace=0.28)
    plt.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    """Run the complete model-performance evaluation workflow."""
    args = parse_args()
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_png = args.output_dir / "figure1_model_performance.png"
    output_pdf = args.output_dir / "figure1_model_performance.pdf"
    output_summary_csv = args.output_dir / "figure1_model_performance_summary.csv"
    output_fold_csv = args.output_dir / "figure1_each_fold_metrics.csv"
    output_roc_csv = args.output_dir / "figure1_mean_roc_curves.csv"

    features, labels = load_and_clean_data(args.input_csv)
    models = build_models(args.seed, labels)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    summary_metrics, fold_metrics = evaluate_models(features, labels, models, cv)
    mean_fpr, roc_summary, roc_curves = calculate_mean_roc_curves(
        features,
        labels,
        models,
        cv,
    )

    print("\n========== Cross-validation performance ==========")
    print(summary_metrics.to_string(index=False))

    print("\n========== ROC AUC results ==========")
    for model_name in MODEL_NAMES:
        print(
            f"{model_name}: Mean AUC = {roc_summary[model_name]['mean_auc']:.4f}, "
            f"AUC SD = {roc_summary[model_name]['std_auc']:.4f}"
        )

    summary_metrics.to_csv(output_summary_csv, index=False, encoding="utf-8")
    fold_metrics.to_csv(output_fold_csv, index=False, encoding="utf-8")
    roc_curves.to_csv(output_roc_csv, index=False, encoding="utf-8")

    plot_model_performance(
        fold_metrics=fold_metrics,
        mean_fpr=mean_fpr,
        roc_summary=roc_summary,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    print("\nSaved outputs:")
    print(output_png)
    print(output_pdf)
    print(output_summary_csv)
    print(output_fold_csv)
    print(output_roc_csv)


if __name__ == "__main__":
    main()
