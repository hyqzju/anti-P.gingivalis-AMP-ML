"""Generate SHAP interpretation results for a P. gingivalis AMP model.

This script trains a Random Forest classifier, computes SHAP values, exports
SHAP-related statistics, and creates a four-panel publication-style figure.

Outputs:
    - SHAP statistics workbook
    - Four-panel SHAP interpretation figure as PNG, PDF, and SVG

Example:
    python shap_interpretation_figure.py --input-csv Pg_AMP_data.csv --output-dir results
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "The shap package is required. Install it with one of the following commands: "
        "pip install shap or conda install -c conda-forge shap"
    ) from exc


FEATURE_COLUMNS = [
    "cLogP",
    "Net_charge",
    "Length",
    "Cationic_comp",
    "Hydrophobic_comp",
    "Has_neutral",
]

LABEL_COLUMN = "Class_gingivalis"

SHAP_INTERVALS = pd.DataFrame(
    {
        "Feature": [
            "cLogP",
            "Net charge",
            "Length",
            "Cationic composition",
            "Hydrophobic composition",
        ],
        "Lower": [-5.768475, 5.75, 13, 0.2902, 0.3824],
        "Upper": [-1.4346, 10.00, 28.5, 0.367075, 0.5833],
        "Unit": ["", "", "aa", "fraction", "fraction"],
    }
)


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducible analysis."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a Random Forest model and generate SHAP interpretation "
            "outputs for antimicrobial peptide classification."
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
        default=Path("figure3_shap_interpretation"),
        help="Directory where the SHAP statistics and figures will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible model training and plotting.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    return parser.parse_args()


def load_and_clean_data(input_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load data and apply numeric cleaning to features and labels."""
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

    return features, labels, clean_data


def build_random_forest(seed: int) -> RandomForestClassifier:
    """Build the Random Forest classifier used for SHAP interpretation."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        criterion="entropy",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def calculate_cross_validated_metrics(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    n_splits: int,
    seed: int,
) -> tuple[float, float]:
    """Calculate cross-validated AP and ROC AUC as supplementary metrics."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predicted_probabilities = cross_val_predict(
        model,
        features,
        labels,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    ap_score = average_precision_score(labels, predicted_probabilities)
    auc_score = roc_auc_score(labels, predicted_probabilities)
    return ap_score, auc_score


def calculate_shap_values(
    model: RandomForestClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """Calculate SHAP values for the positive class."""
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(features)

    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1]
    elif len(np.array(raw_shap_values).shape) == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    return np.array(shap_values)


def build_importance_table(shap_values: np.ndarray) -> pd.DataFrame:
    """Create a feature-importance table from mean absolute SHAP values."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Mean_abs_SHAP": mean_abs_shap,
        }
    ).sort_values("Mean_abs_SHAP", ascending=False)


def save_shap_statistics(
    output_excel: Path,
    clean_data: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
    shap_values: np.ndarray,
    importance_table: pd.DataFrame,
) -> None:
    """Save training data, SHAP values, importance, and design intervals."""
    shap_table = pd.DataFrame(
        shap_values,
        columns=[f"SHAP_{column}" for column in FEATURE_COLUMNS],
    )

    feature_value_table = features.reset_index(drop=True).copy()
    feature_value_table[LABEL_COLUMN] = labels.reset_index(drop=True)
    shap_output_table = pd.concat([feature_value_table, shap_table], axis=1)

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        clean_data.to_excel(writer, sheet_name="Training_data_clean", index=False)
        importance_table.to_excel(writer, sheet_name="SHAP_importance", index=False)
        shap_output_table.to_excel(writer, sheet_name="SHAP_values", index=False)
        SHAP_INTERVALS.to_excel(writer, sheet_name="SHAP_intervals", index=False)


def configure_plot_style() -> None:
    """Apply publication-oriented Matplotlib style settings."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def plot_shap_interpretation(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    importance_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
    seed: int,
) -> None:
    """Create a four-panel SHAP interpretation figure."""
    configure_plot_style()

    main_blue = "#4C9FD0"
    main_red = "#C0392B"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4), dpi=300)
    ax_importance, ax_beeswarm, ax_dependence, ax_intervals = axes.flatten()

    importance_for_plot = importance_table.sort_values("Mean_abs_SHAP", ascending=True)
    ax_importance.barh(
        importance_for_plot["Feature"],
        importance_for_plot["Mean_abs_SHAP"],
        color=main_blue,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.90,
    )
    ax_importance.set_xlabel("Mean absolute SHAP value")
    ax_importance.set_ylabel("Feature")
    ax_importance.grid(axis="x", alpha=0.25, linewidth=0.6)
    add_panel_label(ax_importance, "a")

    ordered_features = importance_table["Feature"].tolist()
    rng = np.random.default_rng(seed)
    scatter = None

    for index, feature_name in enumerate(ordered_features):
        feature_index = FEATURE_COLUMNS.index(feature_name)
        shap_feature = shap_values[:, feature_index]
        feature_values = features[feature_name].values
        y_position = np.full_like(shap_feature, fill_value=index, dtype=float)
        y_position = y_position + rng.normal(0, 0.08, size=len(y_position))

        scatter = ax_beeswarm.scatter(
            shap_feature,
            y_position,
            c=feature_values,
            cmap="coolwarm",
            s=22,
            alpha=0.85,
            edgecolor="none",
            rasterized=True,
        )

    ax_beeswarm.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.75)
    ax_beeswarm.set_yticks(range(len(ordered_features)))
    ax_beeswarm.set_yticklabels(ordered_features)
    ax_beeswarm.invert_yaxis()
    ax_beeswarm.set_xlabel("SHAP value for active prediction")
    ax_beeswarm.set_ylabel("Feature")
    ax_beeswarm.grid(axis="x", alpha=0.25, linewidth=0.6)

    if scatter is not None:
        colorbar = plt.colorbar(scatter, ax=ax_beeswarm, fraction=0.046, pad=0.04)
        colorbar.set_label("Feature value", fontsize=9)

    add_panel_label(ax_beeswarm, "b")

    clogp_values = features["cLogP"].values
    shap_clogp = shap_values[:, FEATURE_COLUMNS.index("cLogP")]
    net_charge_values = features["Net_charge"].values

    dependence_scatter = ax_dependence.scatter(
        clogp_values,
        shap_clogp,
        c=net_charge_values,
        cmap="viridis",
        s=42,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.25,
    )
    ax_dependence.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.75)
    ax_dependence.axvspan(
        -5.768475,
        -1.4346,
        color="#F8E5C2",
        alpha=0.45,
        label="Recommended cLogP window",
    )
    ax_dependence.set_xlabel("cLogP")
    ax_dependence.set_ylabel("SHAP value for cLogP")
    ax_dependence.grid(alpha=0.25, linewidth=0.6)

    colorbar2 = plt.colorbar(dependence_scatter, ax=ax_dependence, fraction=0.046, pad=0.04)
    colorbar2.set_label("Net charge", fontsize=9)
    ax_dependence.legend(frameon=False, fontsize=8, loc="best")
    add_panel_label(ax_dependence, "c")

    interval_plot = SHAP_INTERVALS.copy()
    interval_plot["Lower_plot"] = interval_plot["Lower"]
    interval_plot["Upper_plot"] = interval_plot["Upper"]
    interval_plot["Label"] = interval_plot["Feature"]

    for index, row in interval_plot.iterrows():
        if "composition" in row["Feature"]:
            interval_plot.loc[index, "Lower_plot"] = row["Lower"] * 100
            interval_plot.loc[index, "Upper_plot"] = row["Upper"] * 100
            interval_plot.loc[index, "Label"] = row["Feature"] + " (%)"

    y_positions = np.arange(len(interval_plot))
    for index, row in interval_plot.iterrows():
        lower = row["Lower_plot"]
        upper = row["Upper_plot"]
        midpoint = (lower + upper) / 2

        ax_intervals.plot(
            [lower, upper],
            [index, index],
            color=main_red,
            linewidth=5,
            solid_capstyle="round",
            zorder=2,
        )
        ax_intervals.scatter(
            [lower, upper],
            [index, index],
            color=main_red,
            edgecolor="black",
            s=45,
            zorder=3,
        )

        label_text = f"{lower:.2f}~{upper:.2f}"
        label_x = midpoint + 2.2 if row["Feature"] == "cLogP" else midpoint
        ax_intervals.text(
            label_x,
            index + 0.31,
            label_text,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            zorder=4,
        )

    ax_intervals.set_yticks(y_positions)
    ax_intervals.set_yticklabels(interval_plot["Label"])
    ax_intervals.invert_yaxis()
    ax_intervals.set_xlim(-9, 62)
    ax_intervals.set_ylim(len(interval_plot) - 0.15, -0.65)
    ax_intervals.set_xlabel("Recommended interval")
    ax_intervals.set_title("SHAP-informed design window", fontsize=10)
    ax_intervals.grid(axis="x", alpha=0.25, linewidth=0.6)
    add_panel_label(ax_intervals, "d")

    plt.tight_layout(w_pad=2.6, h_pad=2.4)
    plt.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    """Add a lowercase panel label to a Matplotlib axis."""
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )


def print_manuscript_summary(importance_table: pd.DataFrame) -> None:
    """Print a short manuscript-ready SHAP interpretation summary."""
    top_features = importance_table.head(5)["Feature"].tolist()

    print("\n========== Manuscript-ready summary ==========")
    print(
        "The RF-SHAP analysis identified the following dominant contributors "
        "to anti-P. gingivalis AMP prediction: "
        + ", ".join(top_features)
        + "."
    )

    clogp = SHAP_INTERVALS[SHAP_INTERVALS["Feature"] == "cLogP"].iloc[0]
    net_charge = SHAP_INTERVALS[SHAP_INTERVALS["Feature"] == "Net charge"].iloc[0]
    length = SHAP_INTERVALS[SHAP_INTERVALS["Feature"] == "Length"].iloc[0]
    cationic = SHAP_INTERVALS[SHAP_INTERVALS["Feature"] == "Cationic composition"].iloc[0]
    hydrophobic = SHAP_INTERVALS[
        SHAP_INTERVALS["Feature"] == "Hydrophobic composition"
    ].iloc[0]

    print(
        "Based on SHAP-informed prioritization, the recommended physicochemical "
        f"design window was defined as cLogP {clogp['Lower']:.2f} to "
        f"{clogp['Upper']:.2f}, net charge {net_charge['Lower']:.2f} to "
        f"{net_charge['Upper']:.2f}, length {length['Lower']:.0f} to "
        f"{length['Upper']:.1f} residues, cationic composition "
        f"{cationic['Lower'] * 100:.2f}% to {cationic['Upper'] * 100:.2f}%, "
        f"and hydrophobic composition {hydrophobic['Lower'] * 100:.2f}% to "
        f"{hydrophobic['Upper'] * 100:.2f}%."
    )


def main() -> None:
    """Run the full RF-SHAP interpretation workflow."""
    args = parse_args()
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_excel = args.output_dir / "figure3_shap_statistics.xlsx"
    output_png = args.output_dir / "figure3_shap_interpretation.png"
    output_pdf = args.output_dir / "figure3_shap_interpretation.pdf"
    output_svg = args.output_dir / "figure3_shap_interpretation.svg"

    features, labels, clean_data = load_and_clean_data(args.input_csv)
    model = build_random_forest(args.seed)
    model.fit(features, labels)
    print("Random Forest model training completed.")

    ap_score, auc_score = calculate_cross_validated_metrics(
        model=model,
        features=features,
        labels=labels,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    print(f"{args.n_splits}-fold CV AP: {ap_score:.6f}")
    print(f"{args.n_splits}-fold CV AUC: {auc_score:.6f}")

    shap_values = calculate_shap_values(model, features)
    print(f"SHAP values shape: {shap_values.shape}")

    importance_table = build_importance_table(shap_values)
    print("\nSHAP feature importance:")
    print(importance_table.to_string(index=False))

    save_shap_statistics(
        output_excel=output_excel,
        clean_data=clean_data,
        features=features,
        labels=labels,
        shap_values=shap_values,
        importance_table=importance_table,
    )

    plot_shap_interpretation(
        features=features,
        shap_values=shap_values,
        importance_table=importance_table,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        seed=args.seed,
    )

    print("\nSaved outputs:")
    print(output_excel)
    print(output_png)
    print(output_pdf)
    print(output_svg)

    print_manuscript_summary(importance_table)


if __name__ == "__main__":
    main()
