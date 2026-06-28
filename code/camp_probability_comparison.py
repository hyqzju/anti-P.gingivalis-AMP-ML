"""Compare CAMP prediction probabilities across model and rule groups.

This script reads CAMP result tables for three models (RF, ANN, and SVM)
under rule-matching and rule-mismatching conditions. It exports a combined
long-format dataset, summary statistics, and a four-panel figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


MODEL_ORDER = ["RF", "ANN", "SVM"]
GROUP_ORDER = ["Rule-matching", "Rule-mismatching"]
PALETTE = {"Rule-matching": "#c7362b", "Rule-mismatching": "#2f8bc5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CAMP probability comparison plots and summary tables."
    )
    parser.add_argument("--ann-match", required=True, type=Path, help="ANN rule-matching result table.")
    parser.add_argument("--svm-match", required=True, type=Path, help="SVM rule-matching result table.")
    parser.add_argument("--rf-match", required=True, type=Path, help="RF rule-matching result table.")
    parser.add_argument(
        "--ann-mismatch", required=True, type=Path, help="ANN rule-mismatching result table."
    )
    parser.add_argument(
        "--svm-mismatch", required=True, type=Path, help="SVM rule-mismatching result table."
    )
    parser.add_argument(
        "--rf-mismatch", required=True, type=Path, help="RF rule-mismatching result table."
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs"),
        type=Path,
        help="Directory for generated CSV files and figures.",
    )
    parser.add_argument(
        "--threshold",
        default=0.8,
        type=float,
        help="Probability threshold used for candidate proportions.",
    )
    return parser.parse_args()


def read_result_table(path: Path, model: str, group: str) -> pd.DataFrame:
    """Read and normalize one CAMP result table."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    df.columns = [column.strip() for column in df.columns]
    df = df.rename(
        columns={
            "Seq. ID.": "Seq_ID",
            "Class": "Class",
            "AMP Probability": "AMP_probability",
        }
    )

    if "AMP_probability" not in df.columns:
        raise ValueError(f"Missing required column 'AMP Probability' in {path}")

    df["AMP_probability"] = pd.to_numeric(df["AMP_probability"], errors="coerce")
    df = df.dropna(subset=["AMP_probability"]).copy()
    df["Model"] = model
    df["Group"] = group
    return df


def load_data(input_files: dict[tuple[str, str], Path]) -> pd.DataFrame:
    frames = [
        read_result_table(path, model, group)
        for (model, group), path in input_files.items()
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_data(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return data.groupby(["Model", "Group"], as_index=False).agg(
        n=("AMP_probability", "size"),
        mean_probability=("AMP_probability", "mean"),
        median_probability=("AMP_probability", "median"),
        proportion_ge_threshold=("AMP_probability", lambda values: (values >= threshold).mean()),
        count_ge_threshold=("AMP_probability", lambda values: (values >= threshold).sum()),
    )


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "Mann-Whitney P < 0.001"
    return f"Mann-Whitney P = {p_value:.3g}"


def draw_probability_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    model: str,
    panel_label: str,
    threshold: float,
) -> None:
    subset = data[data["Model"] == model]

    sns.boxplot(
        data=subset,
        x="Group",
        y="AMP_probability",
        order=GROUP_ORDER,
        hue="Group",
        palette=PALETTE,
        width=0.38,
        showfliers=False,
        linewidth=1.2,
        legend=False,
        ax=ax,
    )

    matching = subset[subset["Group"] == "Rule-matching"]["AMP_probability"]
    mismatching = subset[subset["Group"] == "Rule-mismatching"]["AMP_probability"]
    p_value = mannwhitneyu(matching, mismatching, alternative="two-sided").pvalue

    ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel(f"{model} probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER)
    ax.text(-0.18, 1.05, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=14)
    ax.text(0.04, 0.93, format_p_value(p_value), transform=ax.transAxes, fontsize=10)


def draw_threshold_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    threshold: float,
) -> None:
    sns.barplot(
        data=summary,
        x="Model",
        y="proportion_ge_threshold",
        hue="Group",
        order=MODEL_ORDER,
        hue_order=GROUP_ORDER,
        palette=PALETTE,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )

    for container in ax.containers:
        labels = [f"{bar.get_height() * 100:.1f}%" for bar in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("")
    ax.set_ylabel(f"Proportion of candidates (prob >= {threshold:g})")
    ax.text(-0.18, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=14)
    ax.legend(frameon=False, loc="upper right")


def create_figure(data: pd.DataFrame, summary: pd.DataFrame, threshold: float) -> plt.Figure:
    sns.set_theme(style="whitegrid", font="Arial")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    axes = axes.flatten()

    for index, model in enumerate(MODEL_ORDER):
        draw_probability_panel(
            ax=axes[index],
            data=data,
            model=model,
            panel_label="abc"[index],
            threshold=threshold,
        )

    draw_threshold_panel(ax=axes[3], summary=summary, threshold=threshold)
    plt.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = {
        ("ANN", "Rule-matching"): args.ann_match,
        ("SVM", "Rule-matching"): args.svm_match,
        ("RF", "Rule-matching"): args.rf_match,
        ("ANN", "Rule-mismatching"): args.ann_mismatch,
        ("SVM", "Rule-mismatching"): args.svm_mismatch,
        ("RF", "Rule-mismatching"): args.rf_mismatch,
    }

    data = load_data(input_files)
    summary = summarize_data(data, args.threshold)

    data.to_csv(args.output_dir / "camp_long.csv", index=False)
    summary.to_csv(args.output_dir / "camp_summary.csv", index=False)

    fig = create_figure(data, summary, args.threshold)
    fig.savefig(args.output_dir / "figure3.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "figure3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Analysis completed. Files saved to: {args.output_dir.resolve()}")
    print("Generated files: camp_long.csv, camp_summary.csv, figure3.pdf, figure3.png")


if __name__ == "__main__":
    main()
