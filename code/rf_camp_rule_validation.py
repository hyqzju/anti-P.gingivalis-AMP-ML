"""Validate SHAP-informed AMP design rules with RF and CAMP probabilities.

This script compares rule-matching and rule-mismatching peptide candidates
using probabilities from a P. gingivalis-specific Random Forest model and CAMP.
It exports statistical summaries and creates a four-panel validation figure.

Expected input:
    An Excel workbook containing a sheet named "Merged_RF_CAMP" with columns:
    RF_probability, CAMP_probability, and Group.

Outputs:
    - Statistical summary workbook
    - Four-panel validation figure as PNG, PDF, and SVG

Example:
    python rf_camp_rule_validation_figure.py --input-excel RF_CAMP_summary.xlsx --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr


DEFAULT_SHEET_NAME = "Merged_RF_CAMP"
REQUIRED_COLUMNS = ["RF_probability", "CAMP_probability", "Group"]
MATCHING_GROUP = "Rule-matching"
MISMATCHING_GROUP = "Rule-mismatching"
THRESHOLD = 0.80

COLOR_MATCHING = "#C0392B"
COLOR_MISMATCHING = "#2E86C1"
GROUP_COLORS = [COLOR_MATCHING, COLOR_MISMATCHING]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate SHAP-informed antimicrobial peptide design rules using "
            "RF and CAMP probability scores."
        )
    )
    parser.add_argument(
        "--input-excel",
        type=Path,
        required=True,
        help="Input Excel file containing RF, CAMP, and group annotations.",
    )
    parser.add_argument(
        "--sheet-name",
        default=DEFAULT_SHEET_NAME,
        help="Excel sheet name to read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rf_camp_rule_validation"),
        help="Directory where statistical tables and figures will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Probability threshold used to define high-confidence candidates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for point sampling and jitter.",
    )
    parser.add_argument(
        "--max-scatter-points-per-group",
        type=int,
        default=3500,
        help="Maximum number of points per group shown in the scatter panel.",
    )
    return parser.parse_args()


def load_and_clean_data(
    input_excel: Path,
    sheet_name: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean RF-CAMP validation data."""
    data = pd.read_excel(input_excel, sheet_name=sheet_name)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The input file is missing required columns: "
            f"{missing_columns}. Available columns: {data.columns.tolist()}"
        )

    data["RF_probability"] = pd.to_numeric(data["RF_probability"], errors="coerce")
    data["CAMP_probability"] = pd.to_numeric(data["CAMP_probability"], errors="coerce")
    data = data.dropna(subset=REQUIRED_COLUMNS).copy()

    data["RF_high"] = data["RF_probability"] >= threshold
    data["CAMP_high"] = data["CAMP_probability"] >= threshold
    data["Dual_high"] = data["RF_high"] & data["CAMP_high"]

    matching = data[data["Group"] == MATCHING_GROUP].copy()
    mismatching = data[data["Group"] == MISMATCHING_GROUP].copy()

    if len(matching) == 0 or len(mismatching) == 0:
        raise ValueError(
            "Both Rule-matching and Rule-mismatching groups must be present "
            "in the Group column."
        )

    print(f"Rows after cleaning: {len(data)}")
    print(data["Group"].value_counts().to_string())

    return data, matching, mismatching


def calculate_statistics(
    data: pd.DataFrame,
    matching: pd.DataFrame,
    mismatching: pd.DataFrame,
) -> dict[str, object]:
    """Calculate group comparisons, correlations, and enrichment statistics."""
    rf_u, rf_p = mannwhitneyu(
        matching["RF_probability"],
        mismatching["RF_probability"],
        alternative="two-sided",
    )
    camp_u, camp_p = mannwhitneyu(
        matching["CAMP_probability"],
        mismatching["CAMP_probability"],
        alternative="two-sided",
    )

    all_pearson_r, all_pearson_p = pearsonr(
        data["RF_probability"],
        data["CAMP_probability"],
    )
    all_spearman_r, all_spearman_p = spearmanr(
        data["RF_probability"],
        data["CAMP_probability"],
    )
    matching_pearson_r, matching_pearson_p = pearsonr(
        matching["RF_probability"],
        matching["CAMP_probability"],
    )
    mismatching_pearson_r, mismatching_pearson_p = pearsonr(
        mismatching["RF_probability"],
        mismatching["CAMP_probability"],
    )
    matching_spearman_r, matching_spearman_p = spearmanr(
        matching["RF_probability"],
        matching["CAMP_probability"],
    )
    mismatching_spearman_r, mismatching_spearman_p = spearmanr(
        mismatching["RF_probability"],
        mismatching["CAMP_probability"],
    )

    rf_table, rf_odds_ratio, rf_high_p = fisher_by_group(data, "RF_high")
    camp_table, camp_odds_ratio, camp_high_p = fisher_by_group(data, "CAMP_high")
    dual_table, dual_odds_ratio, dual_high_p = fisher_by_group(data, "Dual_high")

    summary_table = build_summary_table(data)
    correlation_table = pd.DataFrame(
        [
            {
                "Group": "All",
                "N": len(data),
                "Pearson_r": all_pearson_r,
                "Pearson_p": all_pearson_p,
                "Spearman_rho": all_spearman_r,
                "Spearman_p": all_spearman_p,
            },
            {
                "Group": MATCHING_GROUP,
                "N": len(matching),
                "Pearson_r": matching_pearson_r,
                "Pearson_p": matching_pearson_p,
                "Spearman_rho": matching_spearman_r,
                "Spearman_p": matching_spearman_p,
            },
            {
                "Group": MISMATCHING_GROUP,
                "N": len(mismatching),
                "Pearson_r": mismatching_pearson_r,
                "Pearson_p": mismatching_pearson_p,
                "Spearman_rho": mismatching_spearman_r,
                "Spearman_p": mismatching_spearman_p,
            },
        ]
    )
    test_table = pd.DataFrame(
        [
            {
                "Comparison": "RF probability distribution",
                "Test": "Mann-Whitney U",
                "Statistic_or_OR": rf_u,
                "P_value": rf_p,
            },
            {
                "Comparison": "CAMP probability distribution",
                "Test": "Mann-Whitney U",
                "Statistic_or_OR": camp_u,
                "P_value": camp_p,
            },
            {
                "Comparison": "RF >= 0.8 proportion",
                "Test": "Fisher exact",
                "Statistic_or_OR": rf_odds_ratio,
                "P_value": rf_high_p,
            },
            {
                "Comparison": "CAMP >= 0.8 proportion",
                "Test": "Fisher exact",
                "Statistic_or_OR": camp_odds_ratio,
                "P_value": camp_high_p,
            },
            {
                "Comparison": "Dual RF >= 0.8 and CAMP >= 0.8 proportion",
                "Test": "Fisher exact",
                "Statistic_or_OR": dual_odds_ratio,
                "P_value": dual_high_p,
            },
        ]
    )

    quadrant_table, overall_quadrant_table = build_quadrant_tables(data)

    return {
        "rf_p": rf_p,
        "camp_p": camp_p,
        "all_pearson_r": all_pearson_r,
        "all_spearman_r": all_spearman_r,
        "summary_table": summary_table,
        "correlation_table": correlation_table,
        "test_table": test_table,
        "quadrant_table": quadrant_table,
        "overall_quadrant_table": overall_quadrant_table,
        "rf_table": rf_table,
        "camp_table": camp_table,
        "dual_table": dual_table,
    }


def fisher_by_group(data: pd.DataFrame, binary_column: str) -> tuple[pd.DataFrame, float, float]:
    """Run Fisher's exact test for a binary endpoint by group."""
    table = pd.crosstab(data["Group"], data[binary_column])
    table = table.reindex(
        index=[MATCHING_GROUP, MISMATCHING_GROUP],
        columns=[False, True],
        fill_value=0,
    )
    odds_ratio, p_value = fisher_exact(table.values)
    return table, odds_ratio, p_value


def build_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """Build group-level descriptive statistics."""
    summary_rows = []
    for group, subset in data.groupby("Group"):
        summary_rows.append(
            {
                "Group": group,
                "N": len(subset),
                "RF_probability_mean": subset["RF_probability"].mean(),
                "RF_probability_median": subset["RF_probability"].median(),
                "CAMP_probability_mean": subset["CAMP_probability"].mean(),
                "CAMP_probability_median": subset["CAMP_probability"].median(),
                "RF_high_count": int(subset["RF_high"].sum()),
                "RF_high_rate": subset["RF_high"].mean(),
                "CAMP_high_count": int(subset["CAMP_high"].sum()),
                "CAMP_high_rate": subset["CAMP_high"].mean(),
                "Dual_high_count": int(subset["Dual_high"].sum()),
                "Dual_high_rate": subset["Dual_high"].mean(),
            }
        )
    return pd.DataFrame(summary_rows)


def build_quadrant_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign RF-CAMP quadrants and summarize counts."""
    data["Quadrant"] = np.select(
        [
            data["RF_high"] & data["CAMP_high"],
            data["RF_high"] & (~data["CAMP_high"]),
            (~data["RF_high"]) & data["CAMP_high"],
            (~data["RF_high"]) & (~data["CAMP_high"]),
        ],
        [
            "RF_high_CAMP_high",
            "RF_high_CAMP_low",
            "RF_low_CAMP_high",
            "RF_low_CAMP_low",
        ],
        default="Unknown",
    )

    quadrant_table = data.groupby(["Group", "Quadrant"]).size().reset_index(name="Count")
    quadrant_table["Rate_within_group"] = quadrant_table.groupby("Group")[
        "Count"
    ].transform(lambda values: values / values.sum())

    overall_quadrant_table = data.groupby("Quadrant").size().reset_index(name="Count")
    overall_quadrant_table["Rate_total"] = overall_quadrant_table["Count"] / len(data)

    return quadrant_table, overall_quadrant_table


def save_statistics(
    output_excel: Path,
    data: pd.DataFrame,
    statistics: dict[str, object],
) -> None:
    """Save validation data and statistical summaries to Excel."""
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Merged_data", index=False)
        statistics["summary_table"].to_excel(writer, sheet_name="Summary", index=False)
        statistics["correlation_table"].to_excel(writer, sheet_name="Correlation", index=False)
        statistics["test_table"].to_excel(writer, sheet_name="Statistical_tests", index=False)
        statistics["quadrant_table"].to_excel(
            writer,
            sheet_name="Quadrants_by_group",
            index=False,
        )
        statistics["overall_quadrant_table"].to_excel(
            writer,
            sheet_name="Quadrants_overall",
            index=False,
        )


def format_p_value(p_value: float) -> str:
    """Format a P value for figure annotation."""
    if p_value == 0 or p_value < 2.2e-16:
        return r"$P < 2.2 \times 10^{-16}$"
    if p_value < 0.001:
        return r"$P < 0.001$"
    if p_value < 0.01:
        return f"$P = {p_value:.3f}$"
    return f"$P = {p_value:.2f}$"


def configure_plot_style() -> None:
    """Apply publication-oriented Matplotlib style settings."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def add_panel_label(axis: plt.Axes, label: str) -> None:
    """Add a lowercase panel label to an axis."""
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


def add_violin_box_points(
    axis: plt.Axes,
    matching_values: pd.Series,
    mismatching_values: pd.Series,
    ylabel: str,
    panel_label: str,
    p_value: float,
    threshold: float,
    seed: int,
    p_position: str = "upper_right",
) -> None:
    """Add a violin plot, boxplot, sampled points, and P-value annotation."""
    data = [
        pd.Series(matching_values).dropna(),
        pd.Series(mismatching_values).dropna(),
    ]

    violin = axis.violinplot(
        data,
        positions=[1, 2],
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(GROUP_COLORS[index])
        body.set_edgecolor("black")
        body.set_alpha(0.22)
        body.set_linewidth(0.8)

    box = axis.boxplot(
        data,
        positions=[1, 2],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.8},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )

    for patch, color in zip(box["boxes"], GROUP_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.86)

    rng = np.random.default_rng(seed)
    n_matching = min(900, len(data[0]))
    n_mismatching = min(900, len(data[1]))
    matching_sample = data[0].sample(n_matching, random_state=seed)
    mismatching_sample = data[1].sample(n_mismatching, random_state=seed)

    axis.scatter(
        rng.normal(1, 0.055, size=n_matching),
        matching_sample,
        s=6,
        color=COLOR_MATCHING,
        alpha=0.16,
        edgecolor="none",
        rasterized=True,
    )
    axis.scatter(
        rng.normal(2, 0.055, size=n_mismatching),
        mismatching_sample,
        s=6,
        color=COLOR_MISMATCHING,
        alpha=0.16,
        edgecolor="none",
        rasterized=True,
    )

    axis.axhline(threshold, linestyle="--", color="black", linewidth=0.8, alpha=0.75)
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Rule-\nmatching", "Rule-\nmismatching"])
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.03, 1.03)
    axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    add_panel_label(axis, panel_label)

    p_text = f"Mann-Whitney {format_p_value(p_value)}"
    if p_position == "lower_left":
        axis.text(
            0.04,
            0.06,
            p_text,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
        )
    else:
        axis.text(
            0.97,
            0.96,
            p_text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
        )


def plot_validation_figure(
    data: pd.DataFrame,
    matching: pd.DataFrame,
    mismatching: pd.DataFrame,
    statistics: dict[str, object],
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
    threshold: float,
    seed: int,
    max_scatter_points_per_group: int,
) -> None:
    """Create the four-panel RF-CAMP validation figure."""
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), dpi=300)
    ax_rf, ax_camp, ax_enrichment, ax_scatter = axes.flatten()

    add_violin_box_points(
        axis=ax_rf,
        matching_values=matching["RF_probability"],
        mismatching_values=mismatching["RF_probability"],
        ylabel="P. gingivalis-specific RF probability",
        panel_label="a",
        p_value=float(statistics["rf_p"]),
        threshold=threshold,
        seed=seed,
        p_position="upper_right",
    )
    add_violin_box_points(
        axis=ax_camp,
        matching_values=matching["CAMP_probability"],
        mismatching_values=mismatching["CAMP_probability"],
        ylabel="CAMP AMP probability",
        panel_label="b",
        p_value=float(statistics["camp_p"]),
        threshold=threshold,
        seed=seed,
        p_position="lower_left",
    )

    metrics = [
        f"RF >= {threshold:.1f}",
        f"CAMP >= {threshold:.1f}",
        f"RF >= {threshold:.1f}\n& CAMP >= {threshold:.1f}",
    ]
    matching_rates = [
        matching["RF_high"].mean(),
        matching["CAMP_high"].mean(),
        matching["Dual_high"].mean(),
    ]
    mismatching_rates = [
        mismatching["RF_high"].mean(),
        mismatching["CAMP_high"].mean(),
        mismatching["Dual_high"].mean(),
    ]

    x = np.arange(len(metrics))
    width = 0.34
    bars_matching = ax_enrichment.bar(
        x - width / 2,
        matching_rates,
        width,
        color=COLOR_MATCHING,
        edgecolor="black",
        linewidth=0.8,
        label=f"Rule-matching (n={len(matching)})",
    )
    bars_mismatching = ax_enrichment.bar(
        x + width / 2,
        mismatching_rates,
        width,
        color=COLOR_MISMATCHING,
        edgecolor="black",
        linewidth=0.8,
        label=f"Rule-mismatching (n={len(mismatching)})",
    )

    ax_enrichment.set_xticks(x)
    ax_enrichment.set_xticklabels(metrics)
    ax_enrichment.set_ylabel("Proportion of candidates")
    ax_enrichment.set_ylim(0, 1.18)
    ax_enrichment.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax_enrichment.legend(frameon=False, fontsize=8, loc="upper right")
    add_panel_label(ax_enrichment, "c")

    for bars in [bars_matching, bars_mismatching]:
        for bar in bars:
            height = bar.get_height()
            ax_enrichment.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.025,
                f"{height * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plot_data = data.groupby("Group", group_keys=False).apply(
        lambda subset: subset.sample(
            min(len(subset), max_scatter_points_per_group),
            random_state=seed,
        )
    )

    for group, color in [
        (MISMATCHING_GROUP, COLOR_MISMATCHING),
        (MATCHING_GROUP, COLOR_MATCHING),
    ]:
        subset = plot_data[plot_data["Group"] == group]
        ax_scatter.scatter(
            subset["RF_probability"],
            subset["CAMP_probability"],
            s=7,
            alpha=0.18,
            color=color,
            edgecolor="none",
            rasterized=True,
            label=f"{group} (n={len(data[data['Group'] == group])})",
        )

    ax_scatter.axvline(threshold, linestyle="--", color="black", linewidth=0.9, alpha=0.75)
    ax_scatter.axhline(threshold, linestyle="--", color="black", linewidth=0.9, alpha=0.75)
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_xlabel("P. gingivalis-specific RF probability")
    ax_scatter.set_ylabel("CAMP AMP probability")
    ax_scatter.grid(alpha=0.22, linewidth=0.6)
    add_panel_label(ax_scatter, "d")

    correlation_text = (
        "All candidates\n"
        f"Pearson r = {statistics['all_pearson_r']:.3f}\n"
        f"Spearman rho = {statistics['all_spearman_r']:.3f}"
    )
    ax_scatter.text(
        0.04,
        0.96,
        correlation_text,
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={
            "boxstyle": "round,pad=0.30",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
    )

    n_dual_high = int(data["Dual_high"].sum())
    ax_scatter.text(
        0.815,
        0.835,
        f"Dual high-confidence\nn = {n_dual_high}",
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.4,
            "alpha": 0.78,
        },
    )
    ax_scatter.legend(frameon=False, loc="lower right", fontsize=8, markerscale=1.4)

    plt.tight_layout(w_pad=2.4, h_pad=2.4)
    plt.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_results(
    statistics: dict[str, object],
    matching: pd.DataFrame,
    mismatching: pd.DataFrame,
    threshold: float,
) -> None:
    """Print tables and a manuscript-ready result sentence."""
    print("\n========== Summary ==========")
    print(statistics["summary_table"].to_string(index=False))

    print("\n========== Correlation ==========")
    print(statistics["correlation_table"].to_string(index=False))

    print("\n========== Statistical tests ==========")
    print(statistics["test_table"].to_string(index=False))

    print("\n========== Suggested manuscript sentence ==========")
    print(
        "Rule-matching candidates showed higher P. gingivalis-specific RF "
        "probabilities and CAMP AMP probabilities than rule-mismatching controls. "
        "The proportion of dual high-confidence candidates "
        f"(RF probability >= {threshold:.1f} and CAMP probability >= {threshold:.1f}) "
        f"was {matching['Dual_high'].mean() * 100:.1f}% in the rule-matching "
        f"library compared with {mismatching['Dual_high'].mean() * 100:.1f}% "
        "in the rule-mismatching library. Across all candidates, RF and CAMP "
        f"probabilities were positively correlated (Pearson r = "
        f"{statistics['all_pearson_r']:.3f}; Spearman rho = "
        f"{statistics['all_spearman_r']:.3f})."
    )


def main() -> None:
    """Run the RF-CAMP validation workflow."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_excel = args.output_dir / "rf_camp_rule_validation_statistics.xlsx"
    output_png = args.output_dir / "rf_camp_rule_validation_figure.png"
    output_pdf = args.output_dir / "rf_camp_rule_validation_figure.pdf"
    output_svg = args.output_dir / "rf_camp_rule_validation_figure.svg"

    data, matching, mismatching = load_and_clean_data(
        input_excel=args.input_excel,
        sheet_name=args.sheet_name,
        threshold=args.threshold,
    )
    statistics = calculate_statistics(data, matching, mismatching)
    save_statistics(output_excel, data, statistics)
    plot_validation_figure(
        data=data,
        matching=matching,
        mismatching=mismatching,
        statistics=statistics,
        output_png=output_png,
        output_pdf=output_pdf,
        output_svg=output_svg,
        threshold=args.threshold,
        seed=args.seed,
        max_scatter_points_per_group=args.max_scatter_points_per_group,
    )

    print("\nSaved outputs:")
    print(output_excel)
    print(output_png)
    print(output_pdf)
    print(output_svg)

    print_results(statistics, matching, mismatching, args.threshold)


if __name__ == "__main__":
    main()
