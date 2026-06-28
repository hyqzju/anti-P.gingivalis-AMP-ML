[README_full_github.md](https://github.com/user-attachments/files/29432967/README_full_github.md)
# Antimicrobial Peptide Design Priciples for *P. gingivalis*

This repository contains the main Python scripts used for a computational machine-learning workflow to identify and interpret antimicrobial peptide candidates against *Porphyromonas gingivalis*.

The workflow includes model performance evaluation, random forest-based SHAP interpretation, and CAMP prediction probability comparison. The analyses are intended to support manuscript preparation, peer review, and research archiving.

This repository does not include wet-lab experimental validation data. The results should be interpreted as computational predictions and model-based analyses.

## Workflow Overview

The repository currently includes three main analysis scripts:

1. `model_performance.py`  
   Evaluates Random Forest, Logistic Regression, and XGBoost models using stratified 5-fold cross-validation. The script reports average precision, accuracy, recall, precision, F1 score, and ROC AUC, and generates a two-panel model performance figure.

2. `shap_interpretation.py`  
   Trains a Random Forest model and performs SHAP-based model interpretation. The script exports SHAP feature importance, sample-level SHAP values, SHAP-informed physicochemical design intervals, and a four-panel interpretation figure.

3. `camp_probability_comparison.py`  
   Compares CAMP prediction probabilities for rule-matching and rule-mismatching peptide candidates across ANN, SVM, and RF prediction outputs. The script exports combined prediction tables, summary statistics, and a four-panel comparison figure.

## Repository Structure

```text
.
├── data/                         # Input datasets, if shared
├── code/                         # Main Python scripts
│   ├── model_performance.py
│   ├── shap_interpretation.py
│   └── camp_probability_comparison.py
├── results/                      # Generated figures and tables
├── README.md
└── requirements.txt
```

If raw datasets cannot be publicly shared, the `data/` directory can be omitted or replaced with a short data availability note.

## Input Data

The model performance and SHAP scripts expect a peptide feature table containing the following columns:

- `cLogP`
- `Net_charge`
- `Length`
- `Cationic_comp`
- `Hydrophobic_comp`
- `Has_neutral`
- `Class_gingivalis`

The CAMP probability comparison script expects tab-separated CAMP result tables containing an `AMP Probability` column.

## Main Outputs

The scripts generate:

- Cross-validation performance tables
- Per-fold model metrics
- Mean ROC curve data
- Model performance figures
- SHAP feature importance tables
- Sample-level SHAP value tables
- SHAP-informed physicochemical design intervals
- CAMP prediction probability summaries
- Publication-style figures in PNG, PDF, and/or SVG formats

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `shap`
- `openpyxl`
- `seaborn`
- `scipy`

## Usage

Run each script from the command line after updating the input and output paths, or after converting the scripts to command-line arguments.

Example:

```bash
python code/model_performance.py
python code/shap_interpretation.py
```

For the CAMP-R4 probability comparison workflow:

```bash
python code/camp_probability_comparison.py \
  --ann-match path/to/ann_rule_matching.txt \
  --svm-match path/to/svm_rule_matching.txt \
  --rf-match path/to/rf_rule_matching.txt \
  --ann-mismatch path/to/ann_rule_mismatching.txt \
  --svm-mismatch path/to/svm_rule_mismatching.txt \
  --rf-mismatch path/to/rf_rule_mismatching.txt \
  --output-dir results
```

## Methodological Notes

- Model performance was estimated using stratified 5-fold cross-validation.
- The evaluated models include Random Forest, Logistic Regression, and XGBoost.
- SHAP analysis was applied to the Random Forest model to interpret feature contributions.
- CAMP prediction outputs were used for candidate probability comparison across model types and rule groups.
- The analyses are computational and do not constitute experimental validation.

## Citation

If you use or adapt this repository, please cite the associated manuscript once available.

