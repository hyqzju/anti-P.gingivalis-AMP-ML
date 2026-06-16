# SHAP-AMP Sequence Generator

`SHAP-AMP Sequence Generator` is a reproducible Python utility for generating antimicrobial peptide candidate sequences under descriptor ranges derived from SHAP-informed model interpretation.

The script generates two datasets:

- Peptides matching the SHAP-informed descriptor rule set.
- Peptides intentionally mismatching the same rule set, suitable as a comparison group.

Both datasets are exported as FASTA files, accompanied by descriptor-summary CSV files and a ZIP archive.

## Features

- Generates unique peptide sequences for each group.
- Uses RDKit `Chem.MolFromFASTA` and `Crippen.MolLogP` for cLogP calculation.
- Applies descriptor filters for cLogP, net charge, sequence length, cationic composition, hydrophobic composition, and neutral-residue presence.
- Provides deterministic output through a configurable random seed.
- Writes publication-friendly FASTA and CSV outputs.

## Installation

RDKit is required. The recommended installation method is Conda:

```bash
conda create -n shap-amp-generator python=3.11 -c conda-forge rdkit
conda activate shap-amp-generator
```

No additional third-party Python packages are required.

## Usage

Generate 10,000 matching and 10,000 mismatching peptide candidates:

```bash
python generate_shap_amp_sequences.py --n-sequences 10000 --output-dir generated_sequences --seed 42
```

The output directory will contain:

- `rdkit_shap_rule_matching.fasta`
- `rdkit_shap_rule_mismatching.fasta`
- `rdkit_shap_rule_matching_summary.csv`
- `rdkit_shap_rule_mismatching_summary.csv`
- `rdkit_shap_rule_sequences.zip`

## SHAP-Informed Rule Set

| Descriptor | Accepted range |
| --- | --- |
| cLogP | -5.768475 to -1.4346 |
| Net charge | 5.75 to 10.00 |
| Length | 13 to 28 amino acids |
| Cationic composition | 0.2902 to 0.367075 |
| Hydrophobic composition | 0.3824 to 0.5833 |
| Neutral residue presence | Required |

Net charge is estimated as:

```text
K + R + 0.1 * H - D - E
```

## Reproducibility

The default random seed is `42`. Use the `--seed` argument to reproduce or intentionally vary the generated datasets.

## Citation

If this code is used in a manuscript or data supplement, cite the associated article or repository.

## License

Add the license required by your journal, institution, or repository before public release.
