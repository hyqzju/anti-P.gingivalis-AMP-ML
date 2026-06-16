"""Generate antimicrobial peptide candidates using SHAP-informed rules.

This script creates two non-overlapping FASTA datasets:

1. Candidate peptides that satisfy a predefined SHAP-informed descriptor range.
2. Candidate peptides that intentionally fall outside the same rule set.

The cLogP descriptor is calculated with RDKit using Crippen.MolLogP on
Chem.MolFromFASTA, matching the molecular representation used by RDKit.
"""

from __future__ import annotations

import argparse
import csv
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen
except ImportError as exc:
    raise ImportError(
        "RDKit is required. Install it with: "
        "conda install -c conda-forge rdkit"
    ) from exc


DEFAULT_RULE_RANGES = {
    "cLogP": (-5.768475, -1.4346),
    "Net_charge": (5.75, 10.00),
    "Length": (13, 28),
    "Cationic_comp": (0.2902, 0.367075),
    "Hydrophobic_comp": (0.3824, 0.5833),
}

CATIONIC_POOL = "KR"
HYDROPHOBIC_POOL = "AVLIFYW"
NEUTRAL_POOL = "GSTNQP"
ACIDIC_POOL = "DE"

CATIONIC_SET = set("KRH")
HYDROPHOBIC_SET = set("AVILMFWYC")
NEUTRAL_SET = set("GSTNQP")


@dataclass(frozen=True)
class PeptideFeatures:
    """Descriptor values used for rule matching."""

    clogp: float | None
    net_charge: float
    length: int
    cationic_comp: float
    hydrophobic_comp: float
    has_neutral: bool


def calculate_rdkit_clogp(sequence: str) -> float | None:
    """Calculate peptide cLogP with RDKit Crippen MolLogP."""
    molecule = Chem.MolFromFASTA(sequence)
    if molecule is None:
        return None
    return float(Crippen.MolLogP(molecule))


def calculate_features(sequence: str) -> PeptideFeatures:
    """Calculate rule descriptors for a peptide sequence."""
    sequence = sequence.upper()
    length = len(sequence)

    net_charge = (
        sequence.count("K")
        + sequence.count("R")
        + 0.1 * sequence.count("H")
        - sequence.count("D")
        - sequence.count("E")
    )

    return PeptideFeatures(
        clogp=calculate_rdkit_clogp(sequence),
        net_charge=net_charge,
        length=length,
        cationic_comp=sum(aa in CATIONIC_SET for aa in sequence) / length,
        hydrophobic_comp=sum(aa in HYDROPHOBIC_SET for aa in sequence) / length,
        has_neutral=any(aa in NEUTRAL_SET for aa in sequence),
    )


def sequence_matches_rule(
    sequence: str,
    rule_ranges: dict[str, tuple[float, float]] = DEFAULT_RULE_RANGES,
) -> bool:
    """Return True when a sequence satisfies all SHAP-informed rules."""
    features = calculate_features(sequence)
    if features.clogp is None:
        return False

    return (
        rule_ranges["cLogP"][0] <= features.clogp <= rule_ranges["cLogP"][1]
        and rule_ranges["Net_charge"][0]
        <= features.net_charge
        <= rule_ranges["Net_charge"][1]
        and rule_ranges["Length"][0] <= features.length <= rule_ranges["Length"][1]
        and rule_ranges["Cationic_comp"][0]
        <= features.cationic_comp
        <= rule_ranges["Cationic_comp"][1]
        and rule_ranges["Hydrophobic_comp"][0]
        <= features.hydrophobic_comp
        <= rule_ranges["Hydrophobic_comp"][1]
        and features.has_neutral
    )


def possible_counts_for_matching(
    length: int,
    rule_ranges: dict[str, tuple[float, float]] = DEFAULT_RULE_RANGES,
) -> list[tuple[int, int, int]]:
    """Return feasible residue-count combinations for matching peptides."""
    cationic_min = int(rule_ranges["Cationic_comp"][0] * length + 0.999999)
    cationic_max = int(rule_ranges["Cationic_comp"][1] * length)

    hydrophobic_min = int(rule_ranges["Hydrophobic_comp"][0] * length + 0.999999)
    hydrophobic_max = int(rule_ranges["Hydrophobic_comp"][1] * length)

    combinations = []
    for cationic_count in range(cationic_min, cationic_max + 1):
        if not (
            rule_ranges["Net_charge"][0]
            <= cationic_count
            <= rule_ranges["Net_charge"][1]
        ):
            continue

        for hydrophobic_count in range(hydrophobic_min, hydrophobic_max + 1):
            neutral_count = length - cationic_count - hydrophobic_count
            if neutral_count >= 1:
                combinations.append(
                    (cationic_count, hydrophobic_count, neutral_count)
                )

    return combinations


def build_matching_count_map(
    rule_ranges: dict[str, tuple[float, float]] = DEFAULT_RULE_RANGES,
) -> dict[int, list[tuple[int, int, int]]]:
    """Map peptide lengths to feasible matching residue-count combinations."""
    min_length, max_length = rule_ranges["Length"]
    return {
        length: possible_counts_for_matching(length, rule_ranges)
        for length in range(int(min_length), int(max_length) + 1)
    }


def generate_matching_candidate(
    rng: random.Random,
    count_map: dict[int, list[tuple[int, int, int]]],
) -> str:
    """Generate one candidate designed to satisfy the rule ranges."""
    valid_lengths = [length for length, combinations in count_map.items() if combinations]
    length = rng.choice(valid_lengths)
    cationic_count, hydrophobic_count, neutral_count = rng.choice(count_map[length])

    residues = (
        rng.choices(CATIONIC_POOL, k=cationic_count)
        + rng.choices(HYDROPHOBIC_POOL, k=hydrophobic_count)
        + rng.choices(NEUTRAL_POOL, k=neutral_count)
    )

    rng.shuffle(residues)
    return "".join(residues)


def generate_mismatching_candidate(rng: random.Random) -> str:
    """Generate one candidate designed to violate the rule ranges."""
    length = rng.randint(13, 28)

    cationic_count = rng.choice([1, 2, 3, 4])
    hydrophobic_count = rng.randint(
        max(2, int(0.20 * length)),
        max(3, int(0.55 * length)),
    )
    acidic_count = rng.choice([0, 1, 2])

    neutral_count = length - cationic_count - hydrophobic_count - acidic_count
    if neutral_count < 1:
        neutral_count = 1
        hydrophobic_count = length - cationic_count - acidic_count - neutral_count

        if hydrophobic_count < 1:
            hydrophobic_count = 1
            acidic_count = 0
            neutral_count = length - cationic_count - hydrophobic_count

    residues = (
        rng.choices(CATIONIC_POOL, k=cationic_count)
        + rng.choices(HYDROPHOBIC_POOL, k=hydrophobic_count)
        + rng.choices(ACIDIC_POOL, k=acidic_count)
        + rng.choices(NEUTRAL_POOL, k=neutral_count)
    )

    rng.shuffle(residues)
    return "".join(residues)


def generate_unique_sequences(
    target_n: int,
    generator_func,
    check_func,
    max_attempts: int,
    progress_interval: int,
) -> tuple[list[str], int]:
    """Generate unique sequences that pass a validation function."""
    sequences: list[str] = []
    seen: set[str] = set()
    attempts = 0

    while len(sequences) < target_n and attempts < max_attempts:
        attempts += 1
        sequence = generator_func()

        if sequence in seen:
            continue

        if check_func(sequence):
            seen.add(sequence)
            sequences.append(sequence)

        if progress_interval and len(sequences) % progress_interval == 0 and sequences:
            print(f"Generated {len(sequences)} sequences after {attempts} attempts.")

    if len(sequences) < target_n:
        raise RuntimeError(
            f"Only generated {len(sequences)} sequences; "
            f"target was {target_n}. Increase --max-attempts."
        )

    return sequences, attempts


def write_fasta(sequences: list[str], path: Path, prefix: str) -> None:
    """Write sequences in FASTA format."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sequence in enumerate(sequences, 1):
            handle.write(f">{prefix}_{index:05d}\n")
            handle.write(f"{sequence}\n")


def summarize_sequences(
    sequences: list[str],
    path: Path,
    label: str,
    attempts: int,
) -> None:
    """Write a compact CSV summary of generated sequence descriptors."""
    features = [calculate_features(sequence) for sequence in sequences]
    rows = [
        ("Group", label),
        ("Number of sequences", len(sequences)),
        ("Generation attempts", attempts),
        ("cLogP calculation", "RDKit Crippen MolLogP from Chem.MolFromFASTA"),
        ("Length range", f"{min(f.length for f in features)} to {max(f.length for f in features)}"),
        ("Mean length", f"{mean(f.length for f in features):.4f}"),
        ("cLogP range", f"{min(f.clogp for f in features):.6f} to {max(f.clogp for f in features):.6f}"),
        ("Mean cLogP", f"{mean(f.clogp for f in features):.6f}"),
        ("Net charge range", f"{min(f.net_charge for f in features):.4f} to {max(f.net_charge for f in features):.4f}"),
        ("Mean net charge", f"{mean(f.net_charge for f in features):.4f}"),
        ("Cationic composition range", f"{min(f.cationic_comp for f in features):.6f} to {max(f.cationic_comp for f in features):.6f}"),
        ("Mean cationic composition", f"{mean(f.cationic_comp for f in features):.6f}"),
        ("Hydrophobic composition range", f"{min(f.hydrophobic_comp for f in features):.6f} to {max(f.hydrophobic_comp for f in features):.6f}"),
        ("Mean hydrophobic composition", f"{mean(f.hydrophobic_comp for f in features):.6f}"),
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Item", "Value"])
        writer.writerows(rows)


def package_outputs(zip_path: Path, paths: list[Path]) -> None:
    """Compress generated files into a single ZIP archive."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP-informed antimicrobial peptide candidate datasets "
            "and matched rule-violating controls."
        )
    )
    parser.add_argument(
        "-n",
        "--n-sequences",
        type=int,
        default=10_000,
        help="Number of unique sequences to generate per group.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("generated_sequences"),
        help="Directory where FASTA, CSV, and ZIP outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sequence generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10_000_000,
        help="Maximum generation attempts per group.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress after this many accepted sequences; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    """Run sequence generation and write output files."""
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matching_fasta = args.output_dir / "rdkit_shap_rule_matching.fasta"
    mismatching_fasta = args.output_dir / "rdkit_shap_rule_mismatching.fasta"
    matching_summary = args.output_dir / "rdkit_shap_rule_matching_summary.csv"
    mismatching_summary = args.output_dir / "rdkit_shap_rule_mismatching_summary.csv"
    zip_path = args.output_dir / "rdkit_shap_rule_sequences.zip"

    count_map = build_matching_count_map()

    print("Generating SHAP-rule-matching peptide candidates...")
    matching_sequences, matching_attempts = generate_unique_sequences(
        target_n=args.n_sequences,
        generator_func=lambda: generate_matching_candidate(rng, count_map),
        check_func=sequence_matches_rule,
        max_attempts=args.max_attempts,
        progress_interval=args.progress_interval,
    )

    print("Generating SHAP-rule-mismatching peptide candidates...")
    mismatching_sequences, mismatching_attempts = generate_unique_sequences(
        target_n=args.n_sequences,
        generator_func=lambda: generate_mismatching_candidate(rng),
        check_func=lambda sequence: not sequence_matches_rule(sequence),
        max_attempts=args.max_attempts,
        progress_interval=args.progress_interval,
    )

    write_fasta(matching_sequences, matching_fasta, "RDKit_SHAPRuleMatch")
    write_fasta(mismatching_sequences, mismatching_fasta, "RDKit_SHAPRuleMismatch")
    summarize_sequences(
        matching_sequences,
        matching_summary,
        "RDKit-SHAP-rule-matching",
        matching_attempts,
    )
    summarize_sequences(
        mismatching_sequences,
        mismatching_summary,
        "RDKit-SHAP-rule-mismatching",
        mismatching_attempts,
    )
    package_outputs(
        zip_path,
        [matching_fasta, mismatching_fasta, matching_summary, mismatching_summary],
    )

    print("Generation completed.")
    print(f"Matching FASTA: {matching_fasta}")
    print(f"Mismatching FASTA: {mismatching_fasta}")
    print(f"ZIP archive: {zip_path}")
    print("First five matching sequences:")
    for index, sequence in enumerate(matching_sequences[:5], 1):
        print(f">RDKit_SHAPRuleMatch_{index:05d}")
        print(sequence)

    print("First five mismatching sequences:")
    for index, sequence in enumerate(mismatching_sequences[:5], 1):
        print(f">RDKit_SHAPRuleMismatch_{index:05d}")
        print(sequence)


if __name__ == "__main__":
    main()
