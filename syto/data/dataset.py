""""""

import os
import random
from typing import Dict, Union, List

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "input_ids": ["input_ids", "genome_sequence", "seq", "dna", "original_seq"],
    "methylation_ids": [
        "methylation_ids",
        "cpg_methylation_sequence",
        "pattern",
        "methyl_seq",
        "methylation_encoding",
    ],
    "label": ["label"],
    "soft_label": ["soft_label"],
    "m6a_methylation_sequence": ["m6a_methylation_sequence"],
    "dmr_label": ["dmr_label"],
}


def resolve_column(columns, canonical_name):
    """Find the first matching alias for a canonical column name."""
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    for alias in aliases:
        if alias in columns:
            return alias
    return None


def generate_example_data(
    sequence_length: int = 150,
    include_cpg_methylation: bool = False,
    include_m6a_methylation: bool = False,
    include_labels: bool = False,
    num_samples: int = 1,
    cpg_proportion: Dict[int, float] = None,
    m6a_proportion: Dict[int, float] = None,
) -> List[List[Union[str, int]]]:
    """
    Generate example data for testing a model.

    Args:
        sequence_length (int): Length of the genome sequence.
        include_cpg_methylation (bool): Whether to include cpg_methylation feature.
        include_m6a_methylation (bool): Whether to include m6a_methylation feature.
        include_labels (bool): Whether to include labels in the data.
        num_samples (int): Number of samples to generate.
        cpg_proportion (Dict[int, float]): Proportions for CpG methylation states (e.g., {2: 0.5, 1: 0.3, 0: 0.2}). Defaults to equal proportions.
        m6a_proportion (Dict[int, float]): Proportions for m6A methylation states (e.g., {2: 0.7, 1: 0.2, 0: 0.1}). Defaults to equal proportions.

    Returns:
        List[List[Union[str, int]]]: A list of rows where the first row is the header,
                                     and subsequent rows are synthetic data.
    """
    if cpg_proportion is None:
        cpg_proportion = {2: 1, 1: 1, 0: 1}
    if m6a_proportion is None:
        m6a_proportion = {2: 1, 1: 1, 0: 1}

    # Validate inputs
    if num_samples < 1:
        raise ValueError("The number of samples (num_samples) must be at least 1.")
    if sequence_length < 30:
        raise ValueError("The sequence length (sequence_length) must be at least 30.")
    if include_cpg_methylation and not cpg_proportion:
        raise ValueError(
            "Proportions for CpG methylation must be specified if it is included."
        )
    if include_m6a_methylation and not m6a_proportion:
        raise ValueError(
            "Proportions for m6A methylation must be specified if it is included."
        )

    # Helper function to generate methylation sequence
    def generate_methylation_sequence(
        sequence: str, target: str, proportions: Dict[int, float]
    ) -> str:
        states, weights = zip(*proportions.items())
        methylation = []
        for nucleotide in sequence:
            if nucleotide == target:
                methylation.append(str(random.choices(states, weights=weights, k=1)[0]))
            else:
                methylation.append("2")
        return "".join(methylation)

    # Create the header
    header = ["input_ids"]
    if include_cpg_methylation:
        header.append("cpg_methylation_sequence")
    if include_m6a_methylation:
        header.append("m6a_methylation_sequence")
    if include_labels:
        header.append("label")

    # Generate random sequences and features
    data = [header]
    for _ in range(num_samples):
        row = []
        # Generate a random sequence of nucleotides (A, T, C, G)
        genome_sequence = "".join(random.choices("ATCG", k=sequence_length))
        row.append(genome_sequence)

        # Add cpg_methylation if required
        if include_cpg_methylation:
            cpg_methylation = generate_methylation_sequence(
                genome_sequence, "C", cpg_proportion
            )
            row.append(cpg_methylation)

        # Add m6a_methylation if required
        if include_m6a_methylation:
            m6a_methylation = generate_methylation_sequence(
                genome_sequence, "A", m6a_proportion
            )
            row.append(m6a_methylation)

        # Add labels if required
        if include_labels:
            row.append(
                random.choices((0, 1), weights=(1, 1), k=1)[0]
            )  # Random 1 or 0 label

        data.append(row)

    return data


def generate_example_data_for_methylbert(
    sequence_length: int = 150,
    include_cpg_methylation: bool = True,
    include_m6a_methylation: bool = False,
    include_labels: bool = True,
    num_samples: int = 1,
    cpg_proportion: Dict[int, float] = None,
    m6a_proportion: Dict[int, float] = None,
) -> List[List[Union[str, int]]]:
    """
    Wrapper for generate example data method that does some transformation specific for methylbert

    """
    if cpg_proportion is None:
        cpg_proportion = {2: 1, 1: 1, 0: 1}
    if m6a_proportion is None:
        m6a_proportion = {2: 1, 1: 1, 0: 1}

    if sequence_length > 512:
        samples_coeff = sequence_length // 512 + int(bool(sequence_length % 512))
        sequence_length = int(sequence_length / samples_coeff)
        num_samples = num_samples * samples_coeff

    synthetic_data = generate_example_data(
        sequence_length=sequence_length + 3,
        include_cpg_methylation=include_cpg_methylation,
        include_m6a_methylation=include_m6a_methylation,
        include_labels=include_labels,
        num_samples=num_samples,  # Single sample per repeat
        cpg_proportion=cpg_proportion,
        m6a_proportion=m6a_proportion,
    )

    synthetic_data[0][0] = "dna_seq"
    synthetic_data[0][1] = "methyl_seq"
    synthetic_data[0][2] = "ctype"

    for row in range(1, len(synthetic_data)):
        synthetic_data[row][0] = " ".join(
            [synthetic_data[row][0][i : (i + 3)] for i in range(sequence_length)]
        )
        synthetic_data[row][1] = synthetic_data[row][1][:sequence_length]

    return synthetic_data
