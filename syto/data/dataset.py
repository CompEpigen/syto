""""""

import os
import csv
import random
from typing import Dict, Sequence, Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import transformers
import numpy as np
import pandas as pd
from syto.data.sequencing.genome import collapse_methylation

COLUMN_ALIASES = {
    "input_ids": ["input_ids", "genome_sequence", "seq", "dna"],
    "methylation_ids": ["methylation_ids", "cpg_methylation_sequence", "pattern"],
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


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning or predicting"""

    def __init__(
        self,
        data_path_or_list: Union[str, list],
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
        first_n_samples: int = None,
        data_interface: str = "csv",
        lazy_tokenization=False,
        include_dmr_ids=False,
        dmr_label_column=None,
        soft_labels=False,
    ):
        """
        Args:
            data_path_or_list (str or list): Path to the CSV file or a structured list.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding inputs.
            kmer (int, optional): K-mer size. Defaults to -1.
        """

        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.inversed_vocab = {y: x for (x, y) in tokenizer.vocab.items()}
        self.cpg_methylation = None
        self.m6a_methylation = None
        self.lazy_tokenization = lazy_tokenization
        self.include_dmr_ids = include_dmr_ids
        self.soft_labels = soft_labels
        self.dmr_label_column = dmr_label_column

        # Determine input type
        if data_interface == "csv":
            if isinstance(data_path_or_list, str):
                # Load data from CSV file
                with open(data_path_or_list + ".csv", "r") as f:
                    data = list(csv.reader(f))
            elif isinstance(data_path_or_list, list):
                # Use data directly as a structured list
                data = data_path_or_list
            else:
                raise ValueError(
                    "data_path_or_list must be a string (CSV path) or a list (structured like a CSV)."
                )
            # Extract header and data
            header = data[0]
            if first_n_samples is not None:
                data = data[1:first_n_samples]
            else:
                data = data[1:]

            # Identify indices dynamically
            indices = {col: idx for idx, col in enumerate(header)}
            genome_index = (
                indices.get("input_ids")
                if "input_ids" in indices
                else indices.get("genome_sequence")
            )
            cpg_index = (
                indices.get("methylation_ids")
                if "methylation_ids" in indices
                else indices.get("cpg_methylation_sequence")
            )
            m6a_index = indices.get("m6a_methylation_sequence")
            labels_index = indices.get("label")

            if genome_index is None:
                raise ValueError("Genome sequence is absent from the input data")

            # Extract genome sequences
            texts = [row[genome_index] for row in data]

            # Extract labels (optional)
            if soft_labels:
                raise NotImplementedError(
                    "Method to encode soft labels with .csv interface is not implemented"
                )
            else:
                self.labels = (
                    [int(row[labels_index]) for row in data]
                    if labels_index is not None
                    else None
                )

            # Extract methylation data (optional)
            self.cpg_methylation = (
                [row[cpg_index] for row in data] if cpg_index is not None else None
            )
            self.m6a_methylation = (
                [row[m6a_index] for row in data] if m6a_index is not None else None
            )
        elif data_interface == "pandas":
            if isinstance(data_path_or_list, pd.DataFrame):
                data = data_path_or_list

            elif os.path.exists(data_path_or_list + ".parquet"):
                data = pd.read_parquet(data_path_or_list + ".parquet")
            else:
                data = pd.read_csv(data_path_or_list + ".csv")

            cols = data.columns
            dna_col = resolve_column(cols, "input_ids")
            meth_col = resolve_column(cols, "methylation_ids")
            label_col = resolve_column(cols, "soft_label" if soft_labels else "label")
            if dna_col is None:
                raise ValueError(
                    f"No recognized DNA sequence column found. Expected one of: {COLUMN_ALIASES['input_ids']}"
                )
            if meth_col is None:
                raise ValueError(
                    f"No recognized Methylation sequence column found. Expected one of: {COLUMN_ALIASES['methylation_ids']}"
                )
            dna, methylation, labels = (
                data[dna_col],
                data[meth_col],
                data[label_col],
            )
            if self.include_dmr_ids:
                if self.dmr_label_column is None:
                    raise ValueError(
                        "dmr_label_column must not be none if include_dmr_ids is set to True"
                    )
                self.dmr_ids = data[self.dmr_label_column]
            self.labels = labels.to_list()
            self.cpg_methylation = methylation.to_list()
            texts = dna.to_list()
        if not lazy_tokenization:
            # Tokenize genome sequences
            output = tokenizer(
                texts,  # pylint: disable=possibily-used-before-assignment
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            self.input_ids = output["input_ids"]
            self.attention_mask = output["attention_mask"]

            # Tokenize methylation sequences if present
            if self.cpg_methylation is not None:
                self.cpg_methylation = torch.tensor(
                    [
                        self.tokenize_methyl_sequences(input_ids, methyl_seq)
                        for input_ids, methyl_seq in zip(
                            self.input_ids, self.cpg_methylation
                        )
                    ]
                )

            if self.m6a_methylation is not None:
                self.m6a_methylation = torch.tensor(
                    [
                        self.tokenize_methyl_sequences(input_ids, methyl_seq)
                        for input_ids, methyl_seq in zip(
                            self.input_ids, self.m6a_methylation
                        )
                    ]
                )
        else:
            self.texts = texts

    def tokenize_methyl_sequences(self, input_ids, methyl_seq):
        methyl_seq = [int(x) for x in methyl_seq]
        token_lengths = [len(self.inversed_vocab[x]) for x in input_ids.tolist()]
        token_breaks = np.cumsum(token_lengths)
        token_starts = np.insert(token_breaks[:-1], 0, 0)
        return [
            collapse_methylation(methyl_seq[start:end])
            for start, end in zip(token_starts, token_breaks)
        ]

    def __len__(self):
        if self.lazy_tokenization:
            return len(self.texts)
        else:
            return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.lazy_tokenization:
            encoded = self.tokenizer(
                text=self.texts[i],
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            item = {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            }
            if self.cpg_methylation is not None:
                item["cpg_methylation"] = torch.tensor(
                    self.tokenize_methyl_sequences(
                        encoded["input_ids"].squeeze(0), self.cpg_methylation[i]
                    )
                )
            if self.m6a_methylation is not None:
                item["m6a_methylation"] = torch.tensor(
                    self.tokenize_methyl_sequences(
                        encoded["input_ids"].squeeze(0), self.m6a_methylation[i]
                    )
                )
        else:
            item = {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
            }
            if self.cpg_methylation is not None:
                item["cpg_methylation"] = self.cpg_methylation[i]
            if self.m6a_methylation is not None:
                item["m6a_methylation"] = self.m6a_methylation[i]
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])
        if self.include_dmr_ids:
            item["dmr_ids"] = torch.tensor(self.dmr_ids[i])
        return item


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    soft_labels: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = instances[0].keys()
        batch = {key: [instance[key] for instance in instances] for key in keys}

        # Pad input_ids and attention_mask
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        # Handle optional fields
        if "cpg_methylation" in batch:
            batch["cpg_methylation"] = torch.nn.utils.rnn.pad_sequence(
                batch["cpg_methylation"], batch_first=True, padding_value=2
            )
        if "m6a_methylation" in batch:
            batch["m6a_methylation"] = torch.nn.utils.rnn.pad_sequence(
                batch["m6a_methylation"], batch_first=True, padding_value=2
            )
        if "labels" in batch:
            if self.soft_labels:
                batch["labels"] = torch.stack(batch["labels"]).float()
            else:
                batch["labels"] = torch.stack(batch["labels"]).long()
        if "dmr_ids" in batch:
            batch["dmr_ids"] = torch.tensor(batch["dmr_ids"], dtype=torch.long)

        return batch


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
