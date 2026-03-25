import os
import csv
import random
from typing import Dict, Sequence, Union, Tuple, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import transformers
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm
from methyldl.data.sequencing.genome import collapse_methylation

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
                texts,
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


def select_optimal_subsequence(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Select optimal subsequence from DNA read based on methylation patterns.

    Parameters:
    -----------
    input_ids : str
        DNA sequence string
    methylation_ids : str
        Methylation status string (0=unmethylated, 1=methylated, 2=irrelevant/unknown)
    target_read_length : int
        Desired length of the subsequence
    selection_criteria : str
        Either "counts" (maximize labeled CpGs) or "entropy" (minimize entropy)
    min_labeled_cpgs : int
        Minimum number of labeled CpGs required for entropy calculation
    stride:int
        Stride of a sliding window for approximation (bigger stride --> more efficient, less accurated)
    entropy_weight_factor : float
        Factor to balance entropy vs count of labeled CpGs (for entropy mode)

    Returns:
    --------
    Tuple[str, str, int, float]
        (selected_dna_sequence, selected_methylation_sequence, labeled_cpg_count, entropy_score)
    """

    # Validate inputs
    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    # if target_read_length > len(input_ids):
    #     raise ValueError("Target read length cannot exceed sequence length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)
    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))  # (labeled_count, entropy)
    if read_length > target_read_length:
        # Slide window across the sequence
        for start_idx in range(0, read_length - target_read_length + 1, stride):
            end_idx = start_idx + target_read_length

            # Extract subsequence
            subseq_methylation = methylation_ids[start_idx:end_idx]

            # Count labeled methylation sites (excluding '2')
            labeled_positions = [pos for pos in subseq_methylation if pos != "2"]
            labeled_count = len(labeled_positions)

            # Skip if no labeled positions
            if labeled_count == 0:
                continue

            # Calculate entropy for labeled positions
            if labeled_count >= min_labeled_cpgs:
                # Convert to numeric for entropy calculation
                numeric_labels = [int(pos) for pos in labeled_positions]
                unique, counts = np.unique(numeric_labels, return_counts=True)
                probabilities = counts / len(numeric_labels)
                entropy_score = entropy(probabilities, base=2)  # Use base 2 for bits
            else:
                entropy_score = float(
                    "inf"
                )  # Penalize sequences with too few labeled CpGs

            # Determine score based on selection criteria
            if selection_criteria == "counts":
                score = labeled_count
                is_better = score > best_score
            else:  # entropy
                if labeled_count < min_labeled_cpgs:
                    # Heavily penalize sequences with insufficient labeled CpGs
                    score = float("inf")
                else:
                    # Lower entropy is better, but also consider count of labeled CpGs
                    # Add small penalty for fewer labeled CpGs to break ties
                    score = entropy_score + entropy_weight_factor * (
                        1.0 / labeled_count
                    )
                is_better = score < best_score

            # Update best subsequence if this one is better
            if is_better:
                best_score = score
                best_start_idx = start_idx
                best_stats = (labeled_count, entropy_score)

        # Extract the best subsequence
        best_end_idx = best_start_idx + target_read_length
        selected_dna = input_ids[best_start_idx:best_end_idx]
        selected_methylation = methylation_ids[best_start_idx:best_end_idx]
    else:  # delegates to vectorized version if sequence is shorter than target
        return select_optimal_subsequence_vectorized(
            input_ids,
            methylation_ids,
            target_read_length,
            selection_criteria,
            min_labeled_cpgs,
            stride,
            entropy_weight_factor,
        )

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def select_optimal_subsequence_rolling_window(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Ultra-optimized version using rolling window statistics for maximum efficiency.
    Best for very long sequences with small strides.
    """

    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)

    if read_length <= target_read_length:
        return select_optimal_subsequence_vectorized(
            input_ids,
            methylation_ids,
            target_read_length,
            selection_criteria,
            min_labeled_cpgs,
            stride,
            entropy_weight_factor,
        )

    # Convert to numpy arrays
    meth_array = np.array(list(methylation_ids))

    # Create binary masks for efficient counting
    is_labeled = meth_array != "2"
    is_methylated = meth_array == "1"
    is_unmethylated = meth_array == "0"

    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))

    # Use stride-based sampling for efficiency
    for start_idx in range(0, read_length - target_read_length + 1, stride):
        end_idx = start_idx + target_read_length

        # Fast counting using pre-computed masks
        window_labeled = is_labeled[start_idx:end_idx]
        labeled_count = np.sum(window_labeled)

        if labeled_count == 0:
            continue

        # Calculate entropy efficiently
        if labeled_count >= min_labeled_cpgs:
            window_methylated = is_methylated[start_idx:end_idx]
            window_unmethylated = is_unmethylated[start_idx:end_idx]

            meth_count = np.sum(window_methylated)
            unmeth_count = np.sum(window_unmethylated)

            # Fast entropy calculation for binary case
            if meth_count > 0 and unmeth_count > 0:
                p_meth = meth_count / labeled_count
                p_unmeth = unmeth_count / labeled_count
                entropy_score = -(
                    p_meth * np.log2(p_meth) + p_unmeth * np.log2(p_unmeth)
                )
            else:
                entropy_score = 0.0  # All same class = no entropy
        else:
            entropy_score = float("inf")

        # Score calculation
        if selection_criteria == "counts":
            score = labeled_count
            is_better = score > best_score
        else:
            if labeled_count < min_labeled_cpgs:
                score = float("inf")
            else:
                score = entropy_score + entropy_weight_factor * (1.0 / labeled_count)
            is_better = score < best_score

        if is_better:
            best_score = score
            best_start_idx = start_idx
            best_stats = (labeled_count, entropy_score)

    # Extract best subsequence
    best_end_idx = best_start_idx + target_read_length
    selected_dna = input_ids[best_start_idx:best_end_idx]
    selected_methylation = methylation_ids[best_start_idx:best_end_idx]

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def extract_optimal_subsequence_dataframe_chunked(
    df: pd.DataFrame,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    entropy_weight_factor: float = 0.1,
    stride: int = 50,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    Memory-efficient version that processes data in chunks.
    Recommended for very large datasets.
    """

    results = []

    for chunk_start in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end]

        chunk_results = []
        for idx, row in chunk_df.iterrows():
            try:
                selected_dna, selected_meth, labeled_count, entropy_score = (
                    select_optimal_subsequence_rolling_window(
                        row["input_ids"],
                        row["methylation_ids"],
                        target_read_length,
                        selection_criteria,
                        min_labeled_cpgs,
                        stride,
                        entropy_weight_factor,
                    )
                )

                result = {
                    "selected_input_ids": selected_dna,
                    "selected_methylation_ids": selected_meth,
                    "labeled_cpg_count": labeled_count,
                    "entropy_score": entropy_score,
                    "selection_criteria": selection_criteria,
                }

                for col in chunk_df.columns:
                    if col not in ["input_ids", "methylation_ids"]:
                        result[col] = row[col]

                chunk_results.append(result)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        results.extend(chunk_results)

        # Optional: Clear memory periodically
        if chunk_start % (chunk_size * 10) == 0:
            import gc

            gc.collect()

    return pd.DataFrame(results)


def select_optimal_subsequence_vectorized(
    input_ids: str,
    methylation_ids: str,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    stride: int = 50,
    entropy_weight_factor: float = 0.1,
) -> Tuple[str, str, int, float]:
    """
    Highly optimized version using numpy vectorization and sliding window optimizations.
    """

    # Validate inputs
    if len(input_ids) != len(methylation_ids):
        raise ValueError("DNA sequence and methylation sequence must have equal length")

    if selection_criteria not in ["counts", "entropy"]:
        raise ValueError("Selection criteria must be either 'counts' or 'entropy'")

    read_length = len(input_ids)

    # Handle case where sequence is shorter than target
    if read_length <= target_read_length:
        labeled_positions = [pos for pos in methylation_ids if pos != "2"]
        labeled_count = len(labeled_positions)

        if labeled_count >= min_labeled_cpgs:
            numeric_labels = np.array([int(pos) for pos in labeled_positions])
            _, counts = np.unique(numeric_labels, return_counts=True)
            probabilities = counts / len(numeric_labels)
            entropy_score = entropy(probabilities, base=2)
        else:
            entropy_score = float("inf")

        return input_ids, methylation_ids, labeled_count, entropy_score

    # Convert methylation string to numpy array for faster operations
    meth_array = np.array(list(methylation_ids), dtype="U1")

    # Pre-compute all window positions
    window_starts = np.arange(0, read_length - target_read_length + 1, stride)

    if len(window_starts) == 0:
        # Fallback if stride is too large
        window_starts = np.array([0])

    best_score = float("-inf") if selection_criteria == "counts" else float("inf")
    best_start_idx = 0
    best_stats = (0, float("inf"))

    # Vectorized window processing
    for start_idx in window_starts:
        end_idx = start_idx + target_read_length

        # Extract window using numpy slicing
        window_meth = meth_array[start_idx:end_idx]

        # Count labeled positions using vectorized operations
        labeled_mask = window_meth != "2"
        labeled_count = np.sum(labeled_mask)

        if labeled_count == 0:
            continue

        # Calculate entropy using vectorized operations
        if labeled_count >= min_labeled_cpgs:
            labeled_values = window_meth[labeled_mask].astype(int)
            unique, counts = np.unique(labeled_values, return_counts=True)
            probabilities = counts / labeled_count
            entropy_score = entropy(probabilities, base=2)
        else:
            entropy_score = float("inf")

        # Determine score based on selection criteria
        if selection_criteria == "counts":
            score = labeled_count
            is_better = score > best_score
        else:  # entropy
            if labeled_count < min_labeled_cpgs:
                score = float("inf")
            else:
                score = entropy_score + entropy_weight_factor * (1.0 / labeled_count)
            is_better = score < best_score

        # Update best subsequence if this one is better
        if is_better:
            best_score = score
            best_start_idx = start_idx
            best_stats = (labeled_count, entropy_score)

    # Extract the best subsequence
    best_end_idx = best_start_idx + target_read_length
    selected_dna = input_ids[best_start_idx:best_end_idx]
    selected_methylation = methylation_ids[best_start_idx:best_end_idx]

    return selected_dna, selected_methylation, best_stats[0], best_stats[1]


def extract_optimal_subsequence_dataframe(
    df: pd.DataFrame,
    target_read_length: int,
    selection_criteria: str = "counts",
    min_labeled_cpgs: int = 5,
    entropy_weight_factor: float = 0.1,
    stride: int = 50,
) -> pd.DataFrame:
    """
    Process entire dataframe to select optimal subsequences for all reads.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'input_ids' and 'methylation_ids' columns
    target_read_length : int
        Desired length of subsequences
    selection_criteria : str
        Either "counts" or "entropy"
    min_labeled_cpgs : int
        Minimum labeled CpGs for entropy calculation
    entropy_weight_factor : float
        Weight factor for entropy calculation

    Returns:
    --------
    pd.DataFrame
        DataFrame with original columns plus selected subsequences and statistics
    """

    results = []

    for idx, row in tqdm(df.iterrows()):
        try:
            selected_dna, selected_meth, labeled_count, entropy_score = (
                select_optimal_subsequence(
                    row["input_ids"],
                    row["methylation_ids"],
                    target_read_length,
                    selection_criteria,
                    min_labeled_cpgs,
                    stride,
                    entropy_weight_factor,
                )
            )

            result = {
                "original_input_ids": row["input_ids"],
                "original_methylation_ids": row["methylation_ids"],
                "selected_input_ids": selected_dna,
                "selected_methylation_ids": selected_meth,
                "labeled_cpg_count": labeled_count,
                "entropy_score": entropy_score,
                "selection_criteria": selection_criteria,
            }

            # Add other columns from original dataframe
            for col in df.columns:
                if col not in ["input_ids", "methylation_ids"]:
                    result[col] = row[col]

            results.append(result)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    return pd.DataFrame(results)
