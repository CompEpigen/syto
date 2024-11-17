from torch.utils.data import Dataset
import torch
import transformers
import csv
import numpy as np
from typing import Dict, Sequence, Union
from dataclasses import dataclass

from methyldl.data.genome import collapse_methylation


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning or predicting"""

    def __init__(self, 
                 data_path_or_list: Union[str,list], 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):
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

        # Determine input type
        if isinstance(data_path_or_list, str):
            # Load data from CSV file
            with open(data_path_or_list, "r") as f:
                data = list(csv.reader(f))
        elif isinstance(data_path_or_list, list):
            # Use data directly as a structured list
            data = data_path_or_list
        else:
            raise ValueError("data_path_or_list must be a string (CSV path) or a list (structured like a CSV).")

        # Extract header and data
        header = data[0]
        data = data[1:]

        # Identify indices dynamically
        indices = {col: idx for idx, col in enumerate(header)}
        genome_index = indices.get('input_ids') if 'input_ids' in indices else indices.get('genome_sequence')
        cpg_index = indices.get('methylation_ids') if 'methylation_ids' in indices else indices.get('cpg_methylation_sequence')
        m6a_index = indices.get('m6a_methylation_sequence')
        labels_index = indices.get('label')

        if genome_index is None:
            raise ValueError("Genome sequence is absent from the input data")

        # Extract genome sequences
        texts = [row[genome_index] for row in data]

        # Extract labels (optional)
        self.labels = [int(row[labels_index]) for row in data] if labels_index is not None else None

        # Extract methylation data (optional)
        self.cpg_methylation = (
            [row[cpg_index] for row in data] if cpg_index is not None else None
        )
        self.m6a_methylation = (
            [row[m6a_index] for row in data] if m6a_index is not None else None
        )

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
            self.cpg_methylation = torch.tensor([
                self.tokenize_methyl_sequences(text, input_ids, methyl_seq)
                for text, input_ids, methyl_seq in zip(texts, self.input_ids, self.cpg_methylation)
            ])

        if self.m6a_methylation is not None:
            self.m6a_methylation = torch.tensor([
                self.tokenize_methyl_sequences(text, input_ids, methyl_seq)
                for text, input_ids, methyl_seq in zip(texts, self.input_ids, self.m6a_methylation)
            ])

    def tokenize_methyl_sequences(self, text, input_ids, methyl_seq):
        methyl_seq = [int(x) for x in methyl_seq]
        token_lengths = [len(self.inversed_vocab[x]) for x in input_ids.tolist()]
        token_breaks = np.cumsum(token_lengths)
        token_starts = np.insert(token_breaks[:-1], 0, 0)
        return [
            collapse_methylation(methyl_seq[start:end])
            for start, end in zip(token_starts, token_breaks)
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i])

        if self.cpg_methylation is not None:
            item["cpg_methylation"] = self.cpg_methylation[i]

        if self.m6a_methylation is not None:
            item["m6a_methylation"] = self.m6a_methylation[i]

        return item


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        keys = instances[0].keys()
        batch = {key: [instance[key] for instance in instances] for key in keys}

        # Pad input_ids and attention_mask
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id
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
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)

        return batch


import random
from typing import List, Dict, Union

def generate_example_data(sequence_length: int = 150, 
                          include_cpg_methylation: bool = False, 
                          include_m6a_methylation: bool = False, 
                          include_labels: bool = False, 
                          num_samples: int = 1) -> List[List[Union[str, int]]]:
    """
    Generate example data for testing a model.
    
    Args:
        sequence_length (int): Length of the genome sequence.
        include_cpg_methylation (bool): Whether to include cpg_methylation feature.
        include_m6a_methylation (bool): Whether to include m6a_methylation feature.
        include_labels (bool): Whether to include labels in the data.
        num_samples (int): Number of samples to generate.
        
    Returns:
        List[List[Union[str, int]]]: A list of rows where the first row is the header,
                                     and subsequent rows are synthetic data.
    """
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
            cpg_methylation = "2" * sequence_length
            row.append(cpg_methylation)
        
        # Add m6a_methylation if required
        if include_m6a_methylation:
            m6a_methylation = "2" * sequence_length
            row.append(m6a_methylation)
        
        # Add labels if required
        if include_labels:
            row.append(0)  # All labels are set to 0
        
        data.append(row)
    
    return data
