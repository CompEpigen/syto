import os
import csv
import json
import gc
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple, List, Union, Callable
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
import torch
import torch.nn as nn

import transformers
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from syto.classification.hf_training import (
    AuxLossLoggingTrainer,
    BalancedTrainer,
    apply_early_stopping,
)

from syto.classification.evaluation import (
    compute_metrics,
    preprocess_logits_for_prediction,
    extract_trainer_metrics,
)
from syto.classification.utils import calculate_batch_size
from syto.data.dataset import (
    COLUMN_ALIASES,
    resolve_column,
    generate_example_data,
    generate_example_data_for_methylbert,
)
from syto.data.sequencing.genome import collapse_methylation
from safetensors.torch import load_file
from transformers.models.bert.configuration_bert import BertConfig
from syto.classification.classification_heads import (
    GRGAttentionClassificationHead,
)
from syto.classification.loss import ConfidenceWeightedCrossEntropy
from syto.classification.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)
from syto.classification.mlflow_tracking import mlflow_tracked_fit
from syto.classification.training_progress import use_table_progress_callback
from syto.classification.prediction_aggregation import (
    aggregate_chuncked_predictions_weighted,
)
from pathlib import Path


class DNABERT2FineTuneDataset(Dataset):
    """Dataset for supervised fine-tuning or predicting with DNABERT2 (EpigenBERT)."""

    def __init__(
        self,
        data_path_or_list: Union[str, list],
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
        first_n_samples: int = None,
        data_interface: str = "csv",
        lazy_tokenization=False,
        include_grg_ids=False,
        grg_label_column=None,
        soft_labels=False,
    ):
        """
        Args:
            data_path_or_list (str or list): Path to the CSV file or a structured list.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding inputs.
            kmer (int, optional): K-mer size. Defaults to -1.
        """

        super(DNABERT2FineTuneDataset, self).__init__()
        self.tokenizer = tokenizer
        self.inversed_vocab = {y: x for (x, y) in tokenizer.vocab.items()}
        self.cpg_methylation = None
        self.m6a_methylation = None
        self.lazy_tokenization = lazy_tokenization
        self.include_grg_ids = include_grg_ids
        self.soft_labels = soft_labels
        self.grg_label_column = grg_label_column

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
            if self.include_grg_ids:
                if self.grg_label_column is None:
                    raise ValueError(
                        "grg_label_column must not be none if include_grg_ids is set to True"
                    )
                self.grg_ids = data[self.grg_label_column]
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
        if self.include_grg_ids:
            item["grg_ids"] = torch.tensor(self.grg_ids[i])
        return item


# Backward-compatible alias
SupervisedDataset = DNABERT2FineTuneDataset


@dataclass
class DataCollatorForFineTunedDataset:
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
        if "grg_ids" in batch:
            batch["grg_ids"] = torch.tensor(batch["grg_ids"], dtype=torch.long)

        return batch


class BertEmbeddings(nn.Module):
    """Construct the embeddings for words, ignoring position.
    There are no positional embeddings since we use ALiBi and token_type
    embeddings. This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`, but is
    modified as part of Mosaic BERT's ALiBi implementation. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens. This module ignores the `position_ids`
    input to the `forward` method.
    """

    def __init__(self, model, use_cpg_methylation=True, use_m6a_methylation=False):
        super().__init__()
        self.word_embeddings = model.embeddings.word_embeddings
        # ALiBi doesn't use position embeddings
        self.token_type_embeddings = model.embeddings.token_type_embeddings
        self.use_cpg_methylation = use_cpg_methylation
        self.use_m6a_methylation = use_m6a_methylation
        if self.use_cpg_methylation:
            self.cpg_methylation_embeddings = nn.Embedding(3, model.config.hidden_size)
        if self.use_m6a_methylation:
            self.m6a_methylation_embeddings = nn.Embedding(3, model.config.hidden_size)
        # LayerNorm and dropout from the base embeddings
        self.LayerNorm = model.embeddings.LayerNorm
        self.dropout = model.embeddings.dropout
        # self.n_labels = n_labels

        # Register token type IDs for cases where token_type_ids are not passed
        self.register_buffer(
            "token_type_ids",
            torch.zeros(model.config.max_position_embeddings, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cpg_methylation: Optional[torch.LongTensor] = None,
        m6a_methylation: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError("Must specify either input_ids or inputs_embeds!")

        # Determine input shape based on the input provided
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Use ALiBi, so we skip position embeddings
        if position_ids is None:
            pass

        # Handle token_type_ids by using the registered buffer if not provided
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.word_embeddings.device
                )

        # Compute the embeddings for the main input
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # Compute methylation embeddings if cpg_methylation are provided
        if self.use_cpg_methylation and cpg_methylation is not None:
            cpg_methylation_embeddings = self.cpg_methylation_embeddings(
                cpg_methylation
            )
            embeddings += cpg_methylation_embeddings

        if self.use_m6a_methylation and m6a_methylation is not None:
            m6a_methylation_embeddings = self.m6a_methylation_embeddings(
                m6a_methylation
            )
            embeddings += m6a_methylation_embeddings

        # Apply layer normalization and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertModel(BertPreTrainedModel):
    """Overall BERT model.
    Args:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, pretrained_model):
        super(BertModel, self).__init__(pretrained_model.config)
        self.embeddings = pretrained_model.embeddings
        self.encoder = pretrained_model.encoder
        self.pooler = pretrained_model.pooler

        # self.post_init() #TODO Figure out if it is needed.

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cpg_methylation: Optional[torch.LongTensor] = None,
        m6a_methylation: Optional[torch.LongTensor] = None,
        output_all_encoded_layers: Optional[bool] = False,
        masked_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, cpg_methylation, m6a_methylation
        )  # TODO remove position ids as not needed

        subset_mask = []
        first_col_mask = []

        if masked_tokens_mask is None:
            subset_mask = None
        else:
            first_col_mask = torch.zeros_like(masked_tokens_mask)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            subset_mask=subset_mask,
        )

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = (
                self.pooler(sequence_output) if self.pooler is not None else None
            )
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][
                masked_tokens_mask[attention_mask_bool][subset_idx]
            ]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]
                ]
                pooled_output = self.pooler(pool_input, pool=False)
            else:
                pooled_output = None

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output

        if self.pooler is not None:
            return encoder_outputs, pooled_output

        return encoder_outputs, None


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model transformer with a sequence classification/regression head.
    This head is just a linear layer on top of the pooled output. Used for,
    e.g., GLUE tasks.
    """

    def __init__(
        self, prertained_model, num_labels=None, num_grg_labels=None, soft_labels=False
    ):
        super().__init__(prertained_model.config)
        # Overwritting num_labels if those were provided during constructio since the foundational model features classifier with 2 labels
        # Sometimes, one need to overwrite it before fine-tunning for multi-label learning
        # TODO: Think about more elegant way
        if num_labels is not None:
            self.num_labels = num_labels
            self.config.num_labels = num_labels
        else:
            self.num_labels = prertained_model.config.num_labels
        self.config = prertained_model.config

        self.bert = BertModel(prertained_model.bert)  # Reconstructing original model
        self.dropout = prertained_model.dropout
        self.num_grg_labels = num_grg_labels
        self.soft_labels = soft_labels
        if num_grg_labels is None:
            self.classifier = prertained_model.classifier
        else:
            self.config.num_grg_labels = num_grg_labels
            self.config.num_labels = num_labels
            self.classifier = GRGAttentionClassificationHead(self.config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cpg_methylation: Optional[torch.LongTensor] = None,
        m6a_methylation: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        grg_ids: Optional[torch.Tensor] = None,  # DMR labels
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            cpg_methylation=cpg_methylation,
            m6a_methylation=m6a_methylation,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.num_grg_labels is None:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            logits, _ = self.classifier(sequence_output, grg_ids, attention_mask)

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long
                    or labels.dtype == torch.int
                    or self.soft_labels
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.soft_labels:
                    loss_fct = ConfidenceWeightedCrossEntropy(self.num_labels)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs[0],
            attentions=None,
        )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."}
    )
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = (field(default="steps"),)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=8)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    batch_eval_metrics: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    eval_accumulation_steps: int = field(default=8)
    torch_empty_cache_steps: int = field(default=10)
    prediction_loss_only: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    skip_memory_metrics: bool = field(default=True)
    auto_find_batch_size: bool = field(default=False)


def initialize_model_with_custom_embeddings(
    base_model,
    use_cpg,
    use_m6a,
    num_labels=None,
    num_grg_labels=None,
    soft_labels=False,
):
    base_model.bert.embeddings = BertEmbeddings(base_model.bert, use_cpg, use_m6a)
    model = BertForSequenceClassification(
        base_model,
        num_labels=num_labels,
        num_grg_labels=num_grg_labels,
        soft_labels=soft_labels,
    )
    if num_grg_labels is None:
        model.classifier = nn.Linear(768, out_features=num_labels, bias=True)
    return model


class EpigenDnabert2(AbstractReadClassifier):
    def __init__(
        self,
        foundation_model_huggingface: str = "zhihan1996/DNABERT-2-117M",
        load_weights=True,
        fine_tuned_model_path: Optional[str] = None,
        max_sequence_length: int = 150,
        num_labels: int = 2,
        use_cpg_methylation=True,
        use_m6a_methylation=False,
        trust_remote_code=True,
        use_triton=True,
        training_args=None,
        num_grg_labels=None,
        soft_labels=False,
    ):

        assert len(
            foundation_model_huggingface
        ), "Must specify foundation model path hosted on Hugging Face"
        self.num_grg_labels = num_grg_labels
        self.soft_labels = soft_labels

        config = BertForSequenceClassification.config_class.from_pretrained(
            foundation_model_huggingface
        )
        config = BertConfig(**config.to_dict(), use_triton=use_triton)
        # Does not load weights just yet, because if we have a checkpoint, the weights will be retrived from it
        # base_model = transformers.AutoModelForSequenceClassification.from_config(trust_remote_code=trust_remote_code, config = config)
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            foundation_model_huggingface,
            trust_remote_code=trust_remote_code,
            config=config,
            local_files_only=True,
            cache_dir=None,
        )
        model = initialize_model_with_custom_embeddings(
            base_model,
            use_cpg_methylation,
            use_m6a_methylation,
            num_labels=num_labels,
            num_grg_labels=num_grg_labels,
            soft_labels=soft_labels,
        )

        self.num_labels = num_labels
        model.num_labels = num_labels
        if fine_tuned_model_path is not None:
            checkpoint = load_file(fine_tuned_model_path)
            # TODO: Remove this part after proper checkpoint is generated:
            old_cpg_methylation_key = "bert.embeddings.methylation_embeddings.weight"
            if old_cpg_methylation_key in checkpoint.keys():
                checkpoint["bert.embeddings.cpg_methylation_embeddings.weight"] = (
                    checkpoint.pop(old_cpg_methylation_key)
                )

            model.load_state_dict(checkpoint)

            model.eval()
        elif load_weights:
            base_model = (
                transformers.AutoModelForSequenceClassification.from_pretrained(
                    foundation_model_huggingface,
                    trust_remote_code=trust_remote_code,
                    config=config,
                )
            )
            model = initialize_model_with_custom_embeddings(
                base_model,
                use_cpg_methylation,
                use_m6a_methylation,
                num_labels=num_labels,
                num_grg_labels=num_grg_labels,
                soft_labels=soft_labels,
            )
        self.model = model
        self.num_labels = num_labels
        self.config = config
        model_max_length = round(
            max_sequence_length // 4 + 1
        )  # BPE encoding reduces sequence length approximately by a factor of 4

        recomended_batch_size = calculate_batch_size(
            gb_per_seq=0.029, cpu_batch_size=300
        )

        default_training_args = TrainingArguments(
            run_name="dnabert2_default",
            per_device_train_batch_size=recomended_batch_size,
            per_device_eval_batch_size=int(recomended_batch_size / 2),
            gradient_accumulation_steps=1,
            learning_rate=3e-5,
            fp16=True,
            save_steps=10,
            output_dir="output/dnabert2_default",
            eval_strategy="steps",
            eval_steps=10,
            warmup_steps=100,
            logging_steps=100,
            num_train_epochs=250,
            overwrite_output_dir=True,
            log_level="info",
            find_unused_parameters=False,
            batch_eval_metrics=False,
            eval_and_save_results=True,
            remove_unused_columns=False,
            eval_accumulation_steps=8,
            torch_empty_cache_steps=10,
            prediction_loss_only=False,
            gradient_checkpointing=False,
            skip_memory_metrics=True,
            auto_find_batch_size=False,
        )
        if training_args == None:
            self.training_args = default_training_args
        else:
            self.training_args = training_args
        self.max_sequence_length = max_sequence_length
        self.model_max_length = model_max_length
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            foundation_model_huggingface,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        self.data_collator = DataCollatorForFineTunedDataset(
            tokenizer=self.tokenizer, soft_labels=self.soft_labels
        )
        self.trainer = None
        self.history: List[dict] = []

    def __str__(self):
        return str(self.model)

    def _init_trainer(
        self,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        signal_mask=None,
        bg_ratio: float = 0.3,
    ):
        common_kwargs = dict(
            model=self.model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            tokenizer=self.tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_prediction,
            compute_metrics=compute_metrics,
        )

        if signal_mask is None:
            trainer = AuxLossLoggingTrainer(**common_kwargs)
        else:
            trainer = BalancedTrainer(
                signal_mask=signal_mask,
                bg_ratio=bg_ratio,
                **common_kwargs,
            )

        use_table_progress_callback(trainer)
        return trainer

    def predict(self, test_dataset, batch_size=None, clear_cache=True):
        if batch_size is not None:
            training_args = TrainingArguments(
                eval_strategy="no",
                save_strategy="no",
                gradient_checkpointing=False,
                skip_memory_metrics=True,
                auto_find_batch_size=False,
                per_device_eval_batch_size=batch_size,
                output_dir=self.training_args.output_dir,
            )
        else:
            # TODO Must be a better way
            # Also, when using Flash Attention, we are forced to have batch size as multiples of 64 to avoid race conditions.
            if self.max_sequence_length <= 1000:
                training_args = TrainingArguments(
                    eval_strategy="no",
                    save_strategy="no",
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=False,
                    per_device_eval_batch_size=64 * 6,
                    output_dir=self.training_args.output_dir,
                )
            elif self.max_sequence_length <= 2000:
                training_args = TrainingArguments(
                    eval_strategy="no",
                    save_strategy="no",
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=False,
                    per_device_eval_batch_size=64 * 3,
                    output_dir=self.training_args.output_dir,
                )
            elif self.max_sequence_length <= 3000:
                training_args = TrainingArguments(
                    eval_strategy="no",
                    save_strategy="no",
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=False,
                    per_device_eval_batch_size=64,
                    output_dir=self.training_args.output_dir,
                )
            else:
                training_args = TrainingArguments(
                    eval_strategy="no",
                    save_strategy="no",
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=True,
                    output_dir=self.training_args.output_dir,
                )
        prediction_trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_prediction,
            compute_metrics=None,
        )

        prediction = prediction_trainer.predict(test_dataset)
        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()
        # prediction = self.trainer.predict(test_dataset)
        # gc.collect()
        # torch.cuda.empty_cache()
        return prediction

    def safe_save_model_for_hf_trainer(self, output_dir: str):
        """Collects the state dict and dump to disk."""
        state_dict = self.trainer.model.state_dict()
        if self.trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            self.trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

    def fine_tune(
        self,
        data_path: Optional[str] = None,
        training_args: Union[TrainingArguments, None] = None,
        train_dataset: Optional["DNABERT2FineTuneDataset"] = None,
        val_dataset: Optional["DNABERT2FineTuneDataset"] = None,
        test_dataset: Optional["DNABERT2FineTuneDataset"] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        data_interface: str = "csv",
        resume_from_checkpoint: Optional[Union[bool, str]] = None,
        signal_mask=None,
        bg_ratio: float = 0.3,
    ):

        # Ensure that either data_path is provided or all datasets are provided
        assert data_path or (
            train_dataset and val_dataset and test_dataset
        ), "Either 'data_path' must be provided or all of 'train_dataset', 'val_dataset', and 'test_dataset' must not be None."

        # TODO Adding LoRA
        print("Starting to initialize datasets")
        train_dataset = train_dataset or DNABERT2FineTuneDataset(
            tokenizer=self.tokenizer,
            data_path_or_list=os.path.join(data_path, "train"),
            kmer=-1,
            data_interface=data_interface,
        )
        print("Train is initialized")
        val_dataset = val_dataset or DNABERT2FineTuneDataset(
            tokenizer=self.tokenizer,
            data_path_or_list=os.path.join(data_path, "valid"),
            kmer=-1,
            data_interface=data_interface,
        )
        print("Val is initialized")

        if training_args is not None:
            self.training_args = training_args  # overwritting default training args

        self.trainer = self._init_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=self.training_args,
            callbacks=callbacks,
            signal_mask=signal_mask,
            bg_ratio=bg_ratio,
        )

        print("All datasets are successfully initiated")
        # Determine checkpoint resumption strategy
        checkpoint_path = None
        if resume_from_checkpoint is not None:
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                # Resume from the last checkpoint in output_dir
                checkpoint_path = True
            elif isinstance(resume_from_checkpoint, str):
                # Resume from specific checkpoint path
                checkpoint_path = resume_from_checkpoint
        elif hasattr(self, "resume_from_checkpoint") and self.resume_from_checkpoint:
            # Use checkpoint path from initialization if provided
            checkpoint_path = self.resume_from_checkpoint

        self.trainer.train(resume_from_checkpoint=checkpoint_path)
        if self.training_args.save_model:
            self.trainer.save_state()
            self.safe_save_model_for_hf_trainer(output_dir=training_args.output_dir)

    def required_fit_columns(self, config: dict) -> list[str]:
        training = config.get("training", {}) or {}
        cols = ["input_ids", "methylation_ids"]
        if self.num_grg_labels is not None:
            cols.append(training.get("grg_label_column", "dmr_ctype_label"))
        return cols

    def mlflow_fit_params(self) -> dict:
        return {
            "num_labels": self.num_labels,
            "num_grg_labels": self.num_grg_labels,
            "soft_labels": self.soft_labels,
        }

    @mlflow_tracked_fit
    def fit_classificaton(
        self,
        train_df: pd.DataFrame,
        val_df: Union[pd.DataFrame, None] = None,
        output_dir: Union[str, Path, None] = None,
        **kwargs,
    ) -> "EpigenDnabert2":
        """Fit the classifier on training data for compatibility with AbstractReadClassifier."""
        train_dataset = DNABERT2FineTuneDataset(
            data_path_or_list=train_df,
            tokenizer=self.tokenizer,
            kmer=-1,
            data_interface="pandas",
            lazy_tokenization=True,
            include_grg_ids=self.num_grg_labels is not None,
            grg_label_column=kwargs.get("grg_label_column", "dmr_ctype_label"),
            soft_labels=self.soft_labels,
        )

        # fine_tune() requires non-None train/val/test datasets when no data_path is
        # given; fall back to the training set when no validation split is provided.
        val_dataset = train_dataset
        if val_df is not None:
            val_dataset = DNABERT2FineTuneDataset(
                data_path_or_list=val_df,
                tokenizer=self.tokenizer,
                kmer=-1,
                data_interface="pandas",
                lazy_tokenization=True,
                include_grg_ids=self.num_grg_labels is not None,
                grg_label_column=kwargs.get("grg_label_column", "dmr_ctype_label"),
                soft_labels=self.soft_labels,
            )

        # Apply any training_args overrides (e.g. from a YAML config dict) on
        # top of the TrainingArguments built in __init__, preserving its
        # tuned defaults (e.g. remove_unused_columns=False).
        training_args_overrides = dict(kwargs.get("training_args", {}))
        if "output_dir" not in training_args_overrides:
            training_args_overrides["output_dir"] = (
                str(output_dir) if output_dir else self.training_args.output_dir
            )

        self.training_args = replace(self.training_args, **training_args_overrides)

        # Attach an EarlyStoppingCallback when the config carries an
        # ``early_stopping`` block (no-op otherwise). Prerequisites on the
        # TrainingArguments are auto-enforced inside the helper.
        self.training_args, callbacks = apply_early_stopping(
            self.training_args,
            kwargs.get("early_stopping"),
            kwargs.get("callbacks"),
        )

        self.fine_tune(
            data_path=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=val_dataset,
            training_args=self.training_args,
            callbacks=callbacks,
            data_interface="pandas",
            resume_from_checkpoint=kwargs.get("resume_from_checkpoint", None),
            signal_mask=kwargs.get("signal_mask", None),
            bg_ratio=kwargs.get("bg_ratio", 0.3),
        )
        self.history.append(extract_trainer_metrics(self.trainer))

        if output_dir:
            self.save(output_dir)

        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted classifier to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.trainer is not None:
            self.safe_save_model_for_hf_trainer(str(path))

    def predict_split(self, split_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run DNABERT2 prediction on a single split."""
        dataset = DNABERT2FineTuneDataset(
            data_path_or_list=split_df,
            tokenizer=self.tokenizer,
            kmer=-1,
            data_interface="pandas",
            lazy_tokenization=True,
            include_grg_ids=self.num_grg_labels is not None,
            grg_label_column=kwargs.get("grg_label_column", "dmr_ctype_label"),
            soft_labels=self.soft_labels,
        )

        prediction_output = self.predict(
            dataset, batch_size=kwargs.get("batch_size", None)
        )
        probabilities = prediction_output.predictions

        # Apply softmax or sigmoid
        if self.num_labels == 1:
            probabilities = 1 / (1 + np.exp(-probabilities))
        else:
            exp_preds = np.exp(
                probabilities - np.max(probabilities, axis=1, keepdims=True)
            )
            probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        pred_cols = [f"prediction_{i}" for i in range(self.num_labels)]
        pred_df = pd.DataFrame(probabilities, columns=pred_cols)

        result = split_df.copy()
        for col in pred_cols:
            result[col] = pred_df[col].values

        return result

    @classmethod
    def load(cls, path: Union[str, Path, None] = None, **kwargs) -> "EpigenDnabert2":
        """Load EpigenDnabert2 from a checkpoint, or build a fresh instance if path is None."""
        classifier_head_implementation = kwargs.get(
            "classifier_head_implementation", "grg_attention_based"
        )
        num_grg_labels = kwargs.get("num_grg_labels")
        if (
            num_grg_labels is None
            and classifier_head_implementation == "grg_attention_based"
        ):
            num_grg_labels = 39
        return cls(
            foundation_model_huggingface=kwargs.get("foundation_model_path")
            or "zhihan1996/DNABERT-2-117M",
            fine_tuned_model_path=str(path) if path else None,
            max_sequence_length=kwargs.get("seq_length", 150),
            num_labels=kwargs.get("num_labels", 2),
            use_cpg_methylation=kwargs.get("use_cpg_methylation", True),
            use_m6a_methylation=kwargs.get("use_m6a_methylation", False),
            num_grg_labels=num_grg_labels,
            soft_labels=kwargs.get("soft_labels", False),
        )
