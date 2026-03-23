import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from torch.nn.modules.loss import _Loss

import random 
import numpy as np
from transformers.trainer_callback import (
    TrainerCallback
)

from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import os
import gc
import numpy as np
from typing import Optional, Tuple, Union, List
from methyldl.modelling.utils import calculate_batch_size

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertConfig
)

from collections import OrderedDict
import itertools
from methyldl.modelling.evaluation import compute_metrics, preprocess_logits_for_prediction, compute_metrics_soft_labels

from torch.utils.data import Dataset
import numpy as np
from copy import deepcopy
import multiprocessing as mp
from functools import partial
import pickle
from methyldl.modelling.common import DMRAttentionClassifier
from methyldl.modelling.loss import ConfidenceWeightedCrossEntropy, OnTargetSoftLoss


default_methylbert_config = OrderedDict([
    ("lr", 0.0004),
    ("beta", (0.9, 0.98)),
    ("weight_decay", 0.1),
    ("warmup_step", 100),
    ("eps", 1e-6),
    ("with_cuda", True),
    ("log_freq", 10),
    ("eval_freq", 10),
    ("n_hidden", None),
    ("decrease_steps", 200),
    ("eval", False),
    ("amp", True),
    ("gradient_accumulation_steps", 1),
    ("max_grad_norm", 1.0),
    ("save_freq", None),
    ("loss", "bce"),
    ("adam_beta1", 0.9),
    ("adam_beta2", 0.98),
    ("seed", 950410),
    
])

def methylbert_finetune_collator(features):
    """
    Batch features that are already in the correct format (hard labels).
    """
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "token_type_ids": torch.stack([f["token_type_ids"] for f in features]),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        "dmr_ids": torch.tensor([f["dmr_ids"] for f in features], dtype=torch.long),
    }

def methylbert_finetune_soft_collator(features):
    """
    Batch features for soft-label fine-tuning.
    Labels are stacked as float tensors of shape [batch_size, num_classes].
    """
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "token_type_ids": torch.stack([f["token_type_ids"] for f in features]),
        "labels": torch.stack([f["labels"] if isinstance(f["labels"], torch.Tensor) else torch.tensor(f["labels"], dtype=torch.float) for f in features]),
        "dmr_ids": torch.tensor([f["dmr_ids"] for f in features], dtype=torch.long),
        "on_target_mask":  torch.stack([f["on_target_mask"] for f in features])
    }


def methylbert_pretrain_collator(features):
    """
    Convert items from MethylBertPretrainDataset into a single batch dict 
    suitable for a pretraining forward pass (if you adapt your model).
    """
    bert_input = [f["bert_input"] for f in features]
    bert_label = [f["bert_label"] for f in features]
    bert_mask = [f["bert_mask"] for f in features]

    batch_input_ids = torch.stack(bert_input, dim=0)
    batch_labels = torch.stack(bert_label, dim=0)
    batch_mask = torch.stack(bert_mask, dim=0)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,        # or "masked_lm_labels" if your model uses that
        "bert_mask": batch_mask        # optional, depending on your forward
    }


@dataclass
class MethylBertOutput(ModelOutput):
    """
    Custom output type for MethylBertEmbeddedDMR, 
    so we can include both the standard classification 
    outputs and extra `dmr_logits`.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None              # ctype_logits
    dmr_logits: Optional[torch.FloatTensor] = None  # e.g. appended hidden states
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/resolve/main/pytorch_model.bin",
    "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/resolve/main/pytorch_model.bin",
}


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    This code is from : https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py


    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

class FocalLoss(_Loss):
    
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, reduction=self.reduction)

@dataclass
class MethylBertOutput(ModelOutput):
    """
    Custom output type for MethylBertEmbeddedDMR.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None              
    dmr_logits: Optional[torch.FloatTensor] = None  
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[torch.FloatTensor] = None  # For attention-based classifier


class VanillaClassifier(nn.Module):
    """
    Original vanilla classifier with DMR encoding and flattening.
    Extracted as a separate module for clarity.
    """
    def __init__(self, config, seq_len=150):
        super().__init__()
        self.seq_len = seq_len
        self.num_labels = config.num_labels
        self.num_dmr_labels = config.num_dmr_labels
        
        # DMR encoder (embedding)
        self.dmr_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_dmr_labels, embedding_dim=seq_len+1),
        )
        
        # Read classifier with flattening
        self.read_classifier = nn.Sequential(
            nn.Linear((config.hidden_size+1)*(seq_len+1), seq_len+1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.LayerNorm(seq_len+1, eps=config.layer_norm_eps),
            nn.Linear(seq_len+1, self.num_labels)
        )
    
    def forward(self, sequence_output, dmr_ids):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size] - BERT output after dropout
            dmr_ids: [batch_size] - DMR labels
        
        Returns:
            logits: [batch_size, num_labels] - classification logits
            sequence_output_with_dmr: [batch_size, seq_len, hidden_size+1] - for backward compatibility
        """
        batch_size = sequence_output.size(0)
        
        # DMR embedding
        dmr_embedding = self.dmr_encoder(dmr_ids.view(-1))  # [batch_size, seq_len+1]
        
        # Append DMR embedding to each position in the sequence
        # shape -> [batch_size, seq_len, hidden_size+1]
        sequence_output_with_dmr = torch.cat(
            (sequence_output, dmr_embedding.unsqueeze(-1)), 
            dim=-1
        )
        
        # Flatten for classifier
        flat_seq = sequence_output_with_dmr.view(batch_size, -1)  
        logits = self.read_classifier(flat_seq)  # [batch_size, num_labels]
        
        return logits, sequence_output_with_dmr


class MethylBertEmbeddedDMR(BertPreTrainedModel):
    """
    Extended MethylBERT with support for both vanilla and attention-based classifiers.
    """
    pretrained_model_archive_map = {
        "hanyangii/methylbert_hg19_12l": "https://huggingface.co/hanyangii/methylbert_hg19_12l/resolve/main/pytorch_model.bin",
        "hanyangii/methylbert_hg19_8l": "https://huggingface.co/hanyangii/methylbert_hg19_8l/resolve/main/pytorch_model.bin",
        "hanyangii/methylbert_hg19_6l": "https://huggingface.co/hanyangii/methylbert_hg19_6l/resolve/main/pytorch_model.bin",
        "hanyangii/methylbert_hg19_4l": "https://huggingface.co/hanyangii/methylbert_hg19_4l/resolve/main/pytorch_model.bin",
        "hanyangii/methylbert_hg19_2l": "https://huggingface.co/hanyangii/methylbert_hg19_2l/resolve/main/pytorch_model.bin",
    }
    base_model_prefix = "methylbert"

    def __init__(self, config, seq_len=150, classifier_implementation="vanilla"):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_dmr_labels = config.num_dmr_labels
        self.classifier_implementation = classifier_implementation
        
        # Ensure loss is in config
        if not hasattr(config, 'loss'):
            config.loss = "bce"  # Default loss
        
        if config.loss not in ["bce", "focal_bce", "ce", "cwce", "on_target_ce"]:
            raise ValueError(f"loss must be bce, focal_bce, or ce. {config.loss} is given.")
        
        self.loss = config.loss
        self.classification_loss_fct = self._setup_loss(self.loss)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_len = seq_len
        
        # Initialize the appropriate classifier based on implementation choice
        if classifier_implementation == "vanilla":
            print("Using vanilla classifier with DMR encoding and flattening")
            # For vanilla, we create the components directly (no VanillaClassifier wrapper)
            # This avoids tensor sharing issues
            self.dmr_encoder = nn.Sequential(
                nn.Embedding(num_embeddings=self.num_dmr_labels, embedding_dim=seq_len+1),
            )
            
            self.read_classifier = nn.Sequential(
                nn.Linear((config.hidden_size+1)*(seq_len+1), seq_len+1),
                nn.Dropout(0.05),
                nn.ReLU(),
                nn.LayerNorm(seq_len+1, eps=config.layer_norm_eps),
                nn.Linear(seq_len+1, self.num_labels)
            )
            self.classifier = None  # No separate classifier module for vanilla
            
        elif classifier_implementation == "dmr_attention_based":
            print("Using attention-based classifier with DMR context")
            self.classifier = DMRAttentionClassifier(config)
            # These won't be used in attention mode but set to None for clarity
            self.read_classifier = None
            self.dmr_encoder = None
        else:
            raise ValueError(f"Unknown classifier implementation: {classifier_implementation}. "
                           "Choose 'vanilla' or 'dmr_attention_based'")
        
        self.init_weights()

    def _setup_loss(self, loss):
        if loss == "bce":
            print("Binary Cross Entropy loss assigned")
            return nn.BCEWithLogitsLoss()
        elif loss == "focal_bce":
            print("Focal loss assigned")
            return FocalLoss()
        elif loss == "ce":
            print("Cross Entropy loss assigned (multi-class)")
            return nn.CrossEntropyLoss()
        elif loss == "cwce":
            print("Confidence Weighted Cross Entropy assigned (multi-class)")
            return ConfidenceWeightedCrossEntropy(self.num_labels)
        elif loss == "on_target_ce":
            print("On Target Confidence Weighted Cross Entropy assigned (multi-class)")
            Warning("Make sure that dmr_ids are matching target labels!!! Otherwise, it wouldn't work and your model will likely not to learn anything usefull.")
            return OnTargetSoftLoss(self.num_labels)
        else:
            raise ValueError(f"Unknown loss type: {loss}")
    
    def check_model_status(self):
        print(f"Bert model training mode: {self.bert.training}")
        print(f"Dropout training mode: {self.dropout.training}")
        print(f"Classifier ({self.classifier_implementation}) training mode: {self.classifier.training}")
        
    def from_pretrained_read_classifier(self, pretrained_model_name_or_path, device="cpu"):
        if self.classifier_implementation == "vanilla":
            self.classifier.read_classifier.load_state_dict(
                torch.load(pretrained_model_name_or_path, map_location=device)
            )
        else:
            print("Warning: from_pretrained_read_classifier is only applicable for vanilla classifier")
        
    def from_pretrained_dmr_encoder(self, pretrained_model_name_or_path, device="cpu"):
        if self.classifier_implementation == "vanilla":
            self.classifier.dmr_encoder.load_state_dict(
                torch.load(pretrained_model_name_or_path, map_location=device)
            )
        else:
            print("Warning: from_pretrained_dmr_encoder is only applicable for vanilla classifier")
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  # 'methyl_seq' as token_type_ids
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,  # Cell Type labels
        dmr_ids=None,  # DMR labels
        on_target_mask=None # Whether or not the read is on target
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = self.dropout(outputs[0])  # [batch, seq_len, hidden_size]
        
        # Apply the appropriate classifier
        if self.classifier_implementation == "vanilla":
            # DMR embedding
            dmr_embedding = self.dmr_encoder(dmr_ids.view(-1))  # [batch, seq_len+1]
            # Append along last dimension
            # shape -> [batch, seq_len, hidden_size+1]
            sequence_output = torch.cat((sequence_output, dmr_embedding.unsqueeze(-1)), dim=-1)

            # TODO: Think about more elegant way to incorporate dmr_embeddings data before classification. 

            # Flatten for classifier
            batch_size = sequence_output.size(0)
            flat_seq = sequence_output.view(batch_size, -1)  
            ctype_logits = self.read_classifier(flat_seq)  # shape [batch, n_classes]
            dmr_logits=sequence_output
            attention_weights = None
        else:  # dmr_attention_based
            ctype_logits, attention_weights = self.classifier(
                sequence_output, dmr_ids, attention_mask
            )
            dmr_logits = None  # Not applicable for attention-based
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = self.classification_loss_fct(ctype_logits.squeeze(), labels.float())
            elif labels.ndim >= 2 and labels.shape[-1] == self.num_labels:
                # Soft labels: already a [B, C] probability distribution
                if self.loss == "on_target_ce":
                    loss = self.classification_loss_fct(ctype_logits, labels.float(),on_target_mask)
                else:
                    loss = self.classification_loss_fct(ctype_logits, labels.float())
            elif self.num_labels >= 2 and self.loss in ["bce", "focal_bce"]:
                # Hard labels with BCE/focal: one-hot encode first
                ctype_label_onehot = F.one_hot(labels, num_classes=self.num_labels).float()
                loss = self.classification_loss_fct(ctype_logits, ctype_label_onehot)
            else:
                # Hard labels with CE
                loss = self.classification_loss_fct(ctype_logits, labels)
        
        return MethylBertOutput(
            loss=loss,
            logits=ctype_logits,
            dmr_logits=dmr_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_weights=attention_weights  # For interpretability in attention-based classifier
        )


class MethylBert:
    """
    High-level wrapper class for MethylBERT with support for classifier selection.
    """
    def __init__(
        self,
        foundation_model_path: str,
        seq_len: int = 150,
        custom_config: OrderedDict = None,
        load_weights: bool = True,
        fine_tuned_model_path: Optional[str] = None,
        num_labels: int = 2,
        num_dmr_labels: int = 100,
        output_dir: str = "tmp_trainer",
        batch_size = None,
        lazy_tokenization=False,
        cache_dir="./cache",
        classifier_implementation: str = "vanilla",
        soft_labels: bool = False
    ):
        """
        Extended initialization with classifier implementation and soft label selection.
        
        Args:
            ... (existing parameters) ...
            classifier_implementation: str
                Choice of classifier: "vanilla" or "dmr_attention_based"
            soft_labels: bool
                If True, use soft-label collator and tokenizer for probability vectors.
        """
        if custom_config is None:
            raise ValueError("Must provide a custom_config dictionary.")
        
        self._config = custom_config
        self.output_dir = output_dir
        self.lazy_tokenization = lazy_tokenization
        self.cache_dir = cache_dir
        self.classifier_implementation = classifier_implementation
        self.soft_labels = soft_labels
        
        # Validate classifier implementation
        if classifier_implementation not in ["vanilla", "dmr_attention_based"]:
            raise ValueError(f"classifier_implementation must be 'vanilla' or 'dmr_attention_based', "
                           f"got {classifier_implementation}")
        
        # Load BERT config
        if os.path.isdir(foundation_model_path):
            config = BertConfig.from_pretrained(foundation_model_path)
        else:
            config = BertConfig.from_pretrained(foundation_model_path)
        
        config.num_labels = num_labels
        config.num_dmr_labels = num_dmr_labels
        config.loss = self._config["loss"]
        
        # Validate loss type based on num_labels
        if num_labels == 1 and config.loss not in ["bce"]:
            print(f"Warning: num_labels=1 typically uses 'bce' loss, but '{config.loss}' was specified")
        elif num_labels == 2 and config.loss not in ["bce", "focal_bce", "ce"]:
            raise ValueError(f"For binary classification (num_labels=2), loss must be 'bce', 'focal_bce', or 'ce'")
        elif num_labels > 2 and config.loss not in ["ce"]:
            print(f"Warning: Multi-class classification (num_labels={num_labels}) typically uses 'ce' loss")
        
        self.seq_len = seq_len
        
        # Build the model with selected classifier implementation
        if not load_weights:
            print(f"Initializing MethylBertEmbeddedDMR with {classifier_implementation} classifier from config only")
            self.model = MethylBertEmbeddedDMR(
                config, 
                seq_len=seq_len,
                classifier_implementation=classifier_implementation
            )
        else:
            if fine_tuned_model_path:
                print(f"Loading MethylBertEmbeddedDMR with {classifier_implementation} classifier "
                      f"from fine-tuned path: {fine_tuned_model_path}")
                # Note: When loading a pretrained model, you might need to handle
                # the classifier_implementation parameter appropriately
                self.model = MethylBertEmbeddedDMR.from_pretrained(
                    pretrained_model_name_or_path=fine_tuned_model_path,
                    config=config,
                    seq_len=seq_len,
                    classifier_implementation=classifier_implementation,
                    use_safetensors=True
                )
            else:
                print(f"Loading MethylBertEmbeddedDMR with {classifier_implementation} classifier "
                      f"from foundation path: {foundation_model_path}")
                self.model = MethylBertEmbeddedDMR.from_pretrained(
                    foundation_model_path,
                    config=config,
                    seq_len=seq_len,
                    classifier_implementation=classifier_implementation
                )
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(foundation_model_path)
        except:
            self.tokenizer = None
        
        # Calculate recommended batch size
        if batch_size is None:
            from methyldl.modelling.utils import calculate_batch_size
            recommended_batch_size = calculate_batch_size(gb_per_seq=0.0135*2, cpu_batch_size=700/2)
        else:
            recommended_batch_size = batch_size
        
        # Create default TrainingArguments
        from transformers import TrainingArguments
        default_training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self._config["lr"],
            warmup_steps=self._config["warmup_step"],
            weight_decay=self._config["weight_decay"],
            adam_beta1=self._config["beta"][0],
            adam_beta2=self._config["beta"][1],
            adam_epsilon=self._config["eps"],
            fp16=self._config["amp"],
            max_grad_norm=self._config["max_grad_norm"],
            gradient_accumulation_steps=self._config["gradient_accumulation_steps"],
            logging_steps=self._config["log_freq"],
            eval_steps=self._config["eval_freq"],
            save_steps=self._config["eval_freq"],
            per_device_train_batch_size=recommended_batch_size,
            per_device_eval_batch_size=recommended_batch_size,
            num_train_epochs=100,
            eval_strategy="steps",
            remove_unused_columns=False,
            eval_accumulation_steps=8,
            torch_empty_cache_steps=10,
            prediction_loss_only=False,
            gradient_checkpointing=True,
            skip_memory_metrics=True,
            auto_find_batch_size=False,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            run_name=f"methylBERT_{classifier_implementation}"
        )
        
        self.training_args = default_training_args
        self.hf_config = config
        self.trainer = None

    def _init_trainer(self, 
                      train_dataset=None, 
                      eval_dataset=None,
                      data_collator=None,
                      custom_training_args=None,
                      prediction_mode = False,
                      callbacks=None,
                      batch_size=None):
        """
        Internal method to build a HF Trainer.
        """
        if custom_training_args is not None:
            self.training_args = custom_training_args

        # fallback to a user-provided or default data_collator
        if data_collator is None:
            if self.soft_labels:
                data_collator = methylbert_finetune_soft_collator
            else:
                data_collator = methylbert_finetune_collator

        args=self.training_args
        if prediction_mode:
            args.eval_strategy = "no"
            args.do_train = False
            args.do_eval = False
            preprocessing_function = preprocess_logits_for_prediction
        else:
            preprocessing_function = preprocess_logits_for_prediction     

        if batch_size is not None:
                args.per_device_eval_batch_size = batch_size   

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocessing_function,
            compute_metrics= compute_metrics if not self.soft_labels else compute_metrics_soft_labels,
            callbacks = callbacks
        )
 
        return trainer

    def fine_tune(
        self,
        data_path,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        data_collator=None,
        training_args=None,
        callbacks: Optional[List[TrainerCallback]] = None,
        resume_from_checkpoint: Optional[Union[bool, str]] = None
    ):
        """
        Fine-tune your model on a training set, optional validation set, etc.
        """
        assert data_path or (train_dataset and val_dataset), (
            "Either 'data_path' must be provided or all of 'train_dataset' and 'val_dataset' must not be None."
        )

        train_dataset = train_dataset or MethylBertFinetuneDataset(
              data_source=os.path.join(data_path, "train.txt"),
              vocab=MethylVocab(k=3),
              seq_len=self.seq_len,
              lazy_tokenization=self.lazy_tokenization,
              cache_dir=self.cache_dir
              )
        val_dataset = val_dataset or MethylBertFinetuneDataset(
              data_source=os.path.join(data_path, "valid.txt"),
              vocab=MethylVocab(k=3),
              seq_len=self.seq_len,
              lazy_tokenization=self.lazy_tokenization,
              cache_dir=self.cache_dir
              )
        # test_dataset = test_dataset or MethylBertFinetuneDataset(
        #       data_source=os.path.join(data_path, "test.txt"),
        #       vocab=MethylVocab(k=3),
        #       seq_len=self.seq_len
        #       )
        self.trainer = self._init_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            custom_training_args=training_args,
            callbacks = callbacks
        )
        checkpoint_path = None
        if resume_from_checkpoint is not None:
            if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
                # Resume from the last checkpoint in output_dir
                checkpoint_path = True
            elif isinstance(resume_from_checkpoint, str):
                # Resume from specific checkpoint path
                checkpoint_path = resume_from_checkpoint
        elif hasattr(self, 'resume_from_checkpoint') and self.resume_from_checkpoint:
            # Use checkpoint path from initialization if provided
            checkpoint_path = self.resume_from_checkpoint

        self.trainer.train(resume_from_checkpoint=checkpoint_path)

        # if test_dataset:
        #     results = self.trainer.evaluate(test_dataset)
        #     print("Test results:", results)

        # # Save final model
        # # if self.training_args.save_model:
        # if True:
        #     self.trainer.save_state()
        #     self.safe_save_model_for_hf_trainer(output_dir=self.output_dir)

        # get the evaluation results from trainer
        # if training_args.eval_and_save_results:
        # if True:
        #     results_path = os.path.join(self.output_dir, "results", training_args.run_name)
        #     results = self.trainer.evaluate(eval_dataset=test_dataset)
        #     os.makedirs(results_path, exist_ok=True)
        #     with open(os.path.join(results_path, "eval_results.json"), "w") as f:
        #         json.dump(results, f)

        # if self.trainer.args.should_save:
        #     self.trainer.save_model(self.trainer.args.output_dir)
        #     self.safe_save_model_for_hf_trainer(self.trainer.args.output_dir)

    def predict(self, dataset, data_collator=None, batch_size = None,clear_cache=True):
        """
        Use Hugging Face Trainer for prediction on a dataset.
        """
        if self.trainer is None:
            # build a trainer for inference
            self.trainer = self._init_trainer(
                data_collator=data_collator,
                prediction_mode=True,
                batch_size = batch_size
                # no train or val dataset
            )
        predictions = self.trainer.predict(dataset)
        if clear_cache:
            gc.collect()
            torch.cuda.empty_cache()
        return predictions

    def safe_save_model_for_hf_trainer(self, output_dir: str):
        """
        Save final state_dict to disk, offloading to CPU if needed.
        """
        state_dict = self.trainer.model.state_dict()
        if self.trainer.args.should_save:
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            del state_dict
            self.trainer._save(output_dir, state_dict=cpu_state_dict)

    def __str__(self):
        return str(self.model)
        
class MethylVocab(object):
    def __init__(self, k: int=3):
        '''
        Create a look-up table to convert 3-mer tokens to numerical identifiers 

        k: int
            k to create k-mer sequences 
        '''
        print("Building Vocab")
        self.kmers=k

        # Create a look up table with 3-mer tokens
        bases = ["A","G","T","C"]

        vocabs = list(itertools.product(bases, repeat=self.kmers))
        vocabs = sorted(["".join(e) for e in vocabs]) #alphabetical orders

        # Set up special tokens
        special_tokens=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        
        self.itos = list(special_tokens) + vocabs
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    
    def __len__(self):
        return len(self.itos)

    def to_seq(self, sequence) -> list:
        '''
        Convert a 3-mer sequence

        sequence: str or list(str)
            A 3-mer sequence to convert. It can be given as either a string or a list of 3-mer strings 

        '''
        if isinstance(sequence, str):
            sentence = sequence.split()

        seq = [self.stoi.get(kmer, self.unk_index) for kmer in sequence]
        return seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

def _line2tokens_pretrain(l, tokenizer, max_len=120):
	'''
		convert a text line into a list of tokens converted by tokenizer 
			
	'''

	l = l.strip().split(" ")

	tokened = [tokenizer.to_seq([b]) for b in l]
	if len(tokened) > max_len:
		return tokened[:max_len]
	else:
		return tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))]


def _line2tokens_finetune(l, tokenizer, max_len=150, headers=None):
    # Check the header
    if not all([h in headers for h in ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]]):
        raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")
    
    max_len = min(max_len, 511) # Cannot have more then 510 tokens in sequence due to positional embeddings 

	# Separate n-mers tokens and labels from each line 
    l = l.strip().split("\t")
    if len(headers) == len(l):
        l = {k: v for k, v in zip(headers, l)}
    else:
        print(headers, l)
        raise ValueError(f"Only {len(headers)} elements are in the input file header, whereas the line has {len(l)} elements.")
    
    l["dna_seq"] = l["dna_seq"].split(" ")
    l["dna_seq"] = [[f] for f in tokenizer.to_seq(l["dna_seq"])]
    l["methyl_seq"] = [int(m) for m in l["methyl_seq"]]

    l["ctype_label"] = int(l["ctype"])
    l["dmr_label"] = int(l["dmr_label"])
    
    if len(l["dna_seq"]) > max_len:
        l["dna_seq"] = l["dna_seq"][:max_len]
        l["methyl_seq"] = l["methyl_seq"][:max_len]
    else:
        cur_seq_len=len(l["dna_seq"])
        l["dna_seq"] = l["dna_seq"]+[[tokenizer.pad_index] for k in range(max_len-cur_seq_len)]
        l["methyl_seq"] = l["methyl_seq"] + [2 for k in range(max_len-cur_seq_len)]
    
    return l


def _line2tokens_finetune_soft(l, tokenizer, max_len=150, headers=None):
    """
    Like _line2tokens_finetune but parses soft labels.
    The 'ctype' column should contain comma-separated floats, e.g. "0.1,0.0,0.9".
    """
    if not all([h in headers for h in ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]]):
        raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")
    
    max_len = min(max_len, 511)

    l = l.strip().split("\t")
    if len(headers) == len(l):
        l = {k: v for k, v in zip(headers, l)}
    else:
        print(headers, l)
        raise ValueError(f"Only {len(headers)} elements are in the input file header, whereas the line has {len(l)} elements.")
    
    l["dna_seq"] = l["dna_seq"].split(" ")
    l["dna_seq"] = [[f] for f in tokenizer.to_seq(l["dna_seq"])]
    l["methyl_seq"] = [int(m) for m in l["methyl_seq"]]

    # Parse soft labels: comma-separated float vector, or array/list directly
    if isinstance(l["ctype"], str):
        l["ctype_label"] = [float(x) for x in l["ctype"].split(",")]
    else:
        # Already a list or numpy array (e.g. from in-memory data)
        l["ctype_label"] = list(l["ctype"])
    l["dmr_label"] = int(l["dmr_label"])
    
    if len(l["dna_seq"]) > max_len:
        l["dna_seq"] = l["dna_seq"][:max_len]
        l["methyl_seq"] = l["methyl_seq"][:max_len]
    else:
        cur_seq_len=len(l["dna_seq"])
        l["dna_seq"] = l["dna_seq"]+[[tokenizer.pad_index] for k in range(max_len-cur_seq_len)]
        l["methyl_seq"] = l["methyl_seq"] + [2 for k in range(max_len-cur_seq_len)]
    
    return l

class MethylBertDataset(Dataset):
	def __init__(self):
		pass
			
	def __len__(self):
		return self.lines.shape[0] if type(self.lines) == np.array else len(self.lines)


class MethylBertPretrainDataset(MethylBertDataset):
    def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, random_len=False, n_cores=10):

        self.vocab = vocab
        self.seq_len = seq_len
        self.f_path = f_path
        self.random_len = random_len

        # Define a range of tokens to mask based on k-mers
        self.mask_list = self._get_mask()

        # Read all text files and convert the raw sequence into tokens
        with open(self.f_path, "r") as f_input:
            print("Open data : %s" % f_input)
            raw_seqs = f_input.read().splitlines()

        num_lines = len(raw_seqs)
        print("Total number of sequences : ", num_lines)

        # Fix 1: Disable multiprocessing for small datasets
        if num_lines < 10000:
            # Just run in the main process
            line_labels = map(
                partial(_line2tokens_pretrain, tokenizer=self.vocab, max_len=self.seq_len), 
                raw_seqs
            )
            line_labels = list(line_labels)

        else:
            # Multiprocessing for the sequence tokenization
            with mp.Pool(n_cores) as pool:
                line_labels = pool.map(
                    partial(_line2tokens_pretrain, tokenizer=self.vocab, max_len=self.seq_len),
                    raw_seqs
                )
        
        del raw_seqs
        print("Lines are processed")
        self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
        if num_lines == 1:
            # Wrapping in one more dimension for this edge case
            self.lines = torch.unsqueeze(self.lines, 0) 
        del line_labels
        gc.collect()

    def __getitem__(self, index):

        dna_seq = self.lines[index].clone()
        # Random len
        if self.random_len and np.random.random() < 0.5:
            dna_seq = dna_seq[:random.randint(5, self.seq_len)]

        # Padding
        if dna_seq.shape[0] < self.seq_len:
            pad_num = self.seq_len - dna_seq.shape[0]
            dna_seq = torch.cat(
                (
                    dna_seq,
                    torch.tensor([self.vocab.pad_index for _ in range(pad_num)], dtype=torch.int16)
                )
            )

        # Mask
        masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)

        return {
            "bert_input": masked_dna_seq,
            "bert_label": dna_seq,
            "bert_mask": bert_mask
        }

    def subset_data(self, n_seq: int):
        self.lines = random.sample(self.lines, n_seq)

    def _get_mask(self):
        """
        Relative positions from the center of masked region
        e.g) [-1, 0, 1] for 3-mers
        """
        half_length = int(self.vocab.kmers / 2)
        mask_list = [-1 * half_length + i for i in range(half_length)] + [i for i in range(1, half_length + 1)]
        if self.vocab.kmers % 2 == 0:
            mask_list = mask_list[:-1]

        return mask_list

    def _masking(self, inputs: torch.Tensor, threshold=0.15):
        """
        Modified version of a token masking function
        Originally developed by Huggingface (datacollator) and DNABERT

        https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747
        https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

        Added additional tasks to handle each sequence.
        Lines using tokenizer were modified due to different tokenizer object structure.
        """

        labels = inputs.clone()

        # Sample tokens with given probability threshold
        probability_matrix = torch.full(labels.shape, threshold)  # tensor filled with 0.15

        # Handle special tokens (sub-5) -- adjust to your actual logic
        special_tokens_mask = [val < 5 for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        # If you want to also mask out padding (uncomment if needed):
        # padding_mask = labels.eq(self.vocab.pad_index)
        # probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Identify the end of sequence (non-zero probability region)
        end = torch.where(probability_matrix != 0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_indices == 1)[0].tolist())
        new_centers = deepcopy(mask_centers)

        # Extend mask to neighbors (k-mers)
        for center in mask_centers:
            for mask_number in self.mask_list:
                current_index = center + mask_number
                if 0 <= current_index <= end:
                    new_centers.add(current_index)

        new_centers = list(new_centers)
        masked_indices[new_centers] = True

        # Set labels for unmasked tokens to -100 so they don't contribute to loss
        labels[~masked_indices] = -100

        # 80% of the time, replace masked tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.vocab.mask_index

        # 10% of the time, replace masked tokens with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int16)
        inputs[indices_random] = random_words[indices_random]

        # The remaining 10% of the time, keep the original token

        # Special token: EOS (end) 
        if end < inputs.shape[0]:
            inputs[end] = self.vocab.eos_index
        else:
            inputs[-1] = self.vocab.eos_index

        # Add SOS (start) token at the beginning
        labels = torch.cat((torch.tensor([-100]), labels))
        inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
        masked_indices = torch.cat((torch.tensor([False]), masked_indices))

        return inputs, labels, masked_indices

class MethylBertFinetuneDataset(MethylBertDataset):
    def __init__(
        self,
        data_source: Union[str, List[List[Union[str, int]]]],
        vocab: MethylVocab,
        seq_len: int,
        n_cores: int = 10,
        n_seqs: int = None,
        lazy_tokenization: bool = False,
        cache_dir: Optional[str] = None,
        use_mmap: bool = False,
        soft_labels: bool = False
    ):
        """
        MethylBERT dataset with multiple optimization strategies for large datasets.

        Parameters:
        -----------
        data_source : Union[str, List[List[Union[str, int]]]]
            File path (TSV) or in-memory data (list of lists)
        vocab : MethylVocab
            Vocabulary for tokenization
        seq_len : int
            Maximum sequence length
        n_cores : int
            Number of cores for parallel processing (only used if not lazy)
        n_seqs : int, optional
            Limit number of sequences to load
        lazy_tokenization : bool
            If True, tokenize on-the-fly in __getitem__ (saves memory)
        cache_dir : str, optional
            Directory to cache tokenized results to disk
        use_mmap : bool
            If True and cache_dir provided, use memory-mapped arrays for caching
        """
        self.vocab = vocab
        self.seq_len = seq_len
        self.lazy_tokenization = lazy_tokenization
        self.cache_dir = cache_dir
        self.use_mmap = use_mmap
        self.soft_labels = soft_labels
        
        # Select the appropriate tokenizer function
        self._tokenize_fn = _line2tokens_finetune_soft if soft_labels else _line2tokens_finetune
        
        # Create cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(self.cache_dir, "tokenized_cache.pkl")
            self._cache = {}
        
        # Load raw data
        if isinstance(data_source, str):
            self.f_path = data_source
            with open(data_source, "r") as f_input:
                lines = f_input.read().splitlines()
        else:
            self.f_path = None
            header = data_source[0]
            if "dmr_label" not in header:
                header.append("dmr_label")
                for row in data_source[1:]:
                    row.append(0)
            if "dmr_ctype" not in header:
                header.append("dmr_ctype")
                for row in data_source[1:]:
                    row.append(1)
            if "on_target_mask" not in header:
                header.append("on_target_mask")
                for row in data_source[1:]:
                    row.append(0)
            def _serialize_val(v):
                if isinstance(v, np.ndarray):
                    return ",".join(str(x) for x in v)
                return str(v)
            lines = ["\t".join(_serialize_val(x) for x in row) for row in data_source]

        # Parse header and raw sequences
        self.headers = lines[0].split("\t")
        raw_seqs = lines[1:]
        
        if n_seqs is not None:
            raw_seqs = raw_seqs[:n_seqs]
        
        print(f"Total number of sequences: {len(raw_seqs)}")
        
        if lazy_tokenization:
            # LAZY MODE: Store raw strings only
            print("Using lazy tokenization (on-the-fly processing)")
            self.raw_lines = raw_seqs
            self.lines = None  # Not tokenized yet
            
            # Load cache if exists
            if self.cache_dir and os.path.exists(self.cache_file):
                print(f"Loading cache from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"Loaded {len(self._cache)} cached items")
        else:
            # EAGER MODE: Tokenize everything upfront
            print("Using eager tokenization (pre-processing all data)")
            self.raw_lines = None
            
            # Check if cached version exists
            if self.cache_dir and os.path.exists(self.cache_file):
                print(f"Loading pre-tokenized data from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    self.lines = pickle.load(f)
            else:
                # Tokenize all data
                if len(raw_seqs) < 1000:
                    # Small dataset: process sequentially
                    self.lines = [
                        self._tokenize_fn(
                            line, 
                            tokenizer=self.vocab, 
                            max_len=self.seq_len, 
                            headers=self.headers
                        ) for line in raw_seqs
                    ]
                else:
                    # Large dataset: parallel processing
                    with mp.Pool(n_cores) as pool:
                        self.lines = pool.map(
                            partial(
                                self._tokenize_fn,
                                tokenizer=self.vocab,
                                max_len=self.seq_len,
                                headers=self.headers
                            ),
                            raw_seqs
                        )
                
                # Save to cache
                if self.cache_dir:
                    print(f"Saving tokenized data to {self.cache_file}")
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(self.lines, f)
            
            del raw_seqs
            gc.collect()
        
        # Compute statistics
        if not lazy_tokenization:
            self.set_dmr_labels = set([l["dmr_label"] for l in self.lines])
            self.ctype_label_count = self._get_cls_num()
            print("# of reads in each label:", self.ctype_label_count)
        else:
            self.set_dmr_labels = None
            self.ctype_label_count = None

    def _tokenize_single_line(self, index):
        """
        Tokenize a single line (used in lazy mode).
        Implements caching to avoid re-tokenizing the same item.
        """
        # Check cache first
        if self.cache_dir and index in self._cache:
            return self._cache[index]
        
        # Tokenize
        line = self.raw_lines[index]
        tokenized = self._tokenize_fn(
            line,
            tokenizer=self.vocab,
            max_len=self.seq_len,
            headers=self.headers
        )
        
        # Store in cache
        if self.cache_dir:
            self._cache[index] = tokenized
            
            # Periodically save cache to disk (every 1000 items)
            if len(self._cache) % 1000 == 0:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self._cache, f)
        
        return tokenized

    def __len__(self):
        if self.lazy_tokenization:
            return len(self.raw_lines)
        else:
            return len(self.lines)

    def _get_cls_num(self):
        """Count class label distribution."""
        ctype_labels = [l["ctype_label"] for l in self.lines]
        labels = list(set(ctype_labels))
        label_count = np.zeros(len(labels), dtype=int)
        for i, lval in enumerate(labels):
            label_count[i] = ctype_labels.count(lval)
        return label_count

    def num_dmrs(self):
        """Number of possible DMR classes."""
        if self.lazy_tokenization:
            # Compute on-demand if needed
            if self.set_dmr_labels is None:
                dmr_labels = set()
                for i in range(len(self)):
                    item = self._tokenize_single_line(i)
                    dmr_labels.add(item["dmr_label"])
                self.set_dmr_labels = dmr_labels
        return max(len(self.set_dmr_labels), max(self.set_dmr_labels) + 1)

    def subset_data(self, n_seq):
        """Truncate dataset to n_seq samples."""
        if self.lazy_tokenization:
            self.raw_lines = self.raw_lines[:n_seq]
        else:
            self.lines = self.lines[:n_seq]

    def save_cache(self):
        """Manually save cache to disk (useful in lazy mode)."""
        if self.cache_dir and self._cache:
            print(f"Saving cache with {len(self._cache)} items to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)

    def __getitem__(self, index):
        """Return tokenized item with special tokens."""
        # Get tokenized item
        if self.lazy_tokenization:
            item = self._tokenize_single_line(index)
        else:
            item = self.lines[index]
        
        # Deep copy to avoid modifying cached data
        item = deepcopy(item)

        mask_raw = item.get("on_target_mask", False)
        if isinstance(mask_raw, str):
            # Handle string representations like 'False', 'True', '0', '1'
            is_on_target = (mask_raw.lower() in ['true', '1'])
        else:
            is_on_target = bool(mask_raw)
            
        on_target_tensor = torch.tensor(is_on_target, dtype=torch.bool)
        
        # Convert to tensors
        dna_seq = torch.squeeze(torch.tensor(np.array(item["dna_seq"], dtype=np.int32)))
        methyl_seq = torch.squeeze(torch.tensor(np.array(item["methyl_seq"], dtype=np.int8)))

        # Add special tokens (SOS, EOS)
        end_idx = torch.where(dna_seq != self.vocab.pad_index)[0].tolist()[-1] + 1
        if end_idx < dna_seq.shape[0]:
            dna_seq[end_idx] = self.vocab.eos_index
            methyl_seq[end_idx] = 2
        else:
            dna_seq[-1] = self.vocab.eos_index
            methyl_seq[-1] = 2

        dna_seq = torch.cat((torch.tensor([self.vocab.sos_index]), dna_seq))
        methyl_seq = torch.cat((torch.tensor([2]), methyl_seq))

        # For soft labels, return as float tensor; for hard labels, return scalar int
        if self.soft_labels:
            labels = torch.tensor(item["ctype_label"], dtype=torch.float)
        else:
            labels = item["ctype_label"]
        
        return {
            "input_ids": dna_seq,
            "token_type_ids": methyl_seq,
            "labels": labels,
            "dmr_ids": item["dmr_label"],
            "on_target_mask": on_target_tensor
        }

    def __del__(self):
        """Save cache when object is destroyed."""
        if hasattr(self, 'cache_dir') and self.cache_dir and hasattr(self, '_cache'):
            self.save_cache()