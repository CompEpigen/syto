import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from transformers import BertPreTrainedModel, BertModel, BertForMaskedLM
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
import warnings, random 
import numpy as np

from sklearn.metrics import roc_curve, auc, accuracy_score

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

import os
import json
import gc
import numpy as np
from typing import Optional, Union, Dict, List, Callable, Tuple, Any

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertConfig,
    # possibly MethylBERTConfig if you have a custom config
)
from torch.optim import AdamW

from collections import OrderedDict
import itertools
from methyldl.modelling.evaluation import compute_metrics,preprocess_logits_for_metrics

default_methylbert_config = OrderedDict([
    ("lr", 1e-4),
    ("beta", (0.9, 0.999)),
    ("weight_decay", 0.01),
    ("warmup_step", 100),
    ("eps", 1e-6),
    ("with_cuda", True),
    ("log_freq", 20),
    ("eval_freq", 20),
    ("n_hidden", None),
    ("decrease_steps", 200),
    ("eval", False),
    ("amp", True),
    ("gradient_accumulation_steps", 1),
    ("max_grad_norm", 1.0),
    ("save_freq", None),
    ("loss", "bce"),
])

def methylbert_finetune_collator(features):
    """
    Convert a list of items (dicts) from MethylBertFinetuneDataset 
    into a single batch dict.
    """
    # Each `features[i]` has { "dna_seq", "methyl_seq", "ctype_label", "dmr_label", ... }
    dna_seq = [f["dna_seq"] for f in features]         # List[Tensor]
    methyl_seq = [f["methyl_seq"] for f in features]   # List[Tensor]
    ctype_label = [f["ctype_label"] for f in features]
    dmr_label = [f["dmr_label"] for f in features]

    # pad into a single tensor if needed:
    batch_dna_seq = torch.stack(dna_seq, dim=0)        # shape [batch_size, seq_len]
    batch_methyl_seq = torch.stack(methyl_seq, dim=0)  # shape [batch_size, seq_len]
    batch_ctype_label = torch.tensor(ctype_label, dtype=torch.long)
    batch_dmr_label = torch.tensor(dmr_label, dtype=torch.long)

    # Return a dict matching your model.forward(...) signature
    # "input_ids" -> dna_seq
    # "token_type_ids" -> methyl_seq
    # "ctype_label" -> ctype_label
    # "labels" -> dmr_label  (the MethylBert model expects `labels` to be the DMR label)
    return {
        "input_ids": batch_dna_seq,
        "token_type_ids": batch_methyl_seq,
        "labels": batch_ctype_label,
        "dmr_ids": batch_dmr_label,
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
    	

class MethylBertEmbeddedDMR(BertPreTrainedModel):
    pretrained_model_archive_map = METHYLBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "methylbert"

    def __init__(self, config, seq_len=150):
        # from pretrained - calls the init
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_dmr_labels = config.num_dmr_labels
        if config.loss not in ["bce", "focal_bce"]:
            raise ValueError(f"loss must be bce or focal_bce. {config.loss} is given.")

        self.loss = config.loss
        self.classification_loss_fct = self._setup_loss(self.loss)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.read_classifier = nn.Sequential(
            nn.Linear((config.hidden_size+1)*(seq_len+1), seq_len+1),
            nn.Dropout(0.05),#config.hidden_dropout_prob),
            nn.ReLU(),
            nn.LayerNorm(seq_len+1, eps=config.layer_norm_eps),
            nn.Linear(seq_len+1, 2)
        )

        self.seq_len = seq_len

        self.dmr_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_dmr_labels, embedding_dim = seq_len+1),
        )
        
        self.init_weights()

    def _setup_loss(self, loss):
        if loss == "bce":
            print("Cross entropy loss assigned")
            return nn.CrossEntropyLoss() # this function requires unnormalised logits
        elif loss == "focal_bce":
            print("Focal loss assigned")
            return FocalLoss()
    
    def __str__(self):
        str(self.model.__str__())

    def check_model_status(self):
        print("Bert model training mode : %s"%(self.bert.training))
        print("Dropout training mode : %s"%(self.dropout.training))
        print("Read classifier training mode : %s"%(self.read_classifier.training))
        
    def from_pretrained_read_classifier(self, pretrained_model_name_or_path, device="cpu"):
        self.read_classifier.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def from_pretrained_dmr_encoder(self, pretrained_model_name_or_path, device="cpu"):
        self.dmr_encoder.load_state_dict(torch.load(pretrained_model_name_or_path, map_location=device))
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,  #'methyl_seq' as token_type_ids
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, # Cell Type labels
        dmr_ids=None # DMR labels
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

        # DMR embedding
        dmr_embedding = self.dmr_encoder(dmr_ids.view(-1))  # [batch, seq_len+1]
        # Append along last dimension
        # shape -> [batch, seq_len, hidden_size+1]
        sequence_output = torch.cat((sequence_output, dmr_embedding.unsqueeze(-1)), dim=-1)

        # Flatten for classifier
        batch_size = sequence_output.size(0)
        flat_seq = sequence_output.view(batch_size, -1)  
        ctype_logits = self.read_classifier(flat_seq)  # shape [batch, 2]

        loss = None
        if labels is not None:
            ctype_label_onehot = F.one_hot(labels, num_classes=2).float()
            loss = self.classification_loss_fct(ctype_logits, ctype_label_onehot)

        # In orginal code:
        # ctype_logits = ctype_logits.softmax(dim=1)
        # We removed it here for numeric stability and compatibility with HF trainer

        # Return your extra dmr “logits” or embeddings
        return MethylBertOutput(
            loss=loss,
            logits=ctype_logits,          # The main classification logits
            dmr_logits=sequence_output,   # The appended [batch, seq_len, hidden_size+1] if you want it
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Example placeholders for your custom data collator, metrics, etc.
def methylbert_data_collator(features):
    # Return a batch in the format your model expects
    # e.g. input_ids, attention_mask, token_type_ids, labels, ctype_label
    # ...
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "token_type_ids": torch.stack([f["methyl_seq"] for f in features]),
        "labels": torch.stack([f["dmr_label"] for f in features]),
        "ctype_label": torch.stack([f["ctype_label"] for f in features]),
    }

class MethylBert:
    """
    A one-stop class for:
      - Initializing the custom MethylBertEmbeddedDMR model
      - (Optionally) loading pretrained/fine-tuned weights
      - Fine-tuning
      - Prediction
      - etc.
    """
    def __init__(
        self,
        foundation_model_path: str,
        seq_len: int = 150,
        custom_config: OrderedDict = None,
        load_weights: bool = True,
        fine_tuned_model_path: Optional[str] = None,
        num_labels: int = 2,
        num_dmr_labels:int = 100
    ):
        """
        :param foundation_model_path: local or HF repo ID for the base BERT config/weights
        :param seq_len: maximum sequence length your model uses
        :param custom_config: an OrderedDict of hyperparams (like your default_config).
        :param load_weights: whether to load pretrained weights from 'foundation_model_path'
        :param fine_tuned_model_path: if present, load from that checkpoint
        :param num_labels: e.g. 2 for ctype_label classification
        """
        # 1) Store or build config dictionary
        if custom_config is None:
            # fallback to default_config if you prefer
            # or raise an error
            raise ValueError("Must provide a custom_config dictionary.")
        self._config = custom_config

        # 2) Load the base BERT config
        if os.path.isdir(foundation_model_path):
            # local directory with config.json
            config = BertConfig.from_pretrained(foundation_model_path)
        else:
            # HF model hub reference
            config = BertConfig.from_pretrained(foundation_model_path)

        config.num_labels = num_labels
        config.num_dmr_labels = num_dmr_labels
        config.loss = self._config["loss"]  # "bce" or "focal_bce"
        # (optionally store seq_len if your model constructor needs it)
        self.seq_len = seq_len

        # 3) Build your custom MethylBertEmbeddedDMR
        if not load_weights and fine_tuned_model_path is None:
            # from_config only
            self.model = MethylBertEmbeddedDMR(config, seq_len=seq_len)
        else:
            # load from foundation model path or local dir
            self.model = MethylBertEmbeddedDMR.from_pretrained(
                foundation_model_path,
                config=config,
                seq_len=seq_len,
            )

        # 4) If you have a fine-tuned checkpoint, load it
        if fine_tuned_model_path is not None:
            print(f"Loading fine-tuned checkpoint from {fine_tuned_model_path}")
            self.model = MethylBertEmbeddedDMR.from_pretrained(
                fine_tuned_model_path,
                config=config,
                seq_len=seq_len,
            )

        # 5) Tokenizer (optional)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(foundation_model_path)
        except:
            self.tokenizer = None

        # 6) Create default TrainingArguments from your config
        #    We feed in your hyperparams below:
        default_training_args = TrainingArguments(
            output_dir="output/methylbert_default",
            # HPC config from your dictionary:
            learning_rate=self._config["lr"],
            warmup_steps=self._config["warmup_step"],
            weight_decay=self._config["weight_decay"],
            adam_beta1=self._config["beta"][0],
            adam_beta2=self._config["beta"][1],
            adam_epsilon=self._config["eps"],
            fp16=self._config["amp"],  # automatic mixed precision
            max_grad_norm=self._config["max_grad_norm"],
            gradient_accumulation_steps=self._config["gradient_accumulation_steps"],

            # Logging & saving frequency from your config:
            logging_steps=self._config["log_freq"],
            eval_steps=self._config["eval_freq"],
            save_steps=self._config["eval_freq"],  # or some multiple
            # Other defaults
            per_device_train_batch_size=750,
            per_device_eval_batch_size=375,
            num_train_epochs=150,   # you can override later
            evaluation_strategy="steps",  # Evaluate every X steps
            remove_unused_columns=False,
            eval_accumulation_steps = 8,
            torch_empty_cache_steps = 10,
            prediction_loss_only=False,
            gradient_checkpointing=True,
            skip_memory_metrics=True,
            auto_find_batch_size=False,
            save_total_limit=10,
            # label_names=["ctype_label"]
        )

        self.training_args = default_training_args
        self.hf_config = config  # store HF config
        self.trainer = None  # will be created in _init_trainer

    def _init_trainer(self, 
                      train_dataset=None, 
                      eval_dataset=None,
                      data_collator=None,
                      custom_training_args=None):
        """
        Internal method to build a HF Trainer.
        """
        if custom_training_args is not None:
            self.training_args = custom_training_args

        # fallback to a user-provided or default data_collator
        if data_collator is None:
            # default to a finetune data collator, or pretrain, or ...
            data_collator = methylbert_finetune_collator  # default is one for fine-tuning

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics
        )
        return trainer

    def fine_tune(
        self,
        data_path,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        data_collator=None,
        training_args=None
    ):
        """
        Fine-tune your model on a training set, optional validation set, etc.
        """
        assert data_path or (train_dataset and val_dataset and test_dataset), (
            "Either 'data_path' must be provided or all of 'train_dataset', 'val_dataset', and 'test_dataset' must not be None."
        )

        train_dataset = train_dataset or MethylBertFinetuneDataset(
              f_path=os.path.join(data_path, "train.txt"),
              vocab=MethylVocab(k=3),
              seq_len=self.seq_len
              )
        val_dataset = val_dataset or MethylBertFinetuneDataset(
              f_path=os.path.join(data_path, "dev.txt"),
              vocab=MethylVocab(k=3),
              seq_len=self.seq_len
              )
        test_dataset = test_dataset or MethylBertFinetuneDataset(
              f_path=os.path.join(data_path, "test.txt"),
              vocab=MethylVocab(k=3),
              seq_len=self.seq_len
              )
        self.trainer = self._init_trainer(
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            custom_training_args=training_args
        )

        self.trainer.train()

        if test_dataset:
            results = self.trainer.evaluate(test_dataset)
            print("Test results:", results)

        # Save final model
        if self.trainer.args.should_save:
            self.trainer.save_model(self.trainer.args.output_dir)
            self.safe_save_model_for_hf_trainer(self.trainer.args.output_dir)

    def predict(self, dataset, data_collator=None):
        """
        Use Hugging Face Trainer for prediction on a dataset.
        """
        if self.trainer is None:
            # build a trainer for inference
            self.trainer = self._init_trainer(
                data_collator=data_collator,
                # no train or val dataset
            )
        predictions = self.trainer.predict(dataset)
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


# def get_dna_seq(tokens, tokenizer):
# 	# Convert n-mers tokens into a DNA sequence
# 	seq = tokenizer.from_seq(tokens)
# 	seq = [s for s in seq if "<" not in s]
	
# 	seq = seq[0][0] + "".join([s[1] for s in seq]) + seq[-1][-1]
	
# 	return seq

# def set_seed(seed: int):
# 	"""
# 	Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
# 	installed).

# 	Args:
# 		seed (:obj:`int`): The seed to set.
# 	"""
# 	random.seed(seed)
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed)

from torch.utils.data import Dataset
import torch, gc

import numpy as np
from copy import deepcopy
import multiprocessing as mp
from functools import partial
import random
import pandas as pd

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
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

def _line2tokens_pretrain(l, tokenizer, max_len=120):
	'''
		convert a text line into a list of tokens converted by tokenizer 
			
	'''

	l = l.strip().split(" ")

	tokened = [tokenizer.to_seq(b) for b in l]
	if len(tokened) > max_len:
		return tokened[:max_len]
	else:
		return tokened + [[tokenizer.pad_index] for k in range(max_len-len(tokened))]


def _line2tokens_finetune(l, tokenizer, max_len=150, headers=None):
	# Check the header
	if not all([h in headers for h in ["dna_seq", "methyl_seq", "ctype", "dmr_ctype", "dmr_label"]]):
		raise ValueError("The header must contain dna_seq, methyl_seq, ctype, dmr_ctype, dmr_label")

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

	# Cell-type label is binary (whether the cell type corresponds to the DMR cell type)
	l["ctype_label"] = int(l["ctype"] == l["dmr_ctype"]) 
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
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, random_len=False, n_cores=50):

		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path
		self.random_len = random_len

		# Define a range of tokens to mask based on k-mers 
		self.mask_list = self._get_mask()

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			print("Open data : %s"%f_input)
			raw_seqs = f_input.read().splitlines()

		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			line_labels = pool.map(partial(_line2tokens_pretrain, 
								           tokenizer=self.vocab, 
								           max_len=self.seq_len), raw_seqs)
			del raw_seqs
			print("Lines are processed")
			self.lines = torch.squeeze(torch.tensor(np.array(line_labels, dtype=np.int16)))
		del line_labels
		gc.collect()

	def __getitem__(self, index): 

		dna_seq = self.lines[index].clone()

		# Random len
		if self.random_len and np.random.random() < 0.5:
			dna_seq = dna_seq[:random.randint(5, self.seq_len)] 
		
		# Padding
		if dna_seq.shape[0] < self.seq_len:
			pad_num = self.seq_len-dna_seq.shape[0]
			dna_seq = torch.cat((dna_seq, 
								torch.tensor([self.vocab.pad_index for i in range(pad_num)], dtype=torch.int16)))

		# Mask 
		masked_dna_seq, dna_seq, bert_mask = self._masking(dna_seq)
		#print(dna_seq, masked_dna_seq,"\n=============================================\n")
		return {"bert_input": masked_dna_seq,
				"bert_label": dna_seq,
				"bert_mask" : bert_mask}
	
	def subset_data(self, n_seq: int):
		self.lines = random.sample(self.lines, n_seq)

	def _get_mask(self):
		'''
			Relative positions from the centre of masked region 
			e.g) [-1, 0, 1] for 3-mers 
		'''
		half_length = int(self.vocab.kmers/2)
		mask_list = [-1*half_length + i for i in range(half_length)] + [i for i in range(1, half_length+1)]
		if self.vocab.kmers % 2 == 0:
			mask_list = mask_list[:-1]

		return mask_list

	def _masking(self, inputs: torch.Tensor, threshold=0.15):
		""" 
			Moidfied version of masking token function
			Originally developed by Huggingface (datacollator) and DNABERT
			
			https://github.com/huggingface/transformers/blob/9a24b97b7f304fa1ceaaeba031241293921b69d3/src/transformers/data/data_collator.py#L747

			https://github.com/jerryji1993/DNABERT/blob/bed72fc0694a7b04f7e980dc9ce986e2bb785090/examples/run_pretrain.py#L251

			Added additional tasks to handle each sequence
			Lines using tokenizer were modified due to different tokenizer object structure

		"""

		labels = inputs.clone()

		# Sample tokens with given probability threshold
		probability_matrix = torch.full(labels.shape, threshold) # tensor filled with 0.15

		# Handle special tokens and padding
		special_tokens_mask = [
			val < 5 for val in labels.tolist()
		]
		probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
		#padding_mask = labels.eq(self.vocab.pad_index)
		#probability_matrix.masked_fill_(padding_mask, value=0.0)

		masked_indices = torch.bernoulli(probability_matrix).bool() # get masked tokens based on bernoulli only within non-special tokens		

		# change masked indices
		masked_index = deepcopy(masked_indices)
		
		# This function handles each sequence
		end = torch.where(probability_matrix!=0)[0].tolist()[-1] # end of the sequence
		mask_centers = set(torch.where(masked_index==1)[0].tolist()) # mask locations

		new_centers = deepcopy(mask_centers)
		for center in mask_centers:
			for mask_number in self.mask_list:# add neighbour loci 
				current_index = center + mask_number 
				if current_index <= end and current_index >= 0:
					new_centers.add(current_index)

		new_centers = list(new_centers)
		
		masked_indices[new_centers] = True
		
		# Avoid loss calculation on unmasked tokens
		labels[~masked_indices] = -100 

		# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
		indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
		inputs[indices_replaced] = self.vocab.mask_index

		# 10% of the time, we replace masked input tokens with random word
		indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
		random_words = torch.randint(len(self.vocab), labels.shape, dtype=torch.int16)
		inputs[indices_random] = random_words[indices_random]

		# The rest of the time (10% of the time) we keep the masked input tokens unchanged

		# Special tokens (SOS, EOS)
		if end < inputs.shape[0]:
			inputs[end] = self.vocab.eos_index
		else:
			inputs[-1] = self.vocab.eos_index

		labels = torch.cat((torch.tensor([-100]), labels))
		inputs = torch.cat((torch.tensor([self.vocab.sos_index]), inputs))
		masked_index = torch.cat((torch.tensor([False]), masked_index))


		return inputs, labels, masked_index

class MethylBertFinetuneDataset(MethylBertDataset):
	def __init__(self, f_path: str, vocab: MethylVocab, seq_len: int, n_cores: int=10, n_seqs = None):
		'''
		MethylBERT dataset

		f_path: str
			File path to the processed input file
		vocab: MethylVocab
			MethylVocab object to convert DNA and methylation pattern sequences
		seq_len: int
			Length for the processed sequences
		n_cores: int
			Number of cores for multiprocessing
		n_seqs: int
			Number of sequences to subset the input (default: None, do not make a subset)

		'''
		self.vocab = vocab
		self.seq_len = seq_len
		self.f_path = f_path

		# Read all text files and convert the raw sequence into tokens
		with open(self.f_path, "r") as f_input:
			raw_seqs = f_input.read().splitlines()

		# Check if there's a header 
		headers = raw_seqs[0].split("\t")
		raw_seqs = raw_seqs[1:]

		if n_seqs is not None:
			raw_seqs = raw_seqs[:n_seqs]
		print("Total number of sequences : ", len(raw_seqs))

		# Multiprocessing for the sequence tokenisation
		with mp.Pool(n_cores) as pool:
			self.lines = pool.map(partial(_line2tokens_finetune, 
								   tokenizer=self.vocab, max_len=self.seq_len, headers=headers), raw_seqs)
			del raw_seqs
		gc.collect()
		self.set_dmr_labels = set([l["dmr_label"] for l in self.lines])

		self.ctype_label_count = self._get_cls_num()
		print("# of reads in each label: ", self.ctype_label_count)
		
		
	def _get_cls_num(self):
		# unique labels
		ctype_labels=[l["ctype_label"] for l in self.lines]
		labels = list(set(ctype_labels))
		label_count = np.zeros(len(labels))
		for l in labels:
			label_count[l] = sum(np.array(ctype_labels) == l)
		return label_count

	def num_dmrs(self):
		return max(len(self.set_dmr_labels), max(self.set_dmr_labels)+1) # +1 is for the label 0
	
	def subset_data(self, n_seq):
		self.lines = self.lines[:n_seq]

	def __getitem__(self, index): 

		item = deepcopy(self.lines[index])
		item["dna_seq"] = torch.squeeze(torch.tensor(np.array(item["dna_seq"], dtype=np.int32)))
		item["methyl_seq"] = torch.squeeze(torch.tensor(np.array(item["methyl_seq"], dtype=np.int8)))
		
		# Special tokens (SOS, EOS)
		end = torch.where(item["dna_seq"]!=self.vocab.pad_index)[0].tolist()[-1] + 1 # end of the read
		if end < item["dna_seq"].shape[0]:
			item["dna_seq"][end] = self.vocab.eos_index
			item["methyl_seq"][end] = 2
		else:
			item["dna_seq"][-1] = self.vocab.eos_index
			item["methyl_seq"][-1] = 2
		item["dna_seq"] = torch.cat((torch.tensor([self.vocab.sos_index]), item["dna_seq"]))
		item["methyl_seq"] = torch.cat((torch.tensor([2]), item["methyl_seq"]))
		return item
	
