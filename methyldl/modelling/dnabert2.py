import os
import json
import gc
from dataclasses import dataclass, field
from typing import  Optional, Dict, Tuple, List, Union
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import (SequenceClassifierOutput)

from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import (
    TrainerCallback
)
import torch
import torch.nn as nn
from typing import Optional, Callable

import torch
import transformers
from torch.utils.data import Dataset

from methyldl.modelling.evaluation import compute_metrics,preprocess_logits_for_prediction,keep_logits_only
from methyldl.modelling.utils import calculate_batch_size
from methyldl.data.dataset import *
from safetensors.torch import load_file
from transformers.models.bert.configuration_bert import BertConfig
from methyldl.modelling.common import DMRAttentionClassifier


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
        self.register_buffer('token_type_ids',
                             torch.zeros(model.config.max_position_embeddings,
                                         dtype=torch.long),
                             persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cpg_methylation: Optional[torch.LongTensor] = None,
        m6a_methylation: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or inputs_embeds!')
        
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
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                token_type_ids = buffered_token_type_ids.expand(input_shape[0], seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.word_embeddings.device)

        # Compute the embeddings for the main input
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # Compute methylation embeddings if cpg_methylation are provided
        if self.use_cpg_methylation and cpg_methylation is not None:
            cpg_methylation_embeddings = self.cpg_methylation_embeddings(cpg_methylation)
            embeddings += cpg_methylation_embeddings

        if self.use_m6a_methylation and m6a_methylation is not None:
            m6a_methylation_embeddings = self.m6a_methylation_embeddings(m6a_methylation)
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
        **kwargs
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids,
                                           position_ids,cpg_methylation, m6a_methylation) #TODO remove position ids as not needed 

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
            subset_mask=subset_mask)

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[-1]
            pooled_output = self.pooler(
                sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask.bool()
            subset_idx = subset_mask[attention_mask_bool]  # type: ignore
            sequence_output = encoder_outputs[-1][
                masked_tokens_mask[attention_mask_bool][subset_idx]]
            if self.pooler is not None:
                pool_input = encoder_outputs[-1][
                    first_col_mask[attention_mask_bool][subset_idx]]
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

    def __init__(self, prertained_model, num_labels=None, num_dmr_labels=None):
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

        self.bert = BertModel(prertained_model.bert) # Reconstructing original model 
        self.dropout = prertained_model.dropout
        self.num_dmr_labels = num_dmr_labels
        if num_dmr_labels is None:
            self.classifier = prertained_model.classifier
        else:
            self.classifier = DMRAttentionClassifier(prertained_model.config)

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
        dmr_ids: Optional[torch.Tensor] =None  # DMR labels
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        # Labels for computing the sequence classification/regression loss.
        # Indices should be in `[0, ..., config.num_labels - 1]`.
        # If `config.num_labels == 1` a regression loss is computed
        # (mean-square loss). If `config.num_labels > 1` a classification loss
        # is computed (cross-entropy).

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            cpg_methylation = cpg_methylation,
            m6a_methylation = m6a_methylation,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        if self.num_dmr_labels is None:
            logits = self.classifier(pooled_output)
        else:
            logits,_ = self.classifier(
                pooled_output, dmr_ids, attention_mask
            )

        loss = None
        if labels is not None:
            # Compute loss
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or
                                              labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'

            if self.config.problem_type == 'regression':
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
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
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps"),
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

class EpigenDnabert2():
    def __init__(self, 
                  foundation_model_huggingface:str = "zhihan1996/DNABERT-2-117M", 
                  load_weights = True,
                  fine_tuned_model_path: Optional[str] = None,
                  max_sequence_length: int = 150,
                  num_labels:int =2,
                  use_cpg_methylation=True, 
                  use_m6a_methylation=False,
                  trust_remote_code=True,
                  use_triton=True,
                  training_args=None,
                  num_dmr_labels=None):
        
        assert len(foundation_model_huggingface), "Must specify foundation model path hosted on Hugging Face"
        self.num_dmr_labels = num_dmr_labels

        # Helper method to initialize and customize the model
        def initialize_model_with_custom_embeddings(base_model, use_cpg, use_m6a, num_labels=None):
            base_model.bert.embeddings = BertEmbeddings(base_model.bert, use_cpg, use_m6a)
            return BertForSequenceClassification(base_model, num_labels=num_labels,num_dmr_labels=num_dmr_labels)
        
        config = BertForSequenceClassification.config_class.from_pretrained(foundation_model_huggingface)
        config = BertConfig(**config.to_dict(),use_triton=use_triton)
        # Does not load weights just yet, because if we have a checkpoint, the weights will be retrived from it 
        # base_model = transformers.AutoModelForSequenceClassification.from_config(trust_remote_code=trust_remote_code, config = config)
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                foundation_model_huggingface,
                trust_remote_code=trust_remote_code,
                config = config,
                local_files_only=True,  
                cache_dir=None)  
        model = initialize_model_with_custom_embeddings(base_model, use_cpg_methylation, use_m6a_methylation,num_labels=num_labels)
        model.classifier = nn.Linear(768,out_features=num_labels,bias=True)
        self.num_labels=num_labels
        model.num_labels = num_labels
        if fine_tuned_model_path is not None:
            checkpoint = load_file(fine_tuned_model_path)
            # TODO: Remove this part after proper checkpoint is generated:
            old_cpg_methylation_key = 'bert.embeddings.methylation_embeddings.weight'
            if old_cpg_methylation_key in checkpoint.keys():
                checkpoint['bert.embeddings.cpg_methylation_embeddings.weight'] = checkpoint.pop(old_cpg_methylation_key)
            
            model.load_state_dict(checkpoint)

            model.eval()
        elif load_weights:
            base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                foundation_model_huggingface,
                trust_remote_code=trust_remote_code,
                config = config)
            model = initialize_model_with_custom_embeddings(base_model, use_cpg_methylation, use_m6a_methylation,num_labels=num_labels)
            model.classifier = nn.Linear(768,out_features=num_labels,bias=True)
        self.model = model
        self.num_labels = num_labels
        self.config = config
        model_max_length = round(max_sequence_length//4+1) # BPE encoding reduces sequence length approximately by a factor of 4

        recomended_batch_size = calculate_batch_size(gb_per_seq = 0.029, cpu_batch_size=300)

        default_training_args =  TrainingArguments(
            run_name = "dnabert2_default",
            per_device_train_batch_size = recomended_batch_size,
            per_device_eval_batch_size = int(recomended_batch_size/2),
            gradient_accumulation_steps = 1,
            learning_rate = 3e-5,
            fp16 = True,
            save_steps = 10,
            output_dir ="output/dnabert2_default",
            eval_strategy = "steps",
            eval_steps = 10, 
            warmup_steps = 100, 
            logging_steps = 100, 
            num_train_epochs = 250, 
            overwrite_output_dir = True, 
            log_level = "info",
            find_unused_parameters = False,
            batch_eval_metrics = False,
            eval_and_save_results = True,
            remove_unused_columns=False,
            eval_accumulation_steps = 8,
            torch_empty_cache_steps = 10,
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
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)
        self.trainer = self._init_trainer() # default trainer to use for predictions 
    
    def __str__(self):
        str(self.model.__str__())
    
    def _init_trainer(self,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)):
        if self.num_labels==2:
            return transformers.Trainer(model=self.model,
                                args = args,
                                data_collator=self.data_collator,
                                train_dataset = train_dataset,
                                eval_dataset = eval_dataset,
                                model_init = model_init,
                                callbacks = callbacks,
                                optimizers = optimizers,
                                tokenizer=self.tokenizer,
                                preprocess_logits_for_metrics=preprocess_logits_for_prediction,
                                compute_metrics=compute_metrics)
        elif self.num_labels>2:
            return transformers.Trainer(model=self.model,
                                args = args,
                                data_collator=self.data_collator,
                                train_dataset = train_dataset,
                                eval_dataset = eval_dataset,
                                model_init = model_init,
                                callbacks = callbacks,
                                optimizers = optimizers,
                                tokenizer=self.tokenizer)

    def predict(self, test_dataset,batch_size = None, clear_cache=True):
        if batch_size is not None:
            training_args = TrainingArguments(
                    eval_strategy = "no",
                    save_strategy = "no",
                    gradient_checkpointing=False,
                    skip_memory_metrics=True,
                    auto_find_batch_size=False,
                    per_device_eval_batch_size = batch_size,
                    output_dir=self.training_args.output_dir
                    )
        else:
            # TODO Must be a better way
            # Also, when using Flash Attention, we are forced to have batch size as multiples of 64 to avoid race conditions.
            if self.max_sequence_length <= 1000:
                training_args = TrainingArguments(
                        eval_strategy = "no",
                        save_strategy = "no",
                        gradient_checkpointing=False,
                        skip_memory_metrics=True,
                        auto_find_batch_size=False,
                        per_device_eval_batch_size = 64*6
                        )
            elif self.max_sequence_length <= 2000:
                training_args = TrainingArguments(
                        eval_strategy = "no",
                        save_strategy = "no",
                        gradient_checkpointing=False,
                        skip_memory_metrics=True,
                        auto_find_batch_size=False,
                        per_device_eval_batch_size = 64*3,
                        output_dir=self.training_args.output_dir
                        )
            elif self.max_sequence_length <= 3000:
                training_args = TrainingArguments(
                        eval_strategy = "no",
                        save_strategy = "no",
                        gradient_checkpointing=False,
                        skip_memory_metrics=True,
                        auto_find_batch_size=False,
                        per_device_eval_batch_size = 64,
                        output_dir=self.training_args.output_dir
                        )
            else:
                training_args = TrainingArguments(
                        eval_strategy = "no",
                        save_strategy = "no",
                        gradient_checkpointing=False,
                        skip_memory_metrics=True,
                        auto_find_batch_size=True,
                        output_dir=self.training_args.output_dir
                        )
        if self.num_labels==2:
            prediction_trainer = transformers.Trainer(
                model=self.model,
                args=training_args,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                preprocess_logits_for_metrics=preprocess_logits_for_prediction,
                compute_metrics=None
            )

        
        elif self.num_labels>2:
            prediction_trainer = transformers.Trainer(
                model=self.model,
                args=training_args,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                compute_metrics=None,
                preprocess_logits_for_metrics =  keep_logits_only
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

    def fine_tune(self,
                  data_path: Optional[str] = None,
                  training_args: Union[TrainingArguments, None] = None,
                  train_dataset: Optional[SupervisedDataset] = None,
                  val_dataset: Optional[SupervisedDataset] = None,
                  test_dataset: Optional[SupervisedDataset] = None,
                  callbacks: Optional[List[TrainerCallback]] = None,
                  data_interface: str = "csv",
                  resume_from_checkpoint: Optional[Union[bool, str]] = None):
        
        # Ensure that either data_path is provided or all datasets are provided
        assert data_path or (train_dataset and val_dataset and test_dataset), (
            "Either 'data_path' must be provided or all of 'train_dataset', 'val_dataset', and 'test_dataset' must not be None."
        )

        # TODO Adding LoRA
        print("Starting to initialize datasets")
        train_dataset = train_dataset or SupervisedDataset(tokenizer=self.tokenizer, 
                                        data_path_or_list=os.path.join(data_path, "train"), 
                                        kmer=-1,data_interface=data_interface)
        print("Train is initialized")
        val_dataset = val_dataset or SupervisedDataset(tokenizer=self.tokenizer, 
                                        data_path_or_list=os.path.join(data_path, "valid"), 
                                        kmer=-1,data_interface=data_interface)
        print("Val is initialized")
        
        if training_args is not None:
            self.training_args = training_args # overwritting default training args

        self.trainer = self._init_trainer(train_dataset=train_dataset,
                                     eval_dataset = val_dataset,
                                     args=self.training_args,
                                     callbacks=callbacks)
        
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
        elif hasattr(self, 'resume_from_checkpoint') and self.resume_from_checkpoint:
            # Use checkpoint path from initialization if provided
            checkpoint_path = self.resume_from_checkpoint

        self.trainer.train(resume_from_checkpoint=checkpoint_path)
        if self.training_args.save_model:
            self.trainer.save_state()
            self.safe_save_model_for_hf_trainer(output_dir=training_args.output_dir)