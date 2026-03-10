import unittest
import os

import torch

from methyldl.data.dataset import SupervisedDataset, generate_example_data
from methyldl.modelling.dnabert2 import EpigenDnabert2, TrainingArguments
from unittest.mock import patch, MagicMock
import tempfile
from parameterized import parameterized

def _has_triton_capable_gpu():
    """Check if CUDA is available and GPU has compute capability >= 8.0 (Ampere+)."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8

class TestEpigenDnabert2Predict(unittest.TestCase):
    def setUp(self):
        self.default_model_path = "foundationalModels/DNABERT-2-117M"
        self.default_max_sequence_length = 2**12
        self.default_use_m6a_methylation = True
    
    def create_model(self, num_labels):
        default_training_args =  TrainingArguments(
            run_name = "dnabert2_default",
            gradient_accumulation_steps = 1,
            learning_rate = 3e-5,
            fp16 = True,
            save_steps = 10,
            output_dir ="../test_container_tmp/dnabert2_default",
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
            auto_find_batch_size=True,
            )
        return EpigenDnabert2(
            foundation_model_huggingface=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=self.default_use_m6a_methylation,
            num_labels=num_labels,
            training_args=default_training_args,
            use_triton=_has_triton_capable_gpu()
        )
    
    def create_synthetic_data(self, sequence_length, num_samples, include_cpg, include_m6a, include_labels):
        return generate_example_data(
            sequence_length=sequence_length,
            include_cpg_methylation=include_cpg,
            include_m6a_methylation=include_m6a,
            include_labels=include_labels,
            num_samples=num_samples
        )
    
    def create_dataset(self, model, synthetic_data):
        return SupervisedDataset(
            tokenizer=model.tokenizer,
            data_path_or_list=synthetic_data,
            kmer=-1
        )
    
    @parameterized.expand([
        ("binary", 2),
        ("multiclass", 5),
    ])
    def test_predict_with_small_data(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 10, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 10)
    
    @parameterized.expand([
        ("binary", 2),
        ("multiclass", 5),
    ])
    def test_predict_with_long_sequence(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(2048, 1, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)
    
    @parameterized.expand([
        ("binary", 2),
        ("multiclass", 5),
    ])
    def test_predict_without_m6a_methylation(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 1, True, False, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)
    
    @parameterized.expand([
        ("binary", 2),
        ("multiclass", 5),
    ])
    def test_predict_with_labels_in_data(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 1, True, True, True)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)



class TestEpigenDnabert2FineTune(unittest.TestCase):
    def setUp(self):
        self.default_model_path = "foundationalModels/DNABERT-2-117M"
    def _run_fine_tune_test(self, use_triton):
        """Helper method containing the actual test logic."""
        data = generate_example_data(
            sequence_length=150, 
            num_samples=10, 
            include_cpg_methylation=True, 
            include_labels=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model = EpigenDnabert2(
                foundation_model_huggingface=self.default_model_path,
                fine_tuned_model_path=None,
                load_weights=False,
                max_sequence_length=150,
                num_labels=2,
                use_cpg_methylation=True,
                use_m6a_methylation=False,
                use_triton=use_triton
            )
            
            data_prepared = SupervisedDataset(data, tokenizer=model.tokenizer)
            
            training_args = TrainingArguments(
                run_name=f"test_run_triton_{use_triton}",
                per_device_train_batch_size=10,
                per_device_eval_batch_size=10,
                num_train_epochs=1,
                output_dir=temp_dir,
                eval_strategy="steps",
                eval_steps=1,
                save_steps=1,
                logging_steps=1,
                save_total_limit=1,
                overwrite_output_dir=True,
                save_model=False,
                eval_and_save_results=True,
                report_to=[]
            )
            
            model.fine_tune(
                training_args=training_args,
                test_dataset=data_prepared,
                val_dataset=data_prepared,
                train_dataset=data_prepared
            )
    
    def test_fine_tune_default_attention(self):
        """Test with use_triton=False."""
        self._run_fine_tune_test(use_triton=False)
    
    @unittest.skipUnless(
        _has_triton_capable_gpu(), 
        "Triton flash attention requires GPU with compute capability >= 8.0"
    )
    def test_fine_tune_with_triton(self):
        """Test with use_triton=True."""
        self._run_fine_tune_test(use_triton=True)

if __name__ == '__main__':
    unittest.main()