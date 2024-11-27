import unittest

from methyldl.data.dataset import SupervisedDataset, generate_example_data
from methyldl.modelling.dnabert2 import EpigenDnabert2, TrainingArguments
from unittest.mock import patch, MagicMock
import tempfile


class TestEpigenDnabert2Predict(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.default_model_path = None
        self.default_max_sequence_length = 2**12
        self.default_use_m6a_methylation = True

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
    
    def test_predict_with_small_data(self):
        # Test with small dataset
        model = EpigenDnabert2(
            fine_tuned_model_path=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=self.default_use_m6a_methylation
        )
        
        synthetic_data = self.create_synthetic_data(128, 10, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)

        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 10)  # Ensure all samples are predicted


    def test_predict_with_long_sequence(self):
        # Test with large dataset
        model = EpigenDnabert2(
            fine_tuned_model_path=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=self.default_use_m6a_methylation
        )

        synthetic_data = self.create_synthetic_data(2048, 1, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)

        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)  # Ensure all samples are predicted

    def test_predict_without_m6a_methylation(self):
        # Test without m6a methylation
        model = EpigenDnabert2(
            fine_tuned_model_path=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=False
        )

        synthetic_data = self.create_synthetic_data(128, 1, True, False, False)
        dataset = self.create_dataset(model, synthetic_data)

        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)  # Ensure all samples are predicted

    def test_predict_with_labels_in_data(self):
        # Test with labels in data (edge case)
        model = EpigenDnabert2(
            fine_tuned_model_path=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=self.default_use_m6a_methylation
        )

        synthetic_data = self.create_synthetic_data(128, 1, True, True, True)
        dataset = self.create_dataset(model, synthetic_data)

        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)  # Ensure predictions work with labeled data



class TestEpigenDnabert2FineTune(unittest.TestCase):
    def test_fine_tune(
        self, 
    ):
        # Generate example data using generate_example_data
        data = generate_example_data(sequence_length=150, num_samples=10, include_cpg_methylation=True, include_labels=True)
        # Create temp directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize model
            model = EpigenDnabert2(
                load_weights=False,
                max_sequence_length=150,
                num_labels=2,
                use_cpg_methylation=True,
                use_m6a_methylation=False,
            )

            data_prepared = SupervisedDataset(data, tokenizer=model.tokenizer)

            # Define custom training arguments
            training_args = TrainingArguments(
                run_name="test_run",
                per_device_train_batch_size=10,
                per_device_eval_batch_size=10,
                num_train_epochs=1,
                output_dir=temp_dir,
                evaluation_strategy="steps",
                eval_steps=1,
                save_steps=1,
                logging_steps=1,
                save_total_limit=1,
                overwrite_output_dir=True,
                save_model=False,  # Avoid saving to disk
                eval_and_save_results=True,
            )

            # Call fine_tune with the mocked datasets and training arguments
            model.fine_tune(
                training_args=training_args,
                test_dataset=data_prepared,
                val_dataset=data_prepared,
                train_dataset=data_prepared
            )

  