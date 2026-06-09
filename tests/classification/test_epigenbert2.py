import unittest
import os
from types import SimpleNamespace

import torch
import syto.classification.classifiers.dnabert2 as dnabert2_module

from syto.data.dataset import SupervisedDataset, generate_example_data
from syto.classification.classifiers.dnabert2 import EpigenDnabert2, TrainingArguments
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
        default_training_args = TrainingArguments(
            run_name="dnabert2_default",
            gradient_accumulation_steps=1,
            learning_rate=3e-5,
            fp16=True,
            save_steps=10,
            output_dir="../test_container_tmp/dnabert2_default",
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
            auto_find_batch_size=True,
        )
        return EpigenDnabert2(
            foundation_model_huggingface=self.default_model_path,
            max_sequence_length=self.default_max_sequence_length,
            use_m6a_methylation=self.default_use_m6a_methylation,
            num_labels=num_labels,
            training_args=default_training_args,
            use_triton=_has_triton_capable_gpu(),
        )

    def create_synthetic_data(
        self, sequence_length, num_samples, include_cpg, include_m6a, include_labels
    ):
        return generate_example_data(
            sequence_length=sequence_length,
            include_cpg_methylation=include_cpg,
            include_m6a_methylation=include_m6a,
            include_labels=include_labels,
            num_samples=num_samples,
        )

    def create_dataset(self, model, synthetic_data):
        return SupervisedDataset(
            tokenizer=model.tokenizer, data_path_or_list=synthetic_data, kmer=-1
        )

    @parameterized.expand(
        [
            ("binary", 2),
            ("multiclass", 5),
        ]
    )
    def test_predict_with_small_data(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 10, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 10)

    @parameterized.expand(
        [
            ("binary", 2),
            ("multiclass", 5),
        ]
    )
    def test_predict_with_long_sequence(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(2048, 1, True, True, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)

    @parameterized.expand(
        [
            ("binary", 2),
            ("multiclass", 5),
        ]
    )
    def test_predict_without_m6a_methylation(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 1, True, False, False)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)

    @parameterized.expand(
        [
            ("binary", 2),
            ("multiclass", 5),
        ]
    )
    def test_predict_with_labels_in_data(self, name, num_labels):
        model = self.create_model(num_labels)
        synthetic_data = self.create_synthetic_data(128, 1, True, True, True)
        dataset = self.create_dataset(model, synthetic_data)
        predictions = model.predict(dataset)
        self.assertEqual(len(predictions.predictions), 1)


class TestEpigenDnabert2FineTune(unittest.TestCase):
    def setUp(self):
        self.default_model_path = "foundationalModels/DNABERT-2-117M"

    def _build_stub_model(self):
        """Create a lightweight EpigenDnabert2 instance whose trainer lifecycle is fully mocked."""
        model = object.__new__(EpigenDnabert2)
        model._init_trainer = MagicMock(name="_init_trainer")
        model.safe_save_model_for_hf_trainer = MagicMock(
            name="safe_save_model_for_hf_trainer"
        )
        model.training_args = SimpleNamespace(
            save_model=False,
            output_dir="/tmp/default-dnabert2-output",
        )
        return model

    def _run_fine_tune_test(self, use_triton):
        """Helper method containing the actual test logic."""
        data = generate_example_data(
            sequence_length=150,
            num_samples=10,
            include_cpg_methylation=True,
            include_labels=True,
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
                use_triton=use_triton,
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
                report_to=[],
            )

            model.fine_tune(
                training_args=training_args,
                test_dataset=data_prepared,
                val_dataset=data_prepared,
                train_dataset=data_prepared,
            )

    def test_fine_tune_default_attention(self):
        """Test with use_triton=False."""
        self._run_fine_tune_test(use_triton=False)

    @unittest.skipUnless(
        _has_triton_capable_gpu(),
        "Triton flash attention requires GPU with compute capability >= 8.0",
    )
    def test_fine_tune_with_triton(self):
        """Test with use_triton=True."""
        self._run_fine_tune_test(use_triton=True)

    @parameterized.expand(
        [
            ("resume_true", True, None, True),
            ("resume_path", "/tmp/checkpoint-7", None, "/tmp/checkpoint-7"),
            ("resume_attr", None, "/tmp/from-instance", "/tmp/from-instance"),
        ]
    )
    def test_fine_tune_selects_expected_checkpoint_source(
        self,
        name,
        resume_from_checkpoint,
        instance_resume_path,
        expected_checkpoint,
    ):
        """Test that fine_tune forwards the correct checkpoint source to the trainer."""
        model = self._build_stub_model()
        trainer = MagicMock(name="trainer")
        model._init_trainer.return_value = trainer
        if instance_resume_path is not None:
            model.resume_from_checkpoint = instance_resume_path

        training_args = SimpleNamespace(
            save_model=False,
            output_dir="/tmp/branch-selection-output",
        )

        model.fine_tune(
            training_args=training_args,
            train_dataset=object(),
            val_dataset=object(),
            test_dataset=object(),
            resume_from_checkpoint=resume_from_checkpoint,
        )

        trainer.train.assert_called_once_with(
            resume_from_checkpoint=expected_checkpoint
        )
        trainer.save_state.assert_not_called()
        model.safe_save_model_for_hf_trainer.assert_not_called()

    def test_fine_tune_saves_state_and_model_when_requested(self):
        """Test that the save-model branch persists trainer state and forwards the output path."""
        model = self._build_stub_model()
        trainer = MagicMock(name="trainer")
        model._init_trainer.return_value = trainer
        training_args = SimpleNamespace(
            save_model=True,
            output_dir="/tmp/save-model-output",
        )

        model.fine_tune(
            training_args=training_args,
            train_dataset=object(),
            val_dataset=object(),
            test_dataset=object(),
            resume_from_checkpoint=False,
        )

        # `False` intentionally falls through to "start fresh" rather than resuming.
        trainer.train.assert_called_once_with(resume_from_checkpoint=None)
        trainer.save_state.assert_called_once_with()
        model.safe_save_model_for_hf_trainer.assert_called_once_with(
            output_dir=training_args.output_dir
        )


class TestEpigenDnabert2TrainerBranches(unittest.TestCase):
    """Exercise the trainer initialization and prediction branches."""

    def _build_stub_model(
        self,
        num_labels,
        max_sequence_length=128,
        output_dir="/tmp/dnabert2-test-output",
    ):
        """Create a lightweight EpigenDnabert2 instance with only the attributes under test."""
        model = object.__new__(EpigenDnabert2)
        model.num_labels = num_labels
        model.max_sequence_length = max_sequence_length
        model.model = object()
        model.data_collator = object()
        model.tokenizer = object()
        model.training_args = SimpleNamespace(output_dir=output_dir)
        model.trainer = None
        return model

    def test_init_trainer_binary_registers_metrics_hooks(self):
        """Test that binary classification wires prediction preprocessing and metrics."""
        model = self._build_stub_model(num_labels=2)
        args = MagicMock(name="training_args")
        train_dataset = object()
        eval_dataset = object()
        callbacks = [MagicMock()]
        optimizer = MagicMock()
        scheduler = MagicMock()
        model_init = MagicMock()
        trainer_instance = MagicMock()

        with patch(
            "syto.classification.classifiers.dnabert2.transformers.Trainer",
            return_value=trainer_instance,
        ) as mocked_trainer:
            trainer = model._init_trainer(
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                model_init=model_init,
                callbacks=callbacks,
                optimizers=(optimizer, scheduler),
            )

        self.assertIs(trainer, trainer_instance)
        trainer_kwargs = mocked_trainer.call_args.kwargs
        self.assertIs(trainer_kwargs["model"], model.model)
        self.assertIs(trainer_kwargs["args"], args)
        self.assertIs(trainer_kwargs["train_dataset"], train_dataset)
        self.assertIs(trainer_kwargs["eval_dataset"], eval_dataset)
        self.assertIs(trainer_kwargs["model_init"], model_init)
        self.assertEqual(trainer_kwargs["callbacks"], callbacks)
        self.assertEqual(trainer_kwargs["optimizers"], (optimizer, scheduler))
        self.assertIs(
            trainer_kwargs["preprocess_logits_for_metrics"],
            dnabert2_module.preprocess_logits_for_prediction,
        )
        self.assertIs(
            trainer_kwargs["compute_metrics"], dnabert2_module.compute_metrics
        )

    def test_init_trainer_multiclass_omits_binary_only_hooks(self):
        """Test that multiclass trainer initialization skips binary-specific metrics hooks."""
        model = self._build_stub_model(num_labels=4)
        args = MagicMock(name="training_args")
        trainer_instance = MagicMock()

        with patch(
            "syto.classification.classifiers.dnabert2.transformers.Trainer",
            return_value=trainer_instance,
        ) as mocked_trainer:
            trainer = model._init_trainer(args=args)

        self.assertIs(trainer, trainer_instance)
        trainer_kwargs = mocked_trainer.call_args.kwargs
        self.assertIs(trainer_kwargs["model"], model.model)
        self.assertIs(trainer_kwargs["args"], args)
        self.assertNotIn("preprocess_logits_for_metrics", trainer_kwargs)
        self.assertNotIn("compute_metrics", trainer_kwargs)

    def test_predict_uses_explicit_batch_size_and_clears_cache_for_binary_model(self):
        """Test that explicit batch sizes override defaults and still clear caches."""
        model = self._build_stub_model(num_labels=2, max_sequence_length=512)
        dataset = object()
        prediction = MagicMock()
        trainer_instance = MagicMock()
        trainer_instance.predict.return_value = prediction

        with patch(
            "syto.classification.classifiers.dnabert2.transformers.Trainer",
            return_value=trainer_instance,
        ) as mocked_trainer, patch("gc.collect") as mocked_collect, patch(
            "torch.cuda.empty_cache"
        ) as mocked_empty_cache:
            result = model.predict(dataset, batch_size=7, clear_cache=True)

        self.assertIs(result, prediction)
        trainer_instance.predict.assert_called_once_with(dataset)
        trainer_kwargs = mocked_trainer.call_args.kwargs
        trainer_args = trainer_kwargs["args"]
        self.assertEqual(trainer_args.per_device_eval_batch_size, 7)
        self.assertEqual(trainer_args.output_dir, model.training_args.output_dir)
        self.assertIs(trainer_kwargs["compute_metrics"], None)
        self.assertIs(
            trainer_kwargs["preprocess_logits_for_metrics"],
            dnabert2_module.preprocess_logits_for_prediction,
        )
        mocked_collect.assert_called_once()
        mocked_empty_cache.assert_called_once()

    @parameterized.expand(
        [
            ("short_binary", 2, 1000, 64 * 6, False),
            ("medium_multiclass", 4, 2000, 64 * 3, False),
            ("long_binary", 2, 3000, 64, False),
            ("very_long_multiclass", 4, 3001, None, True),
        ]
    )
    def test_predict_automatic_batch_size_paths(
        self,
        name,
        num_labels,
        max_sequence_length,
        expected_eval_batch_size,
        expected_auto_find_batch_size,
    ):
        """Test automatic prediction batch-size selection across all sequence-length branches."""
        model = self._build_stub_model(
            num_labels=num_labels,
            max_sequence_length=max_sequence_length,
        )
        dataset = object()
        prediction = MagicMock()
        trainer_instance = MagicMock()
        trainer_instance.predict.return_value = prediction

        with patch(
            "syto.classification.classifiers.dnabert2.transformers.Trainer",
            return_value=trainer_instance,
        ) as mocked_trainer, patch("gc.collect") as mocked_collect, patch(
            "torch.cuda.empty_cache"
        ) as mocked_empty_cache:
            result = model.predict(dataset, clear_cache=False)

        self.assertIs(result, prediction)
        trainer_instance.predict.assert_called_once_with(dataset)
        trainer_kwargs = mocked_trainer.call_args.kwargs
        trainer_args = trainer_kwargs["args"]

        self.assertEqual(trainer_args.output_dir, model.training_args.output_dir)

        self.assertEqual(
            trainer_args.auto_find_batch_size,
            expected_auto_find_batch_size,
        )
        if expected_eval_batch_size is not None:
            self.assertEqual(
                trainer_args.per_device_eval_batch_size,
                expected_eval_batch_size,
            )

        if num_labels == 2:
            self.assertIs(
                trainer_kwargs["preprocess_logits_for_metrics"],
                dnabert2_module.preprocess_logits_for_prediction,
            )
        else:
            self.assertIs(
                trainer_kwargs["preprocess_logits_for_metrics"],
                dnabert2_module.keep_logits_only,
            )
        self.assertIs(trainer_kwargs["compute_metrics"], None)
        mocked_collect.assert_not_called()
        mocked_empty_cache.assert_not_called()


class _FixedBackbone(torch.nn.Module):
    """Minimal backbone that returns precomputed outputs for forward-path tests."""

    def __init__(self, outputs):
        """Store the tuple that should be returned by each forward call."""
        super().__init__()
        self.outputs = outputs

    def forward(self, *args, **kwargs):
        """Return the stored outputs without performing any computation."""
        return self.outputs


class _FixedClassifier(torch.nn.Module):
    """Minimal classifier head that always emits the provided logits."""

    def __init__(self, logits):
        """Store a deterministic logits tensor for assertion-friendly tests."""
        super().__init__()
        self.logits = logits

    def forward(self, pooled_output):
        """Ignore the pooled input and return the stored logits."""
        return self.logits


class TestBertForSequenceClassificationForward(unittest.TestCase):
    """Exercise small forward branches in BertForSequenceClassification."""

    def _build_stub_model(self, logits, sequence_output, pooled_output, extra_output):
        """Create a lightweight classifier wrapper with deterministic backbone outputs."""
        model = object.__new__(dnabert2_module.BertForSequenceClassification)
        torch.nn.Module.__init__(model)
        model.num_labels = logits.shape[-1]
        model.num_grg_labels = None
        model.config = SimpleNamespace(problem_type=None, use_return_dict=True)
        model.bert = _FixedBackbone((sequence_output, pooled_output, extra_output))
        model.dropout = torch.nn.Identity()
        model.soft_labels = False
        model.classifier = _FixedClassifier(logits)
        return model

    def test_forward_multi_label_without_return_dict_returns_loss_and_tuple(self):
        """Test that float labels trigger BCE loss and tuple output when return_dict is false."""
        logits = torch.tensor([[0.2, -0.1, 0.8]], dtype=torch.float32)
        sequence_output = torch.tensor([[[0.3, 0.1, -0.2]]], dtype=torch.float32)
        pooled_output = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        labels = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        extra_output = ("hidden-marker",)
        model = self._build_stub_model(
            logits=logits,
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            extra_output=extra_output,
        )

        # Float labels force the auto-detection logic onto the BCE multi-label path.
        result = model(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            labels=labels,
            return_dict=False,
        )

        expected_loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        self.assertEqual(model.config.problem_type, "multi_label_classification")
        self.assertEqual(len(result), 3)
        self.assertTrue(torch.isclose(result[0], expected_loss))
        self.assertTrue(torch.equal(result[1], logits))
        # The final tuple element comes from outputs[2:], which verifies the non-dict return path.
        self.assertEqual(result[2], extra_output)


class TestBertForSequenceClassificationSoftLabels(unittest.TestCase):
    """Test soft-label forward path in BertForSequenceClassification."""

    def _build_stub_model(self, logits, sequence_output, pooled_output, extra_output):
        """Create a lightweight classifier wrapper with soft_labels=True."""
        model = object.__new__(dnabert2_module.BertForSequenceClassification)
        torch.nn.Module.__init__(model)
        model.num_labels = logits.shape[-1]
        model.num_grg_labels = None
        model.config = SimpleNamespace(problem_type=None, use_return_dict=True)
        model.bert = _FixedBackbone((sequence_output, pooled_output, extra_output))
        model.dropout = torch.nn.Identity()
        model.soft_labels = True
        model.classifier = _FixedClassifier(logits)
        return model

    def test_forward_soft_labels_uses_cwce_loss(self):
        """When soft_labels=True and labels are float, ConfidenceWeightedCrossEntropy is used."""
        num_classes = 3
        logits = torch.tensor([[0.5, -0.3, 1.0]], dtype=torch.float32)
        sequence_output = torch.randn(1, 3, 3)
        pooled_output = torch.randn(1, 3)
        extra_output = ()

        model = self._build_stub_model(
            logits=logits,
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            extra_output=extra_output,
        )

        # Soft labels (float, sums to 1)
        soft_labels = torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32)

        result = model(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            labels=soft_labels,
        )

        # Should have detected single_label_classification via soft_labels flag
        self.assertEqual(model.config.problem_type, "single_label_classification")
        self.assertIsNotNone(result.loss)
        self.assertFalse(torch.isnan(result.loss))
        self.assertEqual(result.logits.shape, (1, num_classes))

    def test_forward_soft_labels_regression_path(self):
        """When num_labels=1 and soft_labels=True, regression loss is used."""
        logits = torch.tensor([[0.5]], dtype=torch.float32)
        sequence_output = torch.randn(1, 3, 1)
        pooled_output = torch.randn(1, 1)
        extra_output = ()

        model = self._build_stub_model(
            logits=logits,
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            extra_output=extra_output,
        )
        model.num_labels = 1

        labels = torch.tensor([[0.7]], dtype=torch.float32)

        result = model(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            labels=labels,
        )

        self.assertEqual(model.config.problem_type, "regression")
        self.assertIsNotNone(result.loss)


if __name__ == "__main__":
    unittest.main()
