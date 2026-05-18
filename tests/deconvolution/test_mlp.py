"""Tests for methyldl.deconvolution.deep_deconvolvers.mlp.

Covers:
    - _MLPDeconvolverModel  architecture and forward pass
    - MLPDeconvolver        initialization, fit, predict, save/load, CV interface
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from methyldl.deconvolution.deep_deconvolvers.mlp import (
    MLPDeconvolver,
    _MLPDeconvolverModel,
)
from methyldl.deconvolution.deep_deconvolvers.training import TrainingHistory

# ── Shared constants ──────────────────────────────────────────────────────────
N_FEATURES = 12
N_TARGETS = 4
N_TRAIN = 40
N_VAL = 20
N_TEST = 10


# ── Shared helper functions ───────────────────────────────────────────────────


def _make_synthetic_proportions(
    n_samples: int, n_targets: int, seed: int = 42
) -> np.ndarray:
    """Return Dirichlet-sampled proportions (each row sums to 1)."""
    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    return rng.dirichlet(np.ones(n_targets), size=n_samples).astype(np.float32)


def _make_synthetic_data(
    n_features: int = N_FEATURES,
    n_targets: int = N_TARGETS,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
) -> dict:
    """Return a minimal train/val dataset for testing."""
    # pylint: disable=no-member, invalid-name
    rng = np.random.RandomState(0)
    X_train = rng.rand(n_train, n_features).astype(np.float32)
    X_val = rng.rand(n_val, n_features).astype(np.float32)
    y_train = _make_synthetic_proportions(n_train, n_targets, seed=1)
    y_val = _make_synthetic_proportions(n_val, n_targets, seed=2)
    return {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val}


def _fit_minimal_deconvolver(
    n_features: int = N_FEATURES,
    n_targets: int = N_TARGETS,
) -> MLPDeconvolver:
    """Create and fit an MLPDeconvolver using a 2-epoch CPU run."""
    data = _make_synthetic_data(n_features=n_features, n_targets=n_targets)
    deconvolver = MLPDeconvolver(n_input_features=n_features, n_targets=n_targets)
    deconvolver.fit(
        data["X_train"],
        data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        n_epochs=2,
        device="cpu",
        verbose=0,
    )
    return deconvolver


# ═══════════════════════════════════════════════════════════════════════════════
#  _MLPDeconvolverModel
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverModel(unittest.TestCase):
    """Tests for the internal PyTorch model used by MLPDeconvolver."""

    def setUp(self):
        """Instantiate a small model in eval mode for each test."""
        self.n_features = N_FEATURES
        self.n_targets = N_TARGETS
        self.model = _MLPDeconvolverModel(self.n_features, self.n_targets)
        self.model.eval()

    # ── Forward pass ─────────────────────────────────────────────────────────

    def test_forward_output_shape(self):
        """Forward pass should return a tensor of shape (batch_size, n_targets)."""
        batch_size = 8
        x = torch.rand(batch_size, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (batch_size, self.n_targets))

    def test_forward_single_sample(self):
        """Forward should handle a single-sample batch without raising."""
        x = torch.rand(1, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (1, self.n_targets))

    def test_forward_output_is_non_negative(self):
        """All output values should be non-negative (probability constraint)."""
        x = torch.rand(16, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.all(out >= 0.0).item())

    def test_forward_output_sums_to_one(self):
        """Output values should sum to 1 across the target dimension."""
        x = torch.rand(16, self.n_features)
        with torch.no_grad():
            out = self.model(x)
        np.testing.assert_allclose(out.sum(dim=-1).numpy(), np.ones(16), atol=1e-5)

    # ── Architecture checks ───────────────────────────────────────────────────

    def test_architecture_ends_with_softmax(self):
        """The final layer in the sequential chain should be Softmax."""
        last_layer = list(self.model.model.children())[-1]
        self.assertIsInstance(last_layer, nn.Softmax)

    def test_architecture_contains_gelu_activations(self):
        """The model should contain at least one GELU activation layer."""
        gelu_layers = [m for m in self.model.model.modules() if isinstance(m, nn.GELU)]
        self.assertGreater(len(gelu_layers), 0)

    def test_architecture_contains_dropout_layers(self):
        """The model should contain at least one Dropout regularization layer."""
        dropout_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Dropout)
        ]
        self.assertGreater(len(dropout_layers), 0)

    def test_architecture_first_linear_input_dimension(self):
        """The first Linear layer should accept n_input_features as input."""
        linear_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Linear)
        ]
        self.assertEqual(linear_layers[0].in_features, self.n_features)

    def test_architecture_last_linear_output_dimension(self):
        """The final Linear layer should emit n_targets outputs."""
        linear_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Linear)
        ]
        self.assertEqual(linear_layers[-1].out_features, self.n_targets)

    def test_architecture_hidden_bottleneck_matches_input_features(self):
        """The penultimate hidden layer should project back to n_input_features."""
        # Architecture: input→512→256→n_input_features→n_targets
        # So the second-to-last Linear layer has out_features == n_input_features.
        linear_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Linear)
        ]
        self.assertEqual(linear_layers[-2].out_features, self.n_features)

    def test_architecture_standard_hidden_dims(self):
        """The first two hidden layers should have widths 512 and 256."""
        linear_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Linear)
        ]
        # Layer 0: n_features → 512
        self.assertEqual(linear_layers[0].out_features, 512)
        # Layer 1: 512 → 256
        self.assertEqual(linear_layers[1].in_features, 512)
        self.assertEqual(linear_layers[1].out_features, 256)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPDeconvolver – initialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverInitialization(unittest.TestCase):
    """Tests for MLPDeconvolver construction and initial state."""

    def test_stores_n_input_features(self):
        """Constructor should store n_input_features as an instance attribute."""
        deconvolver = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        self.assertEqual(deconvolver.n_input_features, N_FEATURES)

    def test_stores_n_targets(self):
        """Constructor should store n_targets as an instance attribute."""
        deconvolver = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        self.assertEqual(deconvolver.n_targets, N_TARGETS)

    def test_is_fitted_is_false_before_fit(self):
        """is_fitted should be False immediately after construction."""
        deconvolver = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        self.assertFalse(deconvolver.is_fitted)

    def test_inner_model_is_correct_type(self):
        """The wrapped model should be an instance of _MLPDeconvolverModel."""
        deconvolver = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        self.assertIsInstance(deconvolver.model, _MLPDeconvolverModel)

    def test_inner_model_has_matching_dimensions(self):
        """The wrapped model should be built with the given feature and target dims."""
        deconvolver = MLPDeconvolver(n_input_features=8, n_targets=3)
        linear_layers = [
            m for m in deconvolver.model.model.modules() if isinstance(m, nn.Linear)
        ]
        self.assertEqual(linear_layers[0].in_features, 8)
        self.assertEqual(linear_layers[-1].out_features, 3)

    def test_accepts_custom_logger(self):
        """Constructor should accept and store a custom logger."""
        import logging

        logger = logging.getLogger("test_mlp_logger")
        deconvolver = MLPDeconvolver(n_input_features=8, n_targets=3, logger=logger)
        self.assertIs(deconvolver.logger, logger)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPDeconvolver – fit
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverFit(unittest.TestCase):
    """Tests for the fit() method of MLPDeconvolver."""

    def setUp(self):
        """Create a fresh synthetic dataset for each test."""
        self.data = _make_synthetic_data()

    def _make_deconvolver(self) -> MLPDeconvolver:
        return MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)

    def _fit(self, deconvolver: MLPDeconvolver, **extra_kwargs) -> MLPDeconvolver:
        """Fit with minimal epochs on CPU."""
        kwargs = dict(
            X_val=self.data["X_val"],
            y_val=self.data["y_val"],
            n_epochs=2,
            device="cpu",
            verbose=0,
        )
        kwargs.update(extra_kwargs)
        return deconvolver.fit(self.data["X_train"], self.data["y_train"], **kwargs)

    # ── Happy-path ────────────────────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() should return the deconvolver instance to allow method chaining."""
        deconvolver = self._make_deconvolver()
        result = self._fit(deconvolver)
        self.assertIs(result, deconvolver)

    def test_fit_sets_is_fitted_to_true(self):
        """is_fitted should be True after a successful fit() call."""
        deconvolver = self._make_deconvolver()
        self._fit(deconvolver)
        self.assertTrue(deconvolver.is_fitted)

    def test_fit_stores_training_history(self):
        """fit() should store a TrainingHistory with at least one recorded epoch."""
        deconvolver = self._make_deconvolver()
        self._fit(deconvolver)
        self.assertIsInstance(deconvolver.history_, TrainingHistory)
        self.assertGreater(len(deconvolver.history_.val_loss), 0)

    def test_fit_stores_overridden_hyperparameters(self):
        """Custom training kwargs should be persisted as fitted attributes."""
        deconvolver = self._make_deconvolver()
        self._fit(
            deconvolver,
            n_epochs=3,
            batch_size=16,
            lr=5e-4,
            weight_decay=1e-5,
            early_stopping_patience=3,
            early_stopping_metric="val_mae",
            scheduler_type="plateau",
        )
        self.assertEqual(deconvolver.n_epochs_, 3)
        self.assertEqual(deconvolver.batch_size_, 16)
        self.assertAlmostEqual(deconvolver.lr_, 5e-4)
        self.assertAlmostEqual(deconvolver.weight_decay_, 1e-5)
        self.assertEqual(deconvolver.device_, "cpu")
        self.assertEqual(deconvolver.early_stopping_patience_, 3)
        self.assertEqual(deconvolver.scheduler_type_, "plateau")

    def test_fit_uses_documented_defaults(self):
        """Unspecified kwargs should fall back to the documented default values."""
        deconvolver = self._make_deconvolver()
        self._fit(deconvolver)
        self.assertEqual(deconvolver.batch_size_, 64)
        self.assertAlmostEqual(deconvolver.lr_, 1e-3)
        self.assertAlmostEqual(deconvolver.weight_decay_, 1e-4)
        self.assertEqual(deconvolver.early_stopping_metric_, "val_mae")
        self.assertEqual(deconvolver.early_stopping_patience_, 15)
        self.assertEqual(deconvolver.scheduler_type_, "plateau")

    # ── Validation checks ─────────────────────────────────────────────────────

    def test_fit_without_validation_data_raises(self):
        """fit() must raise AssertionError when X_val / y_val are absent."""
        deconvolver = self._make_deconvolver()
        with self.assertRaises(AssertionError):
            deconvolver.fit(
                self.data["X_train"],
                self.data["y_train"],
                n_epochs=2,
                device="cpu",
            )

    def test_fit_without_y_val_raises(self):
        """fit() must raise AssertionError when y_val is missing but X_val is given."""
        deconvolver = self._make_deconvolver()
        with self.assertRaises(AssertionError):
            deconvolver.fit(
                self.data["X_train"],
                self.data["y_train"],
                X_val=self.data["X_val"],
                n_epochs=2,
                device="cpu",
            )

    def test_fit_without_x_val_raises(self):
        """fit() must raise AssertionError when X_val is missing but y_val is given."""
        deconvolver = self._make_deconvolver()
        with self.assertRaises(AssertionError):
            deconvolver.fit(
                self.data["X_train"],
                self.data["y_train"],
                y_val=self.data["y_val"],
                n_epochs=2,
                device="cpu",
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPDeconvolver – predict
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverPredict(unittest.TestCase):
    """Tests for the predict() method of MLPDeconvolver."""

    @classmethod
    def setUpClass(cls):
        """Fit a shared deconvolver once to avoid repeated costly fitting."""
        cls.deconvolver = _fit_minimal_deconvolver()
        cls.rng = np.random.RandomState(99)  # pylint: disable=no-member

    def _random_X(self, n: int = N_TEST) -> np.ndarray:  # pylint: disable=invalid-name
        """Return a random input array of shape (n, N_FEATURES) for testing predict()."""
        return self.rng.rand(n, N_FEATURES).astype(np.float32)

    def test_predict_output_shape(self):
        """predict() should return an array of shape (n_samples, n_targets)."""
        pred = self.deconvolver.predict(self._random_X(N_TEST), device="cpu")
        self.assertEqual(pred.shape, (N_TEST, N_TARGETS))

    def test_predict_output_is_non_negative(self):
        """All predicted values should be non-negative."""
        pred = self.deconvolver.predict(self._random_X(), device="cpu")
        self.assertTrue(np.all(pred >= 0.0))

    def test_predict_output_sums_to_one(self):
        """Predicted proportions should sum to approximately 1 per sample."""
        pred = self.deconvolver.predict(self._random_X(), device="cpu")
        np.testing.assert_allclose(pred.sum(axis=1), np.ones(N_TEST), atol=1e-5)

    def test_predict_returns_numpy_array(self):
        """predict() should return a numpy ndarray, not a torch Tensor."""
        pred = self.deconvolver.predict(self._random_X(), device="cpu")
        self.assertIsInstance(pred, np.ndarray)

    def test_predict_single_sample(self):
        """predict() should handle a single-row input without error."""
        pred = self.deconvolver.predict(self._random_X(1), device="cpu")
        self.assertEqual(pred.shape, (1, N_TARGETS))

    def test_predict_is_deterministic_in_eval_mode(self):
        """Repeated calls with the same input should return identical results."""
        X = self._random_X(5)
        pred_a = self.deconvolver.predict(X, device="cpu")
        pred_b = self.deconvolver.predict(X, device="cpu")
        np.testing.assert_array_equal(pred_a, pred_b)

    def test_predict_accepts_explicit_cpu_device(self):
        """Passing device='cpu' explicitly should not raise."""
        pred = self.deconvolver.predict(self._random_X(5), device="cpu")
        self.assertEqual(pred.shape, (5, N_TARGETS))


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPDeconvolver – save / load
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverSaveLoad(unittest.TestCase):
    """Tests for the save() / load() round-trip of MLPDeconvolver."""

    def setUp(self):
        """Create a fresh temp directory and a fitted deconvolver for each test."""
        self.temp_dir = tempfile.mkdtemp(prefix="mlp_test_")
        self.model_path = Path(self.temp_dir) / "model.pt"
        self.deconvolver = _fit_minimal_deconvolver()

    def tearDown(self):
        """Remove temporary files created during tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ── save() ────────────────────────────────────────────────────────────────

    def test_save_creates_model_file(self):
        """save() should create the .pt weights file at the given path."""
        self.deconvolver.save(self.model_path)
        self.assertTrue(self.model_path.exists())

    def test_save_creates_metadata_json(self):
        """save() should create a companion JSON metadata file."""
        self.deconvolver.save(self.model_path)
        meta_path = self.model_path.parent / (self.model_path.stem + "_metadata.json")
        self.assertTrue(meta_path.exists())

    def test_save_metadata_contains_required_keys(self):
        """The metadata JSON should contain architecture and training parameters."""
        self.deconvolver.save(self.model_path)
        meta_path = self.model_path.parent / (self.model_path.stem + "_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for key in ("n_input_features", "n_targets", "is_fitted", "params"):
            self.assertIn(key, meta, msg=f"Missing key: {key}")

    def test_save_metadata_reflects_fitted_state(self):
        """The metadata should record is_fitted=True after a successful fit."""
        self.deconvolver.save(self.model_path)
        meta_path = self.model_path.parent / (self.model_path.stem + "_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertTrue(meta["is_fitted"])

    def test_save_before_fit_raises_value_error(self):
        """save() should raise ValueError when the model has not been fitted yet."""
        unfitted = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        with self.assertRaises(ValueError):
            unfitted.save(self.model_path)

    def test_save_rejects_non_pt_extension(self):
        """save() should raise AssertionError for paths not ending in .pt."""
        bad_path = Path(self.temp_dir) / "model.pkl"
        with self.assertRaises(AssertionError):
            self.deconvolver.save(bad_path)

    def test_save_uses_custom_metadata_path(self):
        """Passing metadata_path kwarg should write JSON to that location."""
        custom_meta = Path(self.temp_dir) / "custom_meta.json"
        self.deconvolver.save(self.model_path, metadata_path=custom_meta)
        self.assertTrue(custom_meta.exists())

    # ── load() ────────────────────────────────────────────────────────────────

    def test_load_returns_mlp_deconvolver(self):
        """load() should return an MLPDeconvolver instance."""
        self.deconvolver.save(self.model_path)
        loaded = MLPDeconvolver.load(self.model_path)
        self.assertIsInstance(loaded, MLPDeconvolver)

    def test_load_restores_is_fitted(self):
        """Loaded model should have is_fitted=True when saved in fitted state."""
        self.deconvolver.save(self.model_path)
        loaded = MLPDeconvolver.load(self.model_path)
        self.assertTrue(loaded.is_fitted)

    def test_load_restores_architecture_attributes(self):
        """Loaded model should have the same n_input_features and n_targets."""
        self.deconvolver.save(self.model_path)
        loaded = MLPDeconvolver.load(self.model_path)
        self.assertEqual(loaded.n_input_features, self.deconvolver.n_input_features)
        self.assertEqual(loaded.n_targets, self.deconvolver.n_targets)

    def test_load_restores_training_hyperparameters(self):
        """Loaded model should have the training hyperparameters from the fit call."""
        self.deconvolver.save(self.model_path)
        loaded = MLPDeconvolver.load(self.model_path)
        self.assertEqual(loaded.n_epochs_, self.deconvolver.n_epochs_)
        self.assertAlmostEqual(loaded.lr_, self.deconvolver.lr_)
        self.assertEqual(loaded.scheduler_type_, self.deconvolver.scheduler_type_)

    def test_load_produces_same_predictions(self):
        """Predictions from the loaded model should match those of the original."""
        # pylint: disable=no-member, invalid-name
        X_test = np.random.RandomState(7).rand(10, N_FEATURES).astype(np.float32)
        original_pred = self.deconvolver.predict(X_test, device="cpu")

        self.deconvolver.save(self.model_path)
        loaded = MLPDeconvolver.load(self.model_path)
        loaded_pred = loaded.predict(X_test, device="cpu")

        np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-5)

    def test_load_with_custom_metadata_path(self):
        """load() should accept and use a custom metadata_path kwarg."""
        custom_meta = Path(self.temp_dir) / "custom_meta.json"
        self.deconvolver.save(self.model_path, metadata_path=custom_meta)
        loaded = MLPDeconvolver.load(self.model_path, metadata_path=custom_meta)
        self.assertTrue(loaded.is_fitted)

    def test_load_rejects_non_pt_extension(self):
        """load() should raise AssertionError for paths not ending in .pt."""
        bad_path = Path(self.temp_dir) / "model.pkl"
        with self.assertRaises(AssertionError):
            MLPDeconvolver.load(bad_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  MLPDeconvolver – cross-validation interface
# ═══════════════════════════════════════════════════════════════════════════════


class TestMLPDeconvolverCVInterface(unittest.TestCase):
    """Tests for the cross-validation interface exposed by MLPDeconvolver."""

    @classmethod
    def setUpClass(cls):
        """Fit a shared deconvolver instance to avoid repeated fitting."""
        cls.deconvolver = _fit_minimal_deconvolver()
        cls.data = _make_synthetic_data()

    def test_cv_metric_name_is_string(self):
        """cv_metric_name should return a non-empty string."""
        name = self.deconvolver.cv_metric_name
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    def test_cv_metric_name_value(self):
        """cv_metric_name should report 'Validation loss'."""
        self.assertEqual(self.deconvolver.cv_metric_name, "Validation loss")

    def test_get_cv_metric_after_fit_returns_float(self):
        """get_cv_metric should return a scalar float after fitting."""
        metric = self.deconvolver.get_cv_metric(self.data["X_val"], self.data["y_val"])
        self.assertIsInstance(metric, float)

    def test_get_cv_metric_is_finite(self):
        """The CV metric should be a finite number."""
        metric = self.deconvolver.get_cv_metric(self.data["X_val"], self.data["y_val"])
        self.assertTrue(np.isfinite(metric))

    def test_get_cv_metric_before_fit_raises_value_error(self):
        """get_cv_metric should raise ValueError when the model is not yet fitted."""
        unfitted = MLPDeconvolver(n_input_features=N_FEATURES, n_targets=N_TARGETS)
        with self.assertRaises(ValueError):
            unfitted.get_cv_metric(self.data["X_val"], self.data["y_val"])

    def test_get_cv_metric_returns_best_epoch_val_loss(self):
        """get_cv_metric should return the validation loss at the best epoch."""
        expected = self.deconvolver.history_.val_loss[
            self.deconvolver.history_.best_epoch
        ]
        actual = self.deconvolver.get_cv_metric(self.data["X_val"], self.data["y_val"])
        self.assertAlmostEqual(actual, expected, places=10)


if __name__ == "__main__":
    unittest.main()
