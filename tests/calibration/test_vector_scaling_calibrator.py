import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch

from methyldl.calibration.vector_scaling_calibrator import (
    CalibrationMethod,
    _TrainedLinearCalibrationModel,
    VectorScalingCalibrator,
    # VectorScalingCalibratorCV,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_probs(n_samples: int, n_classes: int, seed: int = 42) -> np.ndarray:
    """Generate random probability vectors that sum to 1.

    Uses the exponential distribution so every component is positive,
    then normalises each row to obtain a valid probability simplex.

    Args:
        n_samples: Number of probability vectors to generate.
        n_classes: Dimensionality of each probability vector.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_samples, n_classes) with rows summing to 1.
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(size=(n_samples, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


def _make_hard_labels(n_samples: int, n_classes: int, seed: int = 42) -> np.ndarray:
    """Generate random integer class labels in [0, n_classes).

    Args:
        n_samples: Number of labels to generate.
        n_classes: Number of distinct classes (exclusive upper bound).
        seed: Random seed for reproducibility.

    Returns:
        1-D integer array of shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n_samples)


def _fit_quick_calibrator(**overrides) -> VectorScalingCalibrator:
    """Return a VectorScalingCalibrator fitted on tiny dummy data.

    Provides sensible fast defaults (few epochs, CPU, silent) that can
    be selectively overridden via keyword arguments.  Useful for tests
    that need a fitted calibrator but don't care about the training
    hyper-parameters themselves.

    Args:
        **overrides: Any keyword argument accepted by
            ``VectorScalingCalibrator.__init__``.  Overrides the
            built-in fast defaults.

    Returns:
        A fitted ``VectorScalingCalibrator`` instance.
    """
    defaults = dict(
        max_iter=5,
        patience=3,
        device="cpu",
        verbose=False,
        scheduler="cosine",
    )
    defaults.update(overrides)
    cal = VectorScalingCalibrator(**defaults)
    X = _make_dummy_probs(50, 4)
    y = _make_hard_labels(50, 4)
    cal.fit(X, y)
    return cal


# ===================================================================
# _TrainedLinearCalibrationModel tests
# ===================================================================


class TestCalibrationMethod(unittest.TestCase):
    """Tests for the CalibrationMethod enum."""

    def test_enum_values(self):
        """Each enum member should expose its string value."""
        self.assertEqual(CalibrationMethod.FULL.value, "full")
        self.assertEqual(CalibrationMethod.DIAGONAL.value, "diagonal")
        self.assertEqual(CalibrationMethod.TEMPERATURE.value, "temperature")

    def test_construction_from_string(self):
        """An enum member should be recoverable from its string value."""
        self.assertIs(CalibrationMethod("diagonal"), CalibrationMethod.DIAGONAL)


class TestTrainedLinearCalibrationModel(unittest.TestCase):
    """Tests for all three parametrisations of the calibration model."""

    K = 4  # number of classes

    # ---- Construction ----

    def test_full_method_creates_W_delta_and_bias(self):
        """FULL parametrisation should expose a (K, K) W_delta and a (K,) bias."""
        m = _TrainedLinearCalibrationModel(self.K, CalibrationMethod.FULL)
        self.assertTrue(hasattr(m, "W_delta"))
        self.assertTrue(hasattr(m, "b"))
        self.assertEqual(m.W_delta.shape, (self.K, self.K))
        self.assertEqual(m.b.shape, (self.K,))

    def test_diagonal_method_creates_diag_delta_and_bias(self):
        """DIAGONAL parametrisation should expose a (K,) diag_delta and a (K,) bias."""
        m = _TrainedLinearCalibrationModel(self.K, CalibrationMethod.DIAGONAL)
        self.assertTrue(hasattr(m, "diag_delta"))
        self.assertTrue(hasattr(m, "b"))
        self.assertEqual(m.diag_delta.shape, (self.K,))
        self.assertEqual(m.b.shape, (self.K,))

    def test_temperature_method_creates_scalar_t_delta(self):
        """TEMPERATURE parametrisation should have a single scalar and no bias."""
        m = _TrainedLinearCalibrationModel(self.K, CalibrationMethod.TEMPERATURE)
        self.assertTrue(hasattr(m, "t_delta"))
        self.assertEqual(m.t_delta.shape, (1,))
        # Temperature scaling has no bias parameter by design.
        self.assertFalse(hasattr(m, "b"))

    def test_unknown_method_raises(self):
        """Passing a string that is not a CalibrationMethod should raise ValueError."""
        with self.assertRaises(ValueError):
            _TrainedLinearCalibrationModel(self.K, "invalid")

    # ---- Identity initialisation ----

    def _assert_identity_at_init(self, method: CalibrationMethod):
        """Verify that a freshly-constructed model acts as the identity map.

        Because all delta parameters are zero-initialised, the forward
        pass is logits = x and activate(logits) = softmax(x).  If x
        is already log(q), then softmax(log(q)) = q.  So feeding
        log-probabilities should recover the original probabilities.
        """
        m = _TrainedLinearCalibrationModel(self.K, method)
        m.eval()
        # Feed log-probs so that softmax recovers the original distribution.
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
        x = torch.log(probs)
        with torch.no_grad():
            out = m.activate(m(x))
        np.testing.assert_allclose(out.numpy(), probs.numpy(), atol=1e-12)

    def test_full_identity_init(self):
        """FULL parametrisation should start as the identity calibration map."""
        self._assert_identity_at_init(CalibrationMethod.FULL)

    def test_diagonal_identity_init(self):
        """DIAGONAL parametrisation should start as the identity calibration map."""
        self._assert_identity_at_init(CalibrationMethod.DIAGONAL)

    def test_temperature_identity_init(self):
        """TEMPERATURE parametrisation should start as the identity calibration map."""
        self._assert_identity_at_init(CalibrationMethod.TEMPERATURE)

    # ---- forward ----

    def test_forward_unknown_method_raises(self):
        """forward() should raise ValueError when the method attribute is corrupted."""
        m = _TrainedLinearCalibrationModel(self.K, CalibrationMethod.DIAGONAL)
        # Manually corrupt the method to exercise the else-branch in forward().
        m.method = "bogus"
        x = torch.zeros(2, self.K, dtype=torch.float64)
        with self.assertRaises(ValueError):
            m(x)

    # ---- activate produces valid probabilities ----

    def test_activate_sums_to_one(self):
        """activate() applies softmax, so each row must sum to 1."""
        m = _TrainedLinearCalibrationModel(self.K, CalibrationMethod.DIAGONAL)
        logits = torch.randn(5, self.K, dtype=torch.float64)
        probs = m.activate(logits)
        np.testing.assert_allclose(probs.sum(dim=1).numpy(), np.ones(5), atol=1e-12)


# ===================================================================
# VectorScalingCalibrator tests
# ===================================================================


class TestVectorScalingCalibratorInit(unittest.TestCase):
    """Tests for VectorScalingCalibrator construction and parameter validation."""

    def test_default_parameters(self):
        """Constructor defaults should match the documented signature."""
        cal = VectorScalingCalibrator()
        self.assertEqual(cal.reg_lambda, 0.0)
        self.assertEqual(cal.optimizer, "adam")
        self.assertEqual(cal.scheduler, "plateau")
        self.assertEqual(cal.lr, 0.01)
        self.assertEqual(cal.max_iter, 500)
        self.assertEqual(cal.batch_size, 1000)
        self.assertEqual(cal.patience, 50)
        self.assertEqual(cal.device, "cuda")
        self.assertTrue(cal.verbose)
        self.assertEqual(cal.plateau_factor, 0.5)
        self.assertEqual(cal.plateau_patience, 10)

    def test_unfitted_state(self):
        """Before fit() is called, fitted attributes should be at sentinel values."""
        cal = VectorScalingCalibrator()
        self.assertIsNone(cal.model_)
        self.assertEqual(cal.n_classes_, 0)
        self.assertEqual(cal.final_loss_, float("inf"))
        self.assertEqual(cal.best_epoch_, -1)
        self.assertEqual(cal.best_metrics_, {})
        self.assertEqual(cal.history_, {})

    def test_custom_parameters(self):
        """Explicitly passed parameters should override the defaults."""
        cal = VectorScalingCalibrator(
            reg_lambda=0.1,
            lr=0.001,
            max_iter=100,
            optimizer="sgd",
            scheduler="linear",
            batch_size=64,
            patience=10,
            tol=1e-5,
            device="cpu",
            verbose=False,
        )
        self.assertEqual(cal.reg_lambda, 0.1)
        self.assertEqual(cal.optimizer, "sgd")
        self.assertEqual(cal.scheduler, "linear")
        self.assertEqual(cal.lr, 0.001)

    def test_unsupported_optimizer_raises(self):
        """Only 'adam' and 'sgd' are valid optimizers."""
        with self.assertRaises(AssertionError):
            VectorScalingCalibrator(optimizer="rmsprop")

    def test_unsupported_scheduler_raises(self):
        """Only 'plateau', 'constant', 'linear', 'cosine' are valid schedulers."""
        with self.assertRaises(AssertionError):
            VectorScalingCalibrator(scheduler="cyclic")


class TestVectorScalingCalibratorStaticMethods(unittest.TestCase):
    """Tests for static and private helper methods on VectorScalingCalibrator."""

    def test_clip_and_log_avoids_inf(self):
        """Boundary values 0 and 1 should be clipped to avoid -inf / +inf."""
        X = np.array([[0.0, 1.0, 0.5]], dtype=np.float64)
        result = VectorScalingCalibrator._clip_and_log(X)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_clip_and_log_correct_for_interior_values(self):
        """For values well inside (0, 1), clipping should be a no-op."""
        X = np.array([[0.25, 0.75]], dtype=np.float64)
        result = VectorScalingCalibrator._clip_and_log(X)
        np.testing.assert_allclose(result, np.log(X), atol=1e-12)

    def test_prepare_targets_integer_labels(self):
        """1-D integer labels should be converted to one-hot rows."""
        targets = VectorScalingCalibrator._prepare_targets(
            np.array([0, 2, 1]), n_samples=3, n_classes=3
        )
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(targets, expected)

    def test_prepare_targets_one_hot(self):
        """2-D one-hot input should pass through unchanged."""
        one_hot = np.eye(3, dtype=np.float64)
        targets = VectorScalingCalibrator._prepare_targets(
            one_hot, n_samples=3, n_classes=3
        )
        np.testing.assert_array_equal(targets, one_hot)

    def test_prepare_targets_soft_labels(self):
        """2-D soft probability targets should pass through unchanged."""
        soft = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]], dtype=np.float64)
        targets = VectorScalingCalibrator._prepare_targets(
            soft, n_samples=2, n_classes=3
        )
        np.testing.assert_array_equal(targets, soft)

    def test_prepare_targets_shape_mismatch_raises(self):
        """A 2-D target whose shape disagrees with (n_samples, n_classes) should raise."""
        wrong = np.array([[0.5, 0.5]], dtype=np.float64)  # shape (1, 2)
        with self.assertRaises(ValueError):
            VectorScalingCalibrator._prepare_targets(wrong, n_samples=3, n_classes=3)


class TestVectorScalingCalibratorFit(unittest.TestCase):
    """Tests for the fit() method."""

    def test_fit_returns_self(self):
        """fit() should return the calibrator instance for method chaining."""
        cal = VectorScalingCalibrator(
            max_iter=3, patience=2, device="cpu", verbose=False, scheduler="cosine"
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        result = cal.fit(X, y)
        self.assertIs(result, cal)

    def test_fit_populates_fitted_attributes(self):
        """After fitting, all fitted attributes should be populated."""
        cal = _fit_quick_calibrator()
        self.assertIsNotNone(cal.model_)
        self.assertEqual(cal.n_classes_, 4)
        self.assertIsInstance(cal.final_loss_, float)
        self.assertNotEqual(cal.best_metrics_, {})
        self.assertIn("epoch", cal.history_)
        self.assertIn("train_loss", cal.history_)
        self.assertIn("train_mse", cal.history_)
        self.assertIn("lr", cal.history_)

    def test_fit_with_validation_set(self):
        """Providing X_val/y_val should add val_loss and val_mse to history."""
        cal = VectorScalingCalibrator(
            max_iter=5, patience=3, device="cpu", verbose=False, scheduler="cosine"
        )
        X = _make_dummy_probs(40, 3, seed=0)
        y = _make_hard_labels(40, 3, seed=0)
        # Use a different seed so validation data is independent of training.
        X_val = _make_dummy_probs(20, 3, seed=1)
        y_val = _make_hard_labels(20, 3, seed=1)
        cal.fit(X, y, X_val=X_val, y_val=y_val)

        self.assertIn("val_loss", cal.history_)
        self.assertIn("val_mse", cal.history_)

    def test_fit_with_soft_labels(self):
        """fit() should accept soft probability targets (rows summing to 1)."""
        cal = VectorScalingCalibrator(
            max_iter=5, patience=3, device="cpu", verbose=False, scheduler="cosine"
        )
        X = _make_dummy_probs(30, 3)
        # Soft targets: each row is a probability distribution, not one-hot.
        y = _make_dummy_probs(30, 3, seed=99)
        cal.fit(X, y)
        self.assertIsNotNone(cal.model_)

    def test_fit_full_batch_when_batch_size_zero(self):
        """batch_size=0 is treated as full-batch (entire dataset per epoch)."""
        cal = VectorScalingCalibrator(
            max_iter=3,
            patience=2,
            device="cpu",
            verbose=False,
            batch_size=0,
            scheduler="cosine",
        )
        X = _make_dummy_probs(20, 3)
        y = _make_hard_labels(20, 3)
        cal.fit(X, y)
        self.assertIsNotNone(cal.model_)

    def test_fit_full_batch_when_batch_size_none(self):
        """batch_size=None is treated as full-batch (entire dataset per epoch)."""
        cal = VectorScalingCalibrator(
            max_iter=3,
            patience=2,
            device="cpu",
            verbose=False,
            batch_size=None,
            scheduler="cosine",
        )
        X = _make_dummy_probs(20, 3)
        y = _make_hard_labels(20, 3)
        cal.fit(X, y)
        self.assertIsNotNone(cal.model_)

    def test_history_starts_at_epoch_minus_one(self):
        """The first history entry records the untrained model (epoch -1)."""
        cal = _fit_quick_calibrator()
        self.assertEqual(cal.history_["epoch"][0], -1)

    def test_early_stopping_triggers(self):
        """With patience=1 and many iterations, training should stop early."""
        cal = VectorScalingCalibrator(
            max_iter=1000,
            patience=1,
            device="cpu",
            verbose=False,
            scheduler="cosine",
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        cal.fit(X, y)
        # History length = 1 (epoch -1) + actual epochs; should be << max_iter
        self.assertLess(len(cal.history_["epoch"]), 1000)

    def test_fit_with_sgd_optimizer(self):
        """fit() should work with the SGD optimizer."""
        cal = VectorScalingCalibrator(
            max_iter=3,
            patience=2,
            device="cpu",
            verbose=False,
            optimizer="sgd",
            scheduler="cosine",
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        cal.fit(X, y)
        self.assertIsNotNone(cal.model_)

    def test_fit_with_constant_scheduler(self):
        """With a constant schedule, the learning rate should never change."""
        cal = VectorScalingCalibrator(
            max_iter=3,
            patience=2,
            device="cpu",
            verbose=False,
            scheduler="constant",
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        cal.fit(X, y)
        # Every recorded LR should equal the initial LR.
        self.assertTrue(all(lr == cal.lr for lr in cal.history_["lr"]))

    def test_fit_with_linear_scheduler(self):
        """With a linear schedule, the learning rate should decay over epochs."""
        cal = VectorScalingCalibrator(
            max_iter=5,
            # High patience to avoid early stopping before all epochs run.
            patience=10,
            device="cpu",
            verbose=False,
            scheduler="linear",
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        cal.fit(X, y)
        # Skip epoch -1 (index 0) whose LR is the initial value.
        lrs = cal.history_["lr"][1:]
        self.assertGreater(lrs[0], lrs[-1])

    def test_fit_with_plateau_scheduler(self):
        """fit() should work with the ReduceLROnPlateau scheduler."""
        cal = VectorScalingCalibrator(
            max_iter=5,
            patience=10,
            device="cpu",
            verbose=False,
            scheduler="plateau",
            plateau_factor=0.5,
            # plateau_patience=1 so the scheduler can actually trigger.
            plateau_patience=1,
        )
        X = _make_dummy_probs(30, 3)
        y = _make_hard_labels(30, 3)
        cal.fit(X, y)
        self.assertIsNotNone(cal.model_)

    def test_fit_with_regularisation(self):
        """Heavy L2 regularisation should increase the reported final loss."""
        cal_noreg = _fit_quick_calibrator(reg_lambda=0.0)
        cal_reg = _fit_quick_calibrator(reg_lambda=10.0)
        # The loss includes the L2 penalty, so a large lambda inflates it.
        self.assertGreater(cal_reg.final_loss_, cal_noreg.final_loss_)

    def test_best_metrics_keys_match_history(self):
        """best_metrics_ should have one entry per history key (snapshot at best epoch)."""
        cal = _fit_quick_calibrator()
        for key in cal.history_:
            self.assertIn(key, cal.best_metrics_)


class TestVectorScalingCalibratorPredict(unittest.TestCase):
    """Tests for predict method."""

    def setUp(self):
        """Create a fitted calibrator shared by all predict tests."""
        self.cal = _fit_quick_calibrator()

    def test_predict_shape(self):
        """Output shape should match (n_samples, n_classes)."""
        X = _make_dummy_probs(10, 4, seed=99)
        out = self.cal.predict(X)
        self.assertEqual(out.shape, (10, 4))

    def test_predict_sums_to_one(self):
        """Calibrated probabilities must sum to 1 per sample (softmax output)."""
        X = _make_dummy_probs(10, 4, seed=99)
        out = self.cal.predict(X)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(10), atol=1e-6)

    def test_predict_nonnegative(self):
        """Calibrated probabilities must be non-negative (softmax output)."""
        X = _make_dummy_probs(10, 4, seed=99)
        out = self.cal.predict(X)
        self.assertTrue(np.all(out >= 0))


class TestVectorScalingCalibratorSaveLoad(unittest.TestCase):
    """Tests for save / load round-trip."""

    def test_save_unfitted_raises(self):
        """Saving before fit() should raise RuntimeError."""
        cal = VectorScalingCalibrator(device="cpu")
        with TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                cal.save(Path(td) / "model.npz")

    def test_save_load_round_trip(self):
        """Save then load should produce identical predictions and metadata."""
        cal = _fit_quick_calibrator()
        X_test = _make_dummy_probs(10, 4, seed=99)
        original_preds = cal.predict(X_test)

        with TemporaryDirectory() as td:
            path = Path(td) / "model.npz"
            cal.save(path)

            loaded = VectorScalingCalibrator(device="cpu")
            loaded.load(path)

        np.testing.assert_allclose(loaded.predict(X_test), original_preds, atol=1e-12)
        self.assertEqual(loaded.n_classes_, cal.n_classes_)
        self.assertAlmostEqual(loaded.final_loss_, cal.final_loss_)
        self.assertEqual(loaded.best_epoch_, cal.best_epoch_)

    def test_load_restores_constructor_params(self):
        """load() should restore constructor hyper-parameters from the saved file."""
        cal = _fit_quick_calibrator(reg_lambda=0.05, lr=0.005, max_iter=5)
        with TemporaryDirectory() as td:
            path = Path(td) / "model.npz"
            cal.save(path)
            loaded = VectorScalingCalibrator(device="cpu")
            loaded.load(path)

        self.assertAlmostEqual(loaded.reg_lambda, 0.05)
        self.assertAlmostEqual(loaded.lr, 0.005)
        self.assertEqual(loaded.max_iter, 5)

    def test_save_creates_npz_file(self):
        """save() should create a .npz file on disk."""
        cal = _fit_quick_calibrator()
        with TemporaryDirectory() as td:
            path = Path(td) / "model.npz"
            cal.save(path)
            self.assertTrue(path.exists())


class TestVectorScalingCalibratorComputeLoss(unittest.TestCase):
    """Tests for the internal _compute_loss method."""

    def test_loss_is_scalar(self):
        """_compute_loss should return a 0-dimensional (scalar) tensor."""
        cal = VectorScalingCalibrator(device="cpu", reg_lambda=0.0)
        model = _TrainedLinearCalibrationModel(3, CalibrationMethod.DIAGONAL)
        inputs = torch.randn(5, 3, dtype=torch.float64)
        targets = torch.zeros(5, 3, dtype=torch.float64)
        targets[:, 0] = 1.0
        loss = cal._compute_loss(model, inputs, targets)
        self.assertEqual(loss.dim(), 0)

    def test_regularisation_increases_loss(self):
        """With non-zero parameters, L2 regularisation should add to the loss."""
        model = _TrainedLinearCalibrationModel(3, CalibrationMethod.DIAGONAL)
        # Set non-zero deltas so the L2 penalty is non-trivial.
        with torch.no_grad():
            model.diag_delta.fill_(1.0)
            model.b.fill_(0.5)

        inputs = torch.randn(5, 3, dtype=torch.float64)
        targets = torch.zeros(5, 3, dtype=torch.float64)
        targets[:, 0] = 1.0

        cal_noreg = VectorScalingCalibrator(device="cpu", reg_lambda=0.0)
        cal_reg = VectorScalingCalibrator(device="cpu", reg_lambda=1.0)

        loss_noreg = cal_noreg._compute_loss(model, inputs, targets).item()
        loss_reg = cal_reg._compute_loss(model, inputs, targets).item()
        # The regularised loss = NLL + lambda * ||params||^2, so must be larger.
        self.assertGreater(loss_reg, loss_noreg)


# # ===================================================================
# # VectorScalingCalibratorCV tests
# # ===================================================================


# class TestVectorScalingCalibratorCVInit(unittest.TestCase):
#     """Tests for VectorScalingCalibratorCV construction."""

#     def test_default_unfitted_attributes(self):
#         """Before fit(), all CV result attributes should be None."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[10]
#         )
#         self.assertIsNone(cv.best_calibrator_)
#         self.assertIsNone(cv.best_params_)
#         self.assertIsNone(cv.best_cv_val_loss_)
#         self.assertIsNone(cv.best_metrics_per_param_)

#     def test_constructor_stores_all_params(self):
#         """All constructor arguments should be stored as instance attributes."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0, 0.1],
#             lr_list=[0.01],
#             max_iter_list=[10, 20],
#             optimizer="sgd",
#             scheduler="cosine",
#             patience=5,
#             tol=1e-6,
#             n_folds=3,
#             batch_size=32,
#             verbose=True,
#             plateau_factor=0.3,
#             plateau_patience=5,
#         )
#         self.assertEqual(cv.reg_lambda_list, [0.0, 0.1])
#         self.assertEqual(cv.optimizer, "sgd")
#         self.assertEqual(cv.scheduler, "cosine")
#         self.assertEqual(cv.n_folds, 3)
#         self.assertEqual(cv.plateau_factor, 0.3)
#         self.assertEqual(cv.plateau_patience, 5)


# class TestVectorScalingCalibratorCVFit(unittest.TestCase):
#     """Tests for VectorScalingCalibratorCV.fit()."""

#     @classmethod
#     def setUpClass(cls):
#         """Fit a CV calibrator once for reuse across tests (CPU, fast).

#         Uses 2 folds and 2 reg_lambda values (4 inner fits total) to keep
#         the test suite fast while still exercising the grid-search logic.
#         """
#         cls.n_classes = 3
#         cls.X = _make_dummy_probs(60, cls.n_classes, seed=0)
#         cls.y = _make_hard_labels(60, cls.n_classes, seed=0)

#         cls.cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0, 0.01],
#             lr_list=[0.01],
#             max_iter_list=[5],
#             scheduler="cosine",
#             patience=3,
#             n_folds=2,
#             batch_size=0,
#             verbose=False,
#         )
#         cls.cv.fit(cls.X, cls.y)

#     def test_fit_returns_self(self):
#         """fit() should return the CV instance for method chaining."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0],
#             lr_list=[0.01],
#             max_iter_list=[3],
#             scheduler="cosine",
#             patience=2,
#             n_folds=2,
#             verbose=False,
#         )
#         X = _make_dummy_probs(30, 3)
#         y = _make_hard_labels(30, 3)
#         result = cv.fit(X, y)
#         self.assertIs(result, cv)

#     def test_fit_populates_best_calibrator(self):
#         """After fitting, best_calibrator_ should be a VectorScalingCalibrator."""
#         self.assertIsNotNone(self.cv.best_calibrator_)
#         self.assertIsInstance(self.cv.best_calibrator_, VectorScalingCalibrator)

#     def test_fit_populates_best_params(self):
#         """After fitting, best_params_ should contain the grid-search keys."""
#         self.assertIsNotNone(self.cv.best_params_)
#         self.assertIn("reg_lambda", self.cv.best_params_)
#         self.assertIn("lr", self.cv.best_params_)
#         self.assertIn("max_iter", self.cv.best_params_)

#     def test_fit_populates_cv_val_loss(self):
#         """After fitting, best_cv_val_loss_ should be a finite float."""
#         self.assertIsNotNone(self.cv.best_cv_val_loss_)
#         self.assertIsInstance(self.cv.best_cv_val_loss_, float)

#     def test_fit_populates_metrics_per_param(self):
#         """After fitting, best_metrics_per_param_ should have at least one entry."""
#         self.assertIsNotNone(self.cv.best_metrics_per_param_)
#         self.assertGreater(len(self.cv.best_metrics_per_param_), 0)


# class TestVectorScalingCalibratorCVPredict(unittest.TestCase):
#     """Tests for CV predict / predict_proba methods.

#     These delegate to the inner best_calibrator_, so the tests check
#     both the delegation and the unfitted guard.
#     """

#     @classmethod
#     def setUpClass(cls):
#         """Fit a minimal CV calibrator once for prediction tests."""
#         cls.n_classes = 3
#         X = _make_dummy_probs(60, cls.n_classes, seed=0)
#         y = _make_hard_labels(60, cls.n_classes, seed=0)
#         cls.cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0],
#             lr_list=[0.01],
#             max_iter_list=[5],
#             scheduler="cosine",
#             patience=3,
#             n_folds=2,
#             verbose=False,
#         )
#         cls.cv.fit(X, y)

#     def test_predict_proba_shape(self):
#         """Output shape should match (n_samples, n_classes)."""
#         X = _make_dummy_probs(10, self.n_classes, seed=99)
#         out = self.cv.predict_proba(X)
#         self.assertEqual(out.shape, (10, self.n_classes))

#     def test_predict_proba_sums_to_one(self):
#         """Calibrated probabilities must sum to 1 per sample."""
#         X = _make_dummy_probs(10, self.n_classes, seed=99)
#         out = self.cv.predict_proba(X)
#         np.testing.assert_allclose(out.sum(axis=1), np.ones(10), atol=1e-6)

#     def test_predict_is_alias(self):
#         """predict() and predict_proba() should return identical results."""
#         X = _make_dummy_probs(10, self.n_classes, seed=99)
#         np.testing.assert_array_equal(self.cv.predict(X), self.cv.predict_proba(X))

#     def test_predict_proba_unfitted_raises(self):
#         """predict_proba() before fit() should raise RuntimeError."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#         )
#         with self.assertRaises(RuntimeError):
#             cv.predict_proba(_make_dummy_probs(5, 3))

#     def test_predict_unfitted_raises(self):
#         """predict() before fit() should raise RuntimeError."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#         )
#         with self.assertRaises(RuntimeError):
#             cv.predict(_make_dummy_probs(5, 3))


# class TestVectorScalingCalibratorCVSaveLoad(unittest.TestCase):
#     """Tests for CV save / load round-trip.

#     The CV calibrator serialises both the grid-search metadata and the
#     inner best calibrator (including model weights) into a single .npz.
#     """

#     @classmethod
#     def setUpClass(cls):
#         """Fit a minimal CV calibrator once for serialisation tests."""
#         cls.n_classes = 3
#         X = _make_dummy_probs(60, cls.n_classes, seed=0)
#         y = _make_hard_labels(60, cls.n_classes, seed=0)
#         cls.cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0],
#             lr_list=[0.01],
#             max_iter_list=[5],
#             scheduler="cosine",
#             patience=3,
#             n_folds=2,
#             verbose=False,
#         )
#         cls.cv.fit(X, y)

#     def test_save_unfitted_raises(self):
#         """Saving before fit() should raise RuntimeError."""
#         cv = VectorScalingCalibratorCV(
#             reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#         )
#         with TemporaryDirectory() as td:
#             with self.assertRaises(RuntimeError):
#                 cv.save(Path(td) / "model.npz")

#     def test_save_load_round_trip_predictions(self):
#         """Predictions from a loaded CV calibrator should match the original."""
#         X_test = _make_dummy_probs(10, self.n_classes, seed=99)
#         original_preds = self.cv.predict_proba(X_test)

#         with TemporaryDirectory() as td:
#             path = Path(td) / "cv_model.npz"
#             self.cv.save(path)

#             loaded = VectorScalingCalibratorCV(
#                 reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#             )
#             loaded.load(path)

#         np.testing.assert_allclose(
#             loaded.predict_proba(X_test), original_preds, atol=1e-12
#         )

#     def test_save_load_restores_cv_metadata(self):
#         """load() should restore best_params_, best_cv_val_loss_, and inner calibrator."""
#         with TemporaryDirectory() as td:
#             path = Path(td) / "cv_model.npz"
#             self.cv.save(path)

#             loaded = VectorScalingCalibratorCV(
#                 reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#             )
#             loaded.load(path)

#         self.assertEqual(loaded.best_params_, self.cv.best_params_)
#         self.assertAlmostEqual(loaded.best_cv_val_loss_, self.cv.best_cv_val_loss_)
#         self.assertIsNotNone(loaded.best_calibrator_)
#         self.assertEqual(loaded.best_calibrator_.n_classes_, self.n_classes)

#     def test_save_load_restores_metrics_per_param(self):
#         """load() should restore the per-hyperparameter metrics dictionary."""
#         with TemporaryDirectory() as td:
#             path = Path(td) / "cv_model.npz"
#             self.cv.save(path)

#             loaded = VectorScalingCalibratorCV(
#                 reg_lambda_list=[0.0], lr_list=[0.01], max_iter_list=[5]
#             )
#             loaded.load(path)

#         self.assertIsNotNone(loaded.best_metrics_per_param_)
#         self.assertEqual(
#             len(loaded.best_metrics_per_param_),
#             len(self.cv.best_metrics_per_param_),
#         )

#     def test_save_creates_file(self):
#         """save() should create a .npz file on disk."""
#         with TemporaryDirectory() as td:
#             path = Path(td) / "cv_model.npz"
#             self.cv.save(path)
#             self.assertTrue(path.exists())


# if __name__ == "__main__":
#     unittest.main()
