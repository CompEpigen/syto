import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from methyldl.deconvolution.xgbdeconvolver import (
    XGBDeconvolverConfig,
    XGBoostDeconvolver,
    train_xgb_deconvolver,
)
from methyldl.deconvolution.history import DeconvolutionHistory


class DummyEstimator:
    """Minimal estimator stub exposing feature importances."""

    def __init__(self, importance):
        """Store deterministic feature importances for aggregation tests."""
        self.feature_importances_ = np.array(importance, dtype=float)


class DummyMultiOutputModel:
    """Small stand-in for MultiOutputRegressor used by unit tests."""

    def __init__(self, output):
        """Configure the synthetic prediction output returned by predict."""
        self._output = np.array(output, dtype=float)
        self.fit_calls = 0

    def fit(self, X, y):
        """Record fit calls while behaving like a fitted sklearn estimator."""
        self.fit_calls += 1
        return self

    def predict(self, X):
        """Return outputs with a shape compatible with the requested batch size."""
        n_samples = X.shape[0]
        # Repeat a 1D template to mimic one prediction vector per sample.
        if self._output.ndim == 1:
            return np.tile(self._output[np.newaxis, :], (n_samples, 1))
        if self._output.shape[0] == n_samples:
            return self._output
        # Fall back to repeating the first row when shapes do not align.
        return np.tile(self._output[0][np.newaxis, :], (n_samples, 1))


class TestXGBTrainingHistory(unittest.TestCase):
    """Tests for the DeconvolutionHistory convenience dataclass."""

    def test_to_dict_contains_expected_keys_and_values(self):
        """to_dict should expose metric keys and scalar metadata fields."""
        history = DeconvolutionHistory(
            train_loss=[0.1],
            train_metrics={"mae": [0.2]},
            val_loss=[0.3],
            best_epoch=7,
            stopped_early=True,
        )

        as_dict = history.to_dict()

        self.assertIn("train_loss", as_dict)
        self.assertIn("val_loss", as_dict)
        self.assertEqual(as_dict["best_epoch"], 7)
        self.assertTrue(as_dict["stopped_early"])


class TestXGBoostDeconvolverCore(unittest.TestCase):
    """Unit tests for core, non-IO behavior of XGBoostDeconvolver."""

    def setUp(self):
        """Create a compact deterministic fixture for shape and value assertions."""
        self.n_dmr = 3
        self.n_pred_classes = 4
        self.n_cell_types = 3

        self.model = XGBoostDeconvolver(
            n_dmr_groups=self.n_dmr,
            n_pred_classes=self.n_pred_classes,
            n_cell_types=self.n_cell_types,
            with_reject_features=True,
            process_inputs=True,
            output_transform="clip_normalize",
        )

        # Shape: (2, 3, 4)
        self.X_3d = np.array(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
                [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8], [1.9, 2.0, 2.1, 2.2]],
            ],
            dtype=float,
        )
        self.y = np.array([[0.2, 0.3, 0.5], [0.4, 0.1, 0.5]], dtype=float)

    def test_init_sets_feature_count_with_and_without_reject(self):
        """n_features should reflect whether reject-column features are enabled."""
        with_reject = XGBoostDeconvolver(n_dmr_groups=5, with_reject_features=True)
        without_reject = XGBoostDeconvolver(n_dmr_groups=5, with_reject_features=False)

        self.assertEqual(with_reject.n_features, 10)
        self.assertEqual(without_reject.n_features, 5)

    def test_build_model_uses_config(self):
        """_build_model should propagate user config into the base regressor."""
        cfg = XGBDeconvolverConfig(n_estimators=11, max_depth=4, learning_rate=0.05)
        model = XGBoostDeconvolver(config=cfg)

        wrapper = model._build_model(verbose=0)

        self.assertEqual(wrapper.estimator.n_estimators, 11)
        self.assertEqual(wrapper.estimator.max_depth, 4)
        self.assertEqual(wrapper.estimator.learning_rate, 0.05)

    def test_extract_features_with_reject(self):
        """Feature extraction should concatenate diagonal and reject-column values."""
        features = self.model.extract_features(self.X_3d)

        expected_diagonal = np.array([[0.1, 0.6, 1.1], [1.1, 1.6, 2.1]])
        expected_reject = np.array([[0.4, 0.8, 1.2], [1.4, 1.8, 2.2]])
        expected = np.concatenate([expected_diagonal, expected_reject], axis=1)

        np.testing.assert_allclose(features, expected)

    def test_extract_features_without_reject(self):
        """When reject features are disabled, only the diagonal should remain."""
        model = XGBoostDeconvolver(
            n_dmr_groups=self.n_dmr,
            n_pred_classes=self.n_pred_classes,
            with_reject_features=False,
            process_inputs=True,
        )

        features = model.extract_features(self.X_3d)
        expected_diagonal = np.array([[0.1, 0.6, 1.1], [1.1, 1.6, 2.1]])
        np.testing.assert_allclose(features, expected_diagonal)

    def test_extract_features_process_inputs_false_returns_input(self):
        """process_inputs=False should bypass feature extraction entirely."""
        model = XGBoostDeconvolver(process_inputs=False)
        raw = np.array([[1.0, 2.0], [3.0, 4.0]])

        returned = model.extract_features(raw)

        self.assertTrue(returned is raw)

    def test_extract_features_promotes_2d_to_single_sample(self):
        """2D inputs should be promoted to a batch of size one."""
        single = self.X_3d[0]
        features = self.model.extract_features(single)
        self.assertEqual(features.shape, (1, self.n_dmr * 2))

    def test_transform_output_none_clips_and_normalizes(self):
        """The 'none' mode still clips to [0, 1] before per-row normalization."""
        model = XGBoostDeconvolver(output_transform="none")
        raw = np.array([[-1.0, 2.0, 10.0], [0.0, 0.0, 0.0]])

        out = model._transform_output(raw)

        np.testing.assert_allclose(out[0], np.array([0.0, 0.5, 0.5]))
        np.testing.assert_allclose(out[1], np.array([0.0, 0.0, 0.0]))

    def test_transform_output_clip_normalize(self):
        """clip_normalize should remove negatives and normalize to unit sum."""
        model = XGBoostDeconvolver(output_transform="clip_normalize")
        raw = np.array([[-3.0, 1.0, 3.0]])

        out = model._transform_output(raw)

        np.testing.assert_allclose(out, np.array([[0.0, 0.25, 0.75]]))

    def test_transform_output_softmax(self):
        """softmax mode should produce positive probabilities summing to one."""
        model = XGBoostDeconvolver(output_transform="softmax")
        raw = np.array([[1.0, 2.0, 3.0]])

        out = model._transform_output(raw)

        self.assertAlmostEqual(float(out.sum()), 1.0, places=7)
        self.assertTrue((out > 0).all())

    def test_transform_output_unknown_raises(self):
        """Unknown output transforms should raise a ValueError."""
        model = XGBoostDeconvolver(output_transform="invalid")

        with self.assertRaises(ValueError):
            model._transform_output(np.array([[1.0, 2.0]]))

    def test_predict_raises_when_not_fitted(self):
        """predict should guard against use before fit."""
        with self.assertRaises(RuntimeError):
            self.model.predict(self.X_3d)

    def test_predict_returns_single_sample_vector_for_2d_input(self):
        """A single 2D sample should return a 1D prediction vector."""
        self.model._is_fitted = True
        self.model.model = DummyMultiOutputModel(output=[1.0, 1.0, 2.0])

        pred = self.model.predict(self.X_3d[0])

        self.assertEqual(pred.shape, (self.n_cell_types,))
        self.assertAlmostEqual(float(pred.sum()), 1.0)

    def test_predict_returns_batch_for_3d_input(self):
        """A 3D batch should return one normalized row per input sample."""
        self.model._is_fitted = True
        self.model.model = DummyMultiOutputModel(output=[1.0, 1.0, 2.0])

        pred = self.model.predict(self.X_3d)

        self.assertEqual(pred.shape, (2, self.n_cell_types))
        np.testing.assert_allclose(pred.sum(axis=1), np.array([1.0, 1.0]))

    def test_get_feature_importance_raises_when_not_fitted(self):
        """Feature importance requires a fitted model."""
        with self.assertRaises(RuntimeError):
            self.model.get_feature_importance()

    def test_get_feature_importance_aggregate_true(self):
        """aggregate=True should average importances across output estimators."""
        self.model._is_fitted = True
        self.model.model = MagicMock()
        self.model.model.estimators_ = [
            DummyEstimator([1, 2, 3, 4, 5, 6]),
            DummyEstimator([2, 4, 6, 8, 10, 12]),
        ]

        out = self.model.get_feature_importance(aggregate=True)

        np.testing.assert_allclose(out["all"], np.array([1.5, 3, 4.5, 6, 7.5, 9]))
        np.testing.assert_allclose(out["diagonal"], np.array([1.5, 3, 4.5]))
        np.testing.assert_allclose(out["reject"], np.array([6, 7.5, 9]))

    def test_get_feature_importance_aggregate_false(self):
        """aggregate=False should preserve one importance vector per estimator."""
        self.model._is_fitted = True
        self.model.model = MagicMock()
        self.model.model.estimators_ = [
            DummyEstimator([1, 2, 3, 4, 5, 6]),
            DummyEstimator([2, 4, 6, 8, 10, 12]),
        ]

        out = self.model.get_feature_importance(aggregate=False)

        self.assertEqual(out["all"].shape, (2, 6))
        self.assertEqual(out["diagonal"].shape, (2, 3))
        self.assertEqual(out["reject"].shape, (2, 3))


class TestXGBoostDeconvolverFitEvaluateAndIO(unittest.TestCase):
    """Tests for fit/evaluate behavior and persistence helpers."""

    def setUp(self):
        """Build tiny synthetic train/validation fixtures for deterministic tests."""
        self.model = XGBoostDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=3,
            output_transform="clip_normalize",
        )
        self.X_train = np.ones((3, 3, 4), dtype=float)
        self.y_train = np.array(
            [[0.2, 0.3, 0.5], [0.3, 0.4, 0.3], [0.1, 0.7, 0.2]], dtype=float
        )
        self.X_val = np.ones((2, 3, 4), dtype=float)
        self.y_val = np.array([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]], dtype=float)

    @patch("methyldl.deconvolution.xgbdeconvolver.compute_combined_loss")
    @patch("methyldl.deconvolution.xgbdeconvolver.compute_deconvolution_metrics")
    def test_fit_with_validation_populates_train_and_val_history(
        self, mock_metrics, mock_loss
    ):
        """fit should populate both train and validation history when val data exists."""
        mock_metrics.side_effect = [
            {
                "mae": 0.1,
                "mse": 0.2,
                "kl": 0.3,
                "max_error": 0.4,
                "cosine_sim": 0.9,
            },
            {
                "mae": 0.11,
                "mse": 0.21,
                "kl": 0.31,
                "max_error": 0.41,
                "cosine_sim": 0.91,
            },
        ]
        mock_loss.side_effect = [0.5, 0.6]

        dummy = DummyMultiOutputModel(output=[0.2, 0.3, 0.5])
        with patch.object(self.model, "_build_model", return_value=dummy):
            fitted = self.model.fit(
                X=self.X_train,
                y=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                loss_weights={"mse_weight": 2.0, "kl_weight": 1.0},
                verbose=0,
            )

        self.assertIs(fitted, self.model)
        self.assertTrue(self.model._is_fitted)
        self.assertEqual(dummy.fit_calls, 1)
        self.assertEqual(self.model.history.train_loss, [0.5])
        self.assertEqual(self.model.history.val_loss, [0.6])
        self.assertEqual(self.model.history.train_metrics["mae"], [0.1])
        self.assertEqual(self.model.history.val_metrics["mae"], [0.11])

    @patch("methyldl.deconvolution.xgbdeconvolver.compute_combined_loss")
    @patch("methyldl.deconvolution.xgbdeconvolver.compute_deconvolution_metrics")
    def test_fit_without_validation_only_populates_train_history(
        self, mock_metrics, mock_loss
    ):
        """fit should leave validation history empty when no val split is given."""
        mock_metrics.return_value = {
            "mae": 0.1,
            "mse": 0.2,
            "kl": 0.3,
            "max_error": 0.4,
            "cosine_sim": 0.9,
        }
        mock_loss.return_value = 0.5

        dummy = DummyMultiOutputModel(output=[0.2, 0.3, 0.5])
        with patch.object(self.model, "_build_model", return_value=dummy):
            self.model.fit(X=self.X_train, y=self.y_train, verbose=1)

        self.assertEqual(self.model.history.train_loss, [0.5])
        self.assertEqual(self.model.history.val_loss, [])

    @patch("methyldl.deconvolution.xgbdeconvolver.compute_combined_loss")
    @patch("methyldl.deconvolution.xgbdeconvolver.compute_deconvolution_metrics")
    def test_evaluate_with_explicit_loss_weights(self, mock_metrics, mock_loss):
        """evaluate should merge computed metrics with combined loss."""
        self.model._is_fitted = True
        self.model.model = DummyMultiOutputModel(output=[0.2, 0.3, 0.5])
        mock_metrics.return_value = {
            "mae": 0.1,
            "mse": 0.2,
            "kl": 0.3,
            "max_error": 0.4,
            "cosine_sim": 0.9,
        }
        mock_loss.return_value = 0.5

        metrics = self.model.evaluate(
            self.X_train,
            self.y_train,
            loss_weights={"mse_weight": 1.0, "kl_weight": 2.0},
        )

        self.assertIn("loss", metrics)
        self.assertEqual(metrics["loss"], 0.5)

    def test_evaluate_default_loss_weights_path_raises_type_error(self):
        """Document current default-loss-weights mismatch behavior in evaluate."""
        self.model._is_fitted = True
        self.model.model = DummyMultiOutputModel(output=[0.2, 0.3, 0.5])

        with self.assertRaises(TypeError):
            self.model.evaluate(self.X_train, self.y_train)

    def test_save_and_load_round_trip(self):
        """save/load should round-trip a fitted model and its history."""
        self.model._is_fitted = True
        self.model.model = DummyMultiOutputModel(output=[0.2, 0.3, 0.5])
        self.model.history = DeconvolutionHistory(train_loss=[0.1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.joblib")
            self.model.save(path)
            loaded = XGBoostDeconvolver.load(path)

        self.assertIsInstance(loaded, XGBoostDeconvolver)
        self.assertTrue(loaded._is_fitted)
        self.assertEqual(loaded.history.train_loss, [0.1])

    def test_load_raises_file_not_found(self):
        """load should raise FileNotFoundError for missing model files."""
        with self.assertRaises(FileNotFoundError):
            XGBoostDeconvolver.load("/definitely/not/present/model.joblib")

    def test_load_raises_type_error_for_wrong_object(self):
        """load should raise TypeError when the file does not contain this model class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "not_model.joblib")

            import joblib

            joblib.dump({"not": "a model"}, path)

            with self.assertRaises(TypeError):
                XGBoostDeconvolver.load(path)


class TestTrainXGBDeconvolverConvenience(unittest.TestCase):
    """Tests for the train_xgb_deconvolver convenience wrapper."""

    @patch(
        "methyldl.deconvolution.xgbdeconvolver.XGBoostDeconvolver.fit", autospec=True
    )
    def test_train_convenience_uses_default_config_and_returns_history(self, mock_fit):
        """Convenience function should build default config and return model history."""
        X_train = np.ones((2, 39, 40), dtype=float)
        y_train = np.ones((2, 39), dtype=float) / 39.0
        X_val = np.ones((1, 39, 40), dtype=float)
        y_val = np.ones((1, 39), dtype=float) / 39.0

        def fake_fit(self, *args, **kwargs):
            # autospec=True passes model instance as first argument.
            self.history = DeconvolutionHistory(train_loss=[0.2], val_loss=[0.25])
            self._is_fitted = True
            return self

        mock_fit.side_effect = fake_fit

        model, history = train_xgb_deconvolver(
            X_train,
            y_train,
            X_val,
            y_val,
            config=None,
            output_transform="softmax",
            verbose=0,
        )

        self.assertIsInstance(model, XGBoostDeconvolver)
        self.assertIsInstance(history, DeconvolutionHistory)
        self.assertEqual(history.train_loss, [0.2])

    @patch(
        "methyldl.deconvolution.xgbdeconvolver.XGBoostDeconvolver.fit", autospec=True
    )
    def test_train_convenience_respects_provided_config(self, mock_fit):
        """Convenience function should preserve explicitly provided config values."""
        X_train = np.ones((2, 39, 40), dtype=float)
        y_train = np.ones((2, 39), dtype=float) / 39.0
        X_val = np.ones((1, 39, 40), dtype=float)
        y_val = np.ones((1, 39), dtype=float) / 39.0

        cfg = XGBDeconvolverConfig(early_stopping_rounds=99)

        def fake_fit(self, *args, **kwargs):
            # Keep this lightweight: we only need to emulate history population.
            self.history = DeconvolutionHistory(train_loss=[0.3])
            return self

        mock_fit.side_effect = fake_fit

        model, _ = train_xgb_deconvolver(
            X_train,
            y_train,
            X_val,
            y_val,
            config=cfg,
            verbose=0,
        )

        self.assertEqual(model.config.early_stopping_rounds, 99)


if __name__ == "__main__":
    unittest.main()
