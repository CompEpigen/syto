"""
Tests for syto.cross_validation_engine.

Covers:
  - CrossValidationCompatibleModel (abstract interface)
  - CrossValidationEngine initialisation
  - CrossValidationEngine.fit() input validation and core behaviour
  - CrossValidationEngine.predict()
  - CrossValidationEngine.cv_metric_name / get_cv_metric()
  - CrossValidationEngine.save() / load() round-trip
"""

import unittest
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from syto.cross_validation_engine import (
    CrossValidationCompatibleModel,
    CrossValidationEngine,
)

# ==============================================================================
# Helpers / fixture objects
# ==============================================================================


def _make_dummy_probs(n_samples: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Return a random (n_samples, n_classes) probability matrix (rows sum to 1)."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n_samples, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


def _make_hard_labels(n_samples: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Return random integer class labels in [0, n_classes)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n_samples)


class DummyModel(CrossValidationCompatibleModel):
    """Minimal CrossValidationCompatibleModel for testing the engine.

    The cv metric equals ``1.0 / (quality + eps)``, so a higher ``quality``
    hyperparameter produces a lower (better) metric.  This lets tests verify
    that the engine selects the best hyperparameter combination.
    """

    def __init__(
        self,
        quality: float = 1.0,
        n_folds: int = 2,
    ):
        self.quality = quality
        self.n_folds = n_folds
        self._last_metric: float = float("inf")
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "DummyModel":
        """Store a reproducible metric that depends only on ``quality``."""
        self._last_metric = 1.0 / (self.quality + 1e-9)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Return uniform probability predictions."""
        n_samples, n_classes = X.shape
        return np.full((n_samples, n_classes), 1.0 / n_classes)

    def get_cv_metric(self, X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        return self._last_metric

    @property
    def cv_metric_name(self) -> str:
        return "dummy_metric"

    def save(self, path, **kwargs) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path, **kwargs) -> "DummyModel":
        import joblib

        return joblib.load(path)


# ==============================================================================
# Shared data used by multiple test classes
# ==============================================================================

_N_CLASSES = 3
_N_SAMPLES = 30

_shared_X: np.ndarray = _make_dummy_probs(_N_SAMPLES, _N_CLASSES, seed=0)
_shared_y: np.ndarray = _make_hard_labels(_N_SAMPLES, _N_CLASSES, seed=0)

_PARAM_GRID = {
    "quality": [1.0, 2.0],
    "n_folds": [2],
}
_FIT_KWARGS = dict(
    model_class=DummyModel,
    model_param_grid=_PARAM_GRID,
    n_folds=2,
    disable_pbar=True,
)


# ==============================================================================
# TestCrossValidationCompatibleModelABC
# ==============================================================================


class TestCrossValidationCompatibleModelABC(unittest.TestCase):
    """Tests for the CrossValidationCompatibleModel abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """CrossValidationCompatibleModel cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            CrossValidationCompatibleModel()  # pylint: disable=abstract-class-instantiated

    def test_concrete_subclass_is_instantiable(self):
        """A fully implemented subclass can be instantiated without error."""
        model = DummyModel()
        self.assertIsInstance(model, CrossValidationCompatibleModel)

    def test_concrete_subclass_has_all_interface_methods(self):
        """The concrete subclass exposes fit, predict, get_cv_metric, cv_metric_name, save, load."""
        model = DummyModel()
        for attr in (
            "fit",
            "predict",
            "get_cv_metric",
            "cv_metric_name",
            "save",
            "load",
        ):
            self.assertTrue(
                hasattr(model, attr),
                f"DummyModel is missing required attribute '{attr}'",
            )


# ==============================================================================
# TestCrossValidationEngineInit
# ==============================================================================


class TestCrossValidationEngineInit(unittest.TestCase):
    """Tests for CrossValidationEngine.__init__()."""

    def test_is_fitted_false_after_init(self):
        """is_fitted must be False before any call to fit()."""
        cv = CrossValidationEngine()
        self.assertFalse(cv.is_fitted)

    def test_predict_raises_before_fit(self):
        """predict() should raise RuntimeError when the engine has not been fitted."""
        cv = CrossValidationEngine()
        with self.assertRaises(RuntimeError):
            cv.predict(_make_dummy_probs(5, 3))

    def test_save_raises_before_fit(self):
        """save() should raise RuntimeError when the engine has not been fitted."""
        cv = CrossValidationEngine()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                cv.save(Path(td) / "model.joblib")


# ==============================================================================
# TestCrossValidationEngineFitAssertions
# ==============================================================================


class TestCrossValidationEngineFitAssertions(unittest.TestCase):
    """Tests for the input-validation assertions inside CrossValidationEngine.fit()."""

    def _base_fit_kwargs(self) -> dict:
        return dict(
            model_class=DummyModel,
            model_param_grid={"quality": [1.0], "n_folds": [2]},
            n_folds=2,
            disable_pbar=True,
        )

    def test_missing_model_class_raises(self):
        """fit() should raise AssertionError when model_class is not provided."""
        cv = CrossValidationEngine()
        kwargs = self._base_fit_kwargs()
        del kwargs["model_class"]
        with self.assertRaises(AssertionError):
            cv.fit(_shared_X, _shared_y, **kwargs)

    def test_missing_model_param_grid_raises(self):
        """fit() should raise AssertionError when model_param_grid is not provided."""
        cv = CrossValidationEngine()
        kwargs = self._base_fit_kwargs()
        del kwargs["model_param_grid"]
        with self.assertRaises(AssertionError):
            cv.fit(_shared_X, _shared_y, **kwargs)


# ==============================================================================
# TestCrossValidationEngineFit
# ==============================================================================


class TestCrossValidationEngineFit(unittest.TestCase):
    """Tests for CrossValidationEngine.fit() core behaviour."""

    @classmethod
    def setUpClass(cls):
        """Fit a CV engine once and reuse it across all tests in this class."""
        cls.cv = CrossValidationEngine()
        cls.cv.fit(_shared_X, _shared_y, **_FIT_KWARGS)

    def test_fit_returns_self(self):
        """fit() should return the CrossValidationEngine instance for method chaining."""
        cv = CrossValidationEngine()
        result = cv.fit(_shared_X, _shared_y, **_FIT_KWARGS)
        self.assertIs(result, cv)

    def test_is_fitted_true_after_fit(self):
        """is_fitted should be True after a successful call to fit()."""
        self.assertTrue(self.cv.is_fitted)

    def test_fit_populates_best_model(self):
        """best_model_ should be an instance of model_class after fit()."""
        self.assertIsNotNone(self.cv.best_model_)
        self.assertIsInstance(self.cv.best_model_, DummyModel)

    def test_fit_populates_best_params(self):
        """best_params_ should be a dict containing the grid-search keys after fit()."""
        self.assertIsNotNone(self.cv.best_params_)
        self.assertIsInstance(self.cv.best_params_, dict)
        self.assertIn("quality", self.cv.best_params_)

    def test_fit_selects_best_hyperparameters(self):
        """fit() should select the hyperparameter combination with the lowest mean CV metric.

        DummyModel metric = 1.0 / (quality + eps), so quality=2.0 yields the
        lowest metric and should be chosen.
        """
        self.assertEqual(self.cv.best_params_["quality"], 2.0)

    def test_fit_populates_best_metric(self):
        """best_metric_ should be a finite float after fit()."""
        self.assertIsNotNone(self.cv.best_metric_)
        self.assertIsInstance(self.cv.best_metric_, float)
        self.assertTrue(np.isfinite(self.cv.best_metric_))

    def test_fit_populates_metrics_per_param(self):
        """metrics_per_param_ should have one entry per hyperparameter combination."""
        self.assertIsNotNone(self.cv.metrics_per_param_)
        from sklearn.model_selection import ParameterGrid

        expected_n_combinations = len(list(ParameterGrid(_PARAM_GRID)))
        self.assertEqual(len(self.cv.metrics_per_param_), expected_n_combinations)

    def test_best_metric_matches_minimum_in_metrics_per_param(self):
        """best_metric_ should equal the minimum value in metrics_per_param_."""
        min_metric = min(self.cv.metrics_per_param_.values())
        self.assertAlmostEqual(self.cv.best_metric_, min_metric, places=12)


# ==============================================================================
# TestCrossValidationEnginePredict
# ==============================================================================


class TestCrossValidationEnginePredict(unittest.TestCase):
    """Tests for CrossValidationEngine.predict()."""

    @classmethod
    def setUpClass(cls):
        """Fit a minimal CV engine once for prediction tests."""
        cls.n_classes = _N_CLASSES
        cls.cv = CrossValidationEngine()
        cls.cv.fit(_shared_X, _shared_y, **_FIT_KWARGS)

    def test_predict_output_shape(self):
        """predict() should return an array of shape (n_samples, n_classes)."""
        # pylint: disable=invalid-name
        X_test = _make_dummy_probs(10, self.n_classes, seed=99)
        out = self.cv.predict(X_test)
        self.assertEqual(out.shape, (10, self.n_classes))

    def test_predict_delegates_to_best_model(self):
        """predict() result should match calling best_model_.predict() directly."""
        # pylint: disable=invalid-name
        X_test = _make_dummy_probs(10, self.n_classes, seed=99)
        engine_preds = self.cv.predict(X_test)
        direct_preds = self.cv.best_model_.predict(X_test)
        np.testing.assert_array_equal(engine_preds, direct_preds)

    def test_predict_unfitted_raises(self):
        """predict() before fit() should raise RuntimeError."""
        cv = CrossValidationEngine()
        with self.assertRaises(RuntimeError):
            cv.predict(_make_dummy_probs(5, self.n_classes))


# ==============================================================================
# TestCrossValidationEngineCVMetric
# ==============================================================================


class TestCrossValidationEngineCVMetric(unittest.TestCase):
    """Tests for CrossValidationEngine.cv_metric_name and get_cv_metric()."""

    @classmethod
    def setUpClass(cls):
        """Fit a CV engine once for metric tests."""
        cls.cv = CrossValidationEngine()
        cls.cv.fit(_shared_X, _shared_y, **_FIT_KWARGS)

    def test_cv_metric_name_not_fitted_returns_sentinel(self):
        """cv_metric_name before fit() should return the sentinel string 'Not fitted'."""
        cv = CrossValidationEngine()
        self.assertEqual(cv.cv_metric_name, "Not fitted")

    def test_cv_metric_name_after_fit_delegates_to_best_model(self):
        """cv_metric_name after fit() should delegate to best_model_.cv_metric_name."""
        self.assertEqual(self.cv.cv_metric_name, self.cv.best_model_.cv_metric_name)

    def test_get_cv_metric_returns_best_metric(self):
        """get_cv_metric() should return the best_metric_ found during grid search."""
        metric = self.cv.get_cv_metric(_shared_X, _shared_y)
        self.assertAlmostEqual(metric, self.cv.best_metric_, places=12)

    def test_get_cv_metric_is_finite(self):
        """get_cv_metric() should return a finite float."""
        metric = self.cv.get_cv_metric(_shared_X, _shared_y)
        self.assertTrue(np.isfinite(metric))


# ==============================================================================
# TestCrossValidationEngineSaveLoad
# ==============================================================================


class TestCrossValidationEngineSaveLoad(unittest.TestCase):
    """Tests for CrossValidationEngine.save() / load() round-trip."""

    @classmethod
    def setUpClass(cls):
        """Fit a minimal CV engine once for serialisation tests."""
        cls.cv = CrossValidationEngine()
        cls.cv.fit(_shared_X, _shared_y, **_FIT_KWARGS)

    def test_save_unfitted_raises(self):
        """save() before fit() should raise RuntimeError."""
        cv = CrossValidationEngine()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                cv.save(Path(td) / "model.joblib")

    def test_save_unfitted_engine_mode_raises(self):
        """save(mode='save-cv-engine') before fit() should raise RuntimeError."""
        cv = CrossValidationEngine()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                cv.save(Path(td) / "model.joblib", mode="save-cv-engine")

    def test_save_unknown_mode_raises(self):
        """save() with an unknown mode should raise ValueError."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.cv.save(Path(td) / "model.joblib", mode="bogus")

    # ── save-cv-model mode (default) ──────────────────────────────

    def test_default_mode_saves_only_the_model(self):
        """Default save() should persist only the cross-validated model."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.joblib"
            self.cv.save(path)
            self.assertTrue(path.exists())
            loaded = DummyModel.load(path)
            self.assertIsInstance(loaded, DummyModel)
            self.assertNotIsInstance(loaded, CrossValidationEngine)

    def test_default_mode_round_trip_predictions(self):
        """A model saved in default mode should reproduce the engine's predictions."""
        # pylint: disable=invalid-name
        X_test = _make_dummy_probs(10, _N_CLASSES, seed=99)
        original_preds = self.cv.predict(X_test)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.joblib"
            self.cv.save(path)
            loaded = DummyModel.load(path)

        np.testing.assert_array_equal(loaded.predict(X_test), original_preds)

    # ── save-cv-engine mode ───────────────────────────────────────

    def test_engine_mode_wrong_extension_raises(self):
        """save(mode='save-cv-engine') with a non-.joblib extension should raise."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(AssertionError):
                self.cv.save(Path(td) / "model.npz", mode="save-cv-engine")

    def test_load_wrong_extension_raises(self):
        """load() with a non-.joblib extension should raise AssertionError."""
        with self.assertRaises(AssertionError):
            CrossValidationEngine.load("model.npz")

    def test_engine_mode_creates_joblib_file(self):
        """save(mode='save-cv-engine') should create a .joblib file at the path."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            self.assertTrue(path.exists())

    def test_engine_mode_round_trip_predictions(self):
        """Predictions from a loaded engine should match the original engine's predictions."""
        # pylint: disable=invalid-name
        X_test = _make_dummy_probs(10, _N_CLASSES, seed=99)
        original_preds = self.cv.predict(X_test)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            loaded = CrossValidationEngine.load(path)

        np.testing.assert_array_equal(loaded.predict(X_test), original_preds)

    def test_engine_mode_restores_is_fitted(self):
        """load() should restore is_fitted=True."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            loaded = CrossValidationEngine.load(path)

        self.assertTrue(loaded.is_fitted)

    def test_engine_mode_restores_best_params(self):
        """load() should restore best_params_ identically."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            loaded = CrossValidationEngine.load(path)

        self.assertEqual(loaded.best_params_, self.cv.best_params_)

    def test_engine_mode_restores_best_metric(self):
        """load() should restore best_metric_ to the same value."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            loaded = CrossValidationEngine.load(path)

        self.assertAlmostEqual(loaded.best_metric_, self.cv.best_metric_, places=12)

    def test_engine_mode_restores_metrics_per_param(self):
        """load() should restore metrics_per_param_ with the same number of entries."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            loaded = CrossValidationEngine.load(path)

        self.assertEqual(
            len(loaded.metrics_per_param_), len(self.cv.metrics_per_param_)
        )

    def test_save_creates_parent_directories(self):
        """save() should create any missing parent directories."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "subdir" / "cv_engine.joblib"
            self.cv.save(path, mode="save-cv-engine")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
