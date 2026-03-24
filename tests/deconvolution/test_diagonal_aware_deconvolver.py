import unittest

import torch

from methyldl.deconvolution.deep_deconvolvers.diagonal_aware_deconvolver import (
    DiagonalAwareDeconvolver,
)
from methyldl.deconvolution.deep_deconvolvers.training import DeconvolverOutput


class TestDiagonalAwareDeconvolver(unittest.TestCase):
    """Tests for the pathway-aware deconvolver."""

    def setUp(self):
        """Create small deterministic fixtures that expose diagonal, confusion, and reject paths."""
        torch.manual_seed(23)
        self.inputs = torch.tensor(
            [
                [[1.0, 0.2, 0.3, 0.9], [0.4, 2.0, 0.6, 0.8], [0.7, 0.8, 3.0, 0.7]],
                [[1.5, 0.1, 0.4, 0.3], [0.2, 2.5, 0.5, 0.2], [0.6, 0.9, 3.5, 0.1]],
            ],
            dtype=torch.float32,
        )

    def test_init_requires_at_least_one_enabled_pathway(self):
        """Disabling all feature pathways should fail fast with a clear error."""
        with self.assertRaisesRegex(ValueError, "At least one encoder pathway"):
            DiagonalAwareDeconvolver(
                use_diagonal=False,
                use_confusion=False,
                use_reject=False,
            )

    def test_config_and_repr_reflect_enabled_pathways(self):
        """The model metadata helpers should expose configuration and enabled-pathway information."""
        model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
            use_diagonal=True,
            use_confusion=False,
            use_reject=True,
        )

        config = model.config
        rendered = repr(model)

        self.assertEqual(config["combined_dim"], 9)
        self.assertEqual(config["n_dmr_groups"], 3)
        self.assertEqual(config["n_cell_types"], 2)
        self.assertGreater(config["n_parameters"], 0)
        self.assertIn("diagonal", rendered)
        self.assertIn("reject", rendered)
        self.assertNotIn("confusion", rendered)

    def test_extract_input_features_returns_expected_views(self):
        """Feature extraction should separate diagonal, reject-column, and off-diagonal inputs."""
        model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
        )

        features = model.extract_input_features(self.inputs)

        self.assertEqual(
            set(features),
            {"full_matrix", "diagonal", "reject_col", "off_diagonal"},
        )
        torch.testing.assert_close(
            features["diagonal"],
            torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            features["reject_col"],
            torch.tensor([[0.9, 0.8, 0.7], [0.3, 0.2, 0.1]], dtype=torch.float32),
        )
        # The off-diagonal branch clones the matrix and zeros the diagonal entries only.
        self.assertTrue(
            torch.all(
                torch.diagonal(features["off_diagonal"], dim1=1, dim2=2) == 0
            ).item()
        )

    def test_encode_features_handles_disabled_paths_and_ablation_flags(self):
        """Encoding should return None for disabled paths and zeros for ablated enabled paths."""
        full_model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
        )
        full_model.eval()

        diag, confusion, reject = full_model.encode_features(
            self.inputs,
            zero_diagonal=True,
            zero_confusion=True,
            zero_reject=True,
        )

        self.assertEqual(diag.shape, (2, 6))
        self.assertEqual(confusion.shape, (2, 6))
        self.assertEqual(reject.shape, (2, 3))
        self.assertEqual(torch.count_nonzero(diag).item(), 0)
        self.assertEqual(torch.count_nonzero(confusion).item(), 0)
        self.assertEqual(torch.count_nonzero(reject).item(), 0)

        partial_model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
            use_diagonal=False,
            use_confusion=True,
            use_reject=False,
        )
        partial_model.eval()

        diag, confusion, reject = partial_model.encode_features(self.inputs)

        self.assertIsNone(diag)
        self.assertEqual(confusion.shape, (2, 6))
        self.assertIsNone(reject)

    def test_forward_can_return_tensor_or_structured_output(self):
        """The forward path should support both plain predictions and feature-rich outputs."""
        model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
        )
        model.eval()

        plain_output = model(self.inputs)
        structured_output = model(self.inputs, return_features=True)

        self.assertEqual(plain_output.shape, (2, 2))
        self.assertIsInstance(structured_output, DeconvolverOutput)
        self.assertEqual(structured_output.logits.shape, (2, 2))
        self.assertEqual(structured_output.combined_features.shape, (2, 15))
        torch.testing.assert_close(plain_output, structured_output.proportions)
        torch.testing.assert_close(plain_output.sum(dim=-1), torch.ones(2))

    def test_ablation_study_includes_only_valid_conditions_for_each_configuration(self):
        """Ablation outputs should depend on which pathways are enabled in the model."""
        full_model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
        )
        full_model.eval()

        full_results = full_model.ablation_study(self.inputs)

        self.assertEqual(
            set(full_results),
            {
                "full_model",
                "no_diagonal",
                "no_confusion",
                "no_reject",
                "only_diagonal",
                "only_confusion",
                "only_reject",
            },
        )
        for value in full_results.values():
            self.assertEqual(value.shape, (2, 2))

        single_path_model = DiagonalAwareDeconvolver(
            n_dmr_groups=3,
            n_pred_classes=4,
            n_cell_types=2,
            hidden_dim=6,
            use_diagonal=True,
            use_confusion=False,
            use_reject=False,
        )
        single_path_model.eval()

        single_path_results = single_path_model.ablation_study(self.inputs)

        self.assertEqual(set(single_path_results), {"full_model", "no_diagonal"})
