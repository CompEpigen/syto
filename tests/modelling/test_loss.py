import unittest
import torch
import torch.nn.functional as F

from methyldl.modelling.loss import (
    ConfidenceWeightedCrossEntropy,
    FocalLoss,
    sigmoid_focal_loss,
)


class TestConfidenceWeightedCrossEntropy(unittest.TestCase):
    """Test suite for ConfidenceWeightedCrossEntropy loss."""

    def setUp(self):
        self.num_classes = 5
        self.loss_fn = ConfidenceWeightedCrossEntropy(num_classes=self.num_classes)

    def test_output_is_scalar(self):
        """Verify forward returns a scalar tensor."""
        logits = torch.randn(4, self.num_classes)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        loss = self.loss_fn(logits, targets)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_sharp_target_gets_higher_weight(self):
        """A one-hot target should receive weight ~1.0, a flat target ~0.01."""
        # Sharp (one-hot) target
        sharp_target = torch.zeros(1, self.num_classes)
        sharp_target[0, 0] = 1.0

        # Flat (uniform) target
        flat_target = torch.ones(1, self.num_classes) / self.num_classes

        # Compute weights the same way the loss does
        sharp_max = sharp_target.max(dim=1)[0]
        flat_max = flat_target.max(dim=1)[0]
        min_prob = 1.0 / self.num_classes

        sharp_weight = (sharp_max - min_prob) / (1.0 - min_prob)
        flat_weight = torch.clamp((flat_max - min_prob) / (1.0 - min_prob), min=0.01)

        self.assertAlmostEqual(sharp_weight.item(), 1.0, places=5)
        self.assertAlmostEqual(flat_weight.item(), 0.01, places=5)

    def test_custom_num_classes(self):
        """Verify loss works with different num_classes values."""
        for nc in [2, 10, 39]:
            loss_fn = ConfidenceWeightedCrossEntropy(num_classes=nc)
            logits = torch.randn(3, nc)
            targets = F.softmax(torch.randn(3, nc), dim=-1)
            loss = loss_fn(logits, targets)
            self.assertFalse(torch.isnan(loss))
            self.assertGreater(loss.item(), 0)

    def test_gradients_flow(self):
        """Verify backward() works and logits receive gradients."""
        logits = torch.randn(4, self.num_classes, requires_grad=True)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        loss = self.loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.all(logits.grad == 0))

    def test_loss_nonnegative(self):
        """Verify loss is always non-negative."""
        logits = torch.randn(8, self.num_classes)
        targets = F.softmax(torch.randn(8, self.num_classes), dim=-1)
        loss = self.loss_fn(logits, targets)
        self.assertGreaterEqual(loss.item(), 0)

    def test_penalty_scale_zero_equals_plain_ce(self):
        """With penalty_scale=0.0, all weights are 1.0, so loss equals plain CE."""
        loss_fn = ConfidenceWeightedCrossEntropy(
            num_classes=self.num_classes, penalty_scale=0.0
        )
        logits = torch.randn(4, self.num_classes)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        loss = loss_fn(logits, targets)
        plain_ce = F.cross_entropy(logits, targets)
        self.assertAlmostEqual(loss.item(), plain_ce.item(), places=5)

    def test_penalty_scale_intermediate_weights(self):
        """With penalty_scale=0.5, a flat target gets weight 0.5 (not clamped)."""
        loss_fn = ConfidenceWeightedCrossEntropy(
            num_classes=self.num_classes, penalty_scale=0.5
        )
        logits = torch.randn(1, self.num_classes)
        flat_target = torch.ones(1, self.num_classes) / self.num_classes

        loss = loss_fn(logits, flat_target)
        plain_ce = F.cross_entropy(logits, flat_target)

        # raw_weight for flat = 0; scaled = (1-0.5)*1 + 0.5*0 = 0.5; clamp(0.5) = 0.5
        expected = plain_ce * 0.5
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_penalty_scale_invalid_raises(self):
        """penalty_scale outside [0, 1] should raise AssertionError."""
        with self.assertRaises(AssertionError):
            ConfidenceWeightedCrossEntropy(penalty_scale=1.5)
        with self.assertRaises(AssertionError):
            ConfidenceWeightedCrossEntropy(penalty_scale=-0.1)


class TestOnTargetWeight(unittest.TestCase):
    """Tests for the on_target_weight functionality of ConfidenceWeightedCrossEntropy."""

    def setUp(self):
        self.num_classes = 5
        self.loss_fn = ConfidenceWeightedCrossEntropy(
            num_classes=self.num_classes, on_target_weight=1.0
        )

    def test_output_is_scalar(self):
        """Verify forward returns a scalar tensor."""
        logits = torch.randn(4, self.num_classes)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        is_on_target = torch.tensor([True, False, True, False])
        loss = self.loss_fn(logits, targets, is_on_target)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_on_target_reads_have_weight_one(self):
        """On-target reads with on_target_weight=1.0 should equal plain CE for a fully on-target batch."""
        logits = torch.randn(2, self.num_classes)
        flat_target = torch.ones(2, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([True, True])

        on_target_loss = self.loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)
        self.assertAlmostEqual(on_target_loss.item(), plain_ce.item(), places=5)

    def test_off_target_flat_reads_penalized(self):
        """Flat off-target reads should get weight ≈ 0.01 (clamped confidence weight)."""
        logits = torch.randn(1, self.num_classes)
        flat_target = torch.ones(1, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([False])

        loss = self.loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)

        expected = plain_ce * 0.01
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)

    def test_off_target_sharp_reads_less_penalized(self):
        """Sharp off-target reads should get weight close to 1.0."""
        logits = torch.randn(1, self.num_classes)
        sharp_target = torch.zeros(1, self.num_classes)
        sharp_target[0, 0] = 1.0
        is_on_target = torch.tensor([False])

        loss = self.loss_fn(logits, sharp_target, is_on_target)
        plain_ce = F.cross_entropy(logits, sharp_target)

        self.assertAlmostEqual(loss.item(), plain_ce.item(), places=4)

    def test_is_on_target_without_weight_raises(self):
        """Providing is_on_target without setting on_target_weight should raise AssertionError."""
        loss_fn = ConfidenceWeightedCrossEntropy(num_classes=self.num_classes)
        logits = torch.randn(2, self.num_classes)
        targets = F.softmax(torch.randn(2, self.num_classes), dim=-1)
        is_on_target = torch.tensor([True, False])
        with self.assertRaises(AssertionError):
            loss_fn(logits, targets, is_on_target)

    def test_gradients_flow(self):
        """Verify backward() works and logits receive gradients."""
        logits = torch.randn(4, self.num_classes, requires_grad=True)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        is_on_target = torch.tensor([True, False, True, False])
        loss = self.loss_fn(logits, targets, is_on_target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.all(logits.grad == 0))

    def test_custom_on_target_weight(self):
        """Verify that a custom on_target_weight value is applied to on-target reads."""
        loss_fn = ConfidenceWeightedCrossEntropy(
            num_classes=self.num_classes, on_target_weight=2.0
        )
        logits = torch.randn(2, self.num_classes)
        flat_target = torch.ones(2, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([True, True])

        loss = loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)

        expected = plain_ce * 2.0
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)


class TestFocalLoss(unittest.TestCase):
    """Test suite for Focal Loss implementation."""

    def test_focal_loss_basic(self):
        """Test basic focal loss computation."""
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()

        loss = sigmoid_focal_loss(inputs, targets, reduction="mean")

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        self.assertGreaterEqual(loss.item(), 0)

    def test_focal_loss_reduction_modes(self):
        """Test different reduction modes."""
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()

        loss_none = sigmoid_focal_loss(inputs, targets, reduction="none")
        loss_mean = sigmoid_focal_loss(inputs, targets, reduction="mean")
        loss_sum = sigmoid_focal_loss(inputs, targets, reduction="sum")

        self.assertEqual(loss_none.shape, inputs.shape)
        self.assertEqual(loss_mean.shape, torch.Size([]))
        self.assertEqual(loss_sum.shape, torch.Size([]))

    def test_focal_loss_class(self):
        """Test FocalLoss class wrapper."""
        criterion = FocalLoss(reduction="mean")

        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10, 2)).float()

        loss = criterion(inputs, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
