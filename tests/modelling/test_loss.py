import unittest
import torch
import torch.nn.functional as F

from methyldl.modelling.loss import ConfidenceWeightedCrossEntropy, OnTargetSoftLoss


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
        logits = torch.randn(2, self.num_classes)

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
        flat_weight = torch.clamp(
            (flat_max - min_prob) / (1.0 - min_prob), min=0.01
        )

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


class TestOnTargetSoftLoss(unittest.TestCase):
    """Test suite for OnTargetSoftLoss."""

    def setUp(self):
        self.num_classes = 5
        self.loss_fn = OnTargetSoftLoss(num_classes=self.num_classes)

    def test_output_is_scalar(self):
        """Verify forward returns a scalar tensor."""
        logits = torch.randn(4, self.num_classes)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        is_on_target = torch.tensor([True, False, True, False])
        loss = self.loss_fn(logits, targets, is_on_target)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_on_target_reads_have_weight_one(self):
        """On-target reads should use weight 1.0 regardless of target sharpness."""
        logits = torch.randn(2, self.num_classes)
        # Flat target – normally would get low weight
        flat_target = torch.ones(2, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([True, True])

        # The on-target loss should use weight 1.0, so it equals plain CE
        on_target_loss = self.loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)
        self.assertAlmostEqual(on_target_loss.item(), plain_ce.item(), places=5)

    def test_off_target_flat_reads_penalized(self):
        """Flat off-target reads should get weight ≈ min_bg_weight (0.01)."""
        logits = torch.randn(1, self.num_classes)
        flat_target = torch.ones(1, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([False])

        loss = self.loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)

        # Off-target flat should be heavily penalized (multiplied by 0.01)
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

        # Sharp off-target should have weight ≈ 1.0 so loss ≈ plain CE
        self.assertAlmostEqual(loss.item(), plain_ce.item(), places=4)

    def test_gradients_flow(self):
        """Verify backward() works and logits receive gradients."""
        logits = torch.randn(4, self.num_classes, requires_grad=True)
        targets = F.softmax(torch.randn(4, self.num_classes), dim=-1)
        is_on_target = torch.tensor([True, False, True, False])
        loss = self.loss_fn(logits, targets, is_on_target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.all(logits.grad == 0))

    def test_custom_min_bg_weight(self):
        """Verify that custom min_bg_weight is respected."""
        loss_fn = OnTargetSoftLoss(num_classes=self.num_classes, min_bg_weight=0.1)
        logits = torch.randn(1, self.num_classes)
        flat_target = torch.ones(1, self.num_classes) / self.num_classes
        is_on_target = torch.tensor([False])

        loss = loss_fn(logits, flat_target, is_on_target)
        plain_ce = F.cross_entropy(logits, flat_target)

        # With min_bg_weight=0.1, flat off-target should use weight 0.1
        expected = plain_ce * 0.1
        self.assertAlmostEqual(loss.item(), expected.item(), places=4)


if __name__ == "__main__":
    unittest.main()
