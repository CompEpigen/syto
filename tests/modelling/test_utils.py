import unittest

import torch.nn as nn

from methyldl.modelling.utils import *


class TestCountModelParameters(unittest.TestCase):

    def test_count_parameters(self):
        # Example model with known number of parameters
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
                self.layer2 = nn.Linear(5, 2)  # 5*2 + 2 = 12 parameters

        # .train() so that parameters are marked as requires_grad=True
        model = SimpleModel().train()
        total_params = count_model_parameters(model)
        expected_params = 55 + 12
        self.assertEqual(total_params, expected_params)


class TestCalculateBatchSize(unittest.TestCase):

    def test_calculate_batch_size_on_gpu(self):
        with unittest.mock.patch(
            "torch.cuda.is_available", return_value=True
        ), unittest.mock.patch("torch.cuda.get_device_properties") as mock_get_props:
            mock_get_props.return_value.total_memory = 8 * 1024**3  # 8 GB VRAM
            batch_size = calculate_batch_size(gb_per_seq=0.5, cpu_batch_size=16)
            expected_batch_size = int((8 * 0.85 / 0.5) // 10 * 10)
            self.assertEqual(batch_size, expected_batch_size)

    def test_calculate_batch_size_on_cpu(self):
        with unittest.mock.patch("torch.cuda.is_available", return_value=False):
            batch_size = calculate_batch_size(gb_per_seq=0.5, cpu_batch_size=16)
            self.assertEqual(batch_size, 16)


class TestExists(unittest.TestCase):

    def test_exists_with_none(self):
        self.assertFalse(exists(None))

    def test_exists_with_value(self):
        self.assertTrue(exists(0))
        self.assertTrue(exists(""))
        self.assertTrue(exists([]))
        self.assertTrue(exists({}))
        self.assertTrue(exists(set()))
