import unittest

from syto.data.dataset import (
    generate_example_data,
    generate_example_data_for_methylbert,
)


class TestGenerateExampleData(unittest.TestCase):
    """Tests for generic synthetic data generation."""

    def test_default_parameters(self):
        """Test with default parameters."""
        result = generate_example_data()
        self.assertEqual(len(result), 2)  # Header + 1 sample
        self.assertEqual(result[0], ["input_ids"])
        self.assertEqual(len(result[1][0]), 150)  # Sequence length should match default

    def test_include_cpg_methylation(self):
        """Test with cpg_methylation included."""
        result = generate_example_data(
            include_cpg_methylation=True, cpg_proportion={2: 1, 1: 0, 0: 0}
        )
        self.assertEqual(result[0], ["input_ids", "cpg_methylation_sequence"])
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # cpg_methylation sequence

    def test_include_m6a_methylation(self):
        """Test with m6a_methylation included."""
        result = generate_example_data(
            include_m6a_methylation=True, m6a_proportion={2: 1, 1: 0, 0: 0}
        )
        self.assertEqual(result[0], ["input_ids", "m6a_methylation_sequence"])
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # m6a_methylation sequence

    def test_include_labels(self):
        """Test with labels included."""
        result = generate_example_data(include_labels=True)
        self.assertEqual(result[0], ["input_ids", "label"])
        # Extract all labels (skip header)
        labels = [row[1] for row in result[1:]]

        # Check all labels are either 0 or 1
        self.assertTrue(
            all(label in [0, 1] for label in labels),
            "All labels should be either 0 or 1",
        )

    def test_multiple_samples(self):
        """Test with multiple samples."""
        num_samples = 5
        result = generate_example_data(num_samples=num_samples)
        self.assertEqual(len(result), num_samples + 1)  # Header + num_samples

    def test_all_features_included(self):
        """Test with all features included."""
        result = generate_example_data(
            include_cpg_methylation=True,
            include_m6a_methylation=True,
            include_labels=True,
            cpg_proportion={2: 1, 1: 0, 0: 0},
            m6a_proportion={2: 1, 1: 0, 0: 0},
        )
        self.assertEqual(
            result[0],
            [
                "input_ids",
                "cpg_methylation_sequence",
                "m6a_methylation_sequence",
                "label",
            ],
        )
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # cpg_methylation sequence
        self.assertEqual(result[1][2], "2" * 150)  # m6a_methylation sequence
        labels = [row[3] for row in result[1:]]

        # Check all labels are either 0 or 1
        self.assertTrue(
            all(label in [0, 1] for label in labels),
            "All labels should be either 0 or 1",
        )

    def test_custom_sequence_length(self):
        """Test with a custom sequence length."""
        sequence_length = 300
        result = generate_example_data(sequence_length=sequence_length)
        self.assertEqual(
            len(result[1][0]), sequence_length
        )  # Sequence length should match custom value

    def test_num_samples_less_than_one(self):
        """Ensure num_samples lower bound is enforced."""
        with self.assertRaises(ValueError) as context:
            generate_example_data(num_samples=0)
        self.assertEqual(
            str(context.exception),
            "The number of samples (num_samples) must be at least 1.",
        )

    def test_sequence_length_less_than_thirty(self):
        """Ensure sequence_length lower bound is enforced."""
        with self.assertRaises(ValueError) as context:
            generate_example_data(sequence_length=20)
        self.assertEqual(
            str(context.exception),
            "The sequence length (sequence_length) must be at least 30.",
        )


class TestGenerateExampleDataForMethylBert(unittest.TestCase):
    """Tests for MethylBERT-specific synthetic data wrapper."""

    def test_methylbert_output_shape_and_header(self):
        """Validate output header and transformed sample shapes."""
        result = generate_example_data_for_methylbert(sequence_length=30, num_samples=2)

        self.assertEqual(result[0], ["dna_seq", "methyl_seq", "ctype"])
        self.assertEqual(len(result), 3)  # header + 2 samples

        dna_tokens = result[1][0].split(" ")
        self.assertEqual(len(dna_tokens), 30)
        self.assertTrue(all(len(tok) == 3 for tok in dna_tokens))
        self.assertEqual(len(result[1][1]), 30)
        self.assertIn(result[1][2], [0, 1])

    def test_methylbert_long_sequence_splits_samples(self):
        """Verify long sequence requests are split into multiple samples."""
        result = generate_example_data_for_methylbert(
            sequence_length=1025, num_samples=2
        )

        # 1025 is split across 3 chunks -> 6 generated rows
        self.assertEqual(len(result), 7)  # header + 6 samples

        dna_tokens = result[1][0].split(" ")
        self.assertEqual(len(dna_tokens), 341)
        self.assertEqual(len(result[1][1]), 341)


if __name__ == "__main__":
    unittest.main()
