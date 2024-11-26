import unittest
from methyldl.data.dataset import generate_example_data

class TestGenerateExampleData(unittest.TestCase):
    
    def test_default_parameters(self):
        """Test with default parameters."""
        result = generate_example_data()
        self.assertEqual(len(result), 2)  # Header + 1 sample
        self.assertEqual(result[0], ["input_ids"])
        self.assertEqual(len(result[1][0]), 150)  # Sequence length should match default
    
    def test_include_cpg_methylation(self):
        """Test with cpg_methylation included."""
        result = generate_example_data(include_cpg_methylation=True)
        self.assertEqual(result[0], ["input_ids", "cpg_methylation_sequence"])
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # cpg_methylation sequence
    
    def test_include_m6a_methylation(self):
        """Test with m6a_methylation included."""
        result = generate_example_data(include_m6a_methylation=True)
        self.assertEqual(result[0], ["input_ids", "m6a_methylation_sequence"])
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # m6a_methylation sequence
    
    def test_include_labels(self):
        """Test with labels included."""
        result = generate_example_data(include_labels=True)
        self.assertEqual(result[0], ["input_ids", "label"])
        self.assertEqual(result[1][1], 0)  # Label is set to 0
    
    def test_multiple_samples(self):
        """Test with multiple samples."""
        num_samples = 5
        result = generate_example_data(num_samples=num_samples)
        self.assertEqual(len(result), num_samples + 1)  # Header + num_samples
    
    def test_all_features_included(self):
        """Test with all features included."""
        result = generate_example_data(include_cpg_methylation=True,
                                       include_m6a_methylation=True,
                                       include_labels=True)
        self.assertEqual(result[0], ["input_ids", "cpg_methylation_sequence", "m6a_methylation_sequence", "label"])
        self.assertEqual(len(result[1][0]), 150)  # Genome sequence length
        self.assertEqual(result[1][1], "2" * 150)  # cpg_methylation sequence
        self.assertEqual(result[1][2], "2" * 150)  # m6a_methylation sequence
        self.assertEqual(result[1][3], 0)  # Label is set to 0
    
    def test_custom_sequence_length(self):
        """Test with a custom sequence length."""
        sequence_length = 300
        result = generate_example_data(sequence_length=sequence_length)
        self.assertEqual(len(result[1][0]), sequence_length)  # Sequence length should match custom value
    
    def test_num_samples_less_than_one(self):
        with self.assertRaises(ValueError) as context:
            generate_example_data(num_samples=0)
        self.assertEqual(str(context.exception), "The number of samples (num_samples) must be at least 1.")
    
    def test_sequence_length_less_than_thirty(self):
        with self.assertRaises(ValueError) as context:
            generate_example_data(sequence_length=20)
        self.assertEqual(str(context.exception), "The sequence length (sequence_length) must be at least 30.")


if __name__ == "__main__":
    unittest.main()