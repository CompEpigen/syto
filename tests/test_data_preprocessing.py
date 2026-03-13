import os
import tempfile
import unittest
from parameterized import parameterized

from methyldl.data.sequencing.genome import pretrain_data_preprocess


class TestPretrainDataPreprocess(unittest.TestCase):
    """Test suite for pretrain_data_preprocess function."""

    def test_basic_preprocessing(self):
        """Test basic preprocessing functionality."""
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCGATCGATCGATCG\n")
            input_file = f.name

        # Create temporary output file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            # Run preprocessing
            pretrain_data_preprocess(
                f_ref=input_file, k=3, seq_len=12, f_output=output_file, num_cores=1
            )

            # Check output file exists and has content
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r") as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0)

                # Check that lines contain k-mers
                for line in lines:
                    kmers = line.strip().split()
                    for kmer in kmers:
                        self.assertEqual(len(kmer), 3)

        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_multiple_chromosomes(self):
        """Test with multiple chromosomes."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCG\n")
            f.write(">CHR2\n")
            f.write("GCTAGCTAGCTA\n")
            input_file = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            pretrain_data_preprocess(
                f_ref=input_file, k=3, seq_len=12, f_output=output_file, num_cores=1
            )

            with open(output_file, "r") as f:
                lines = f.readlines()
                # Should have sequences from both chromosomes
                self.assertGreaterEqual(len(lines), 2)

        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_skip_invalid_chromosome(self):
        """Test that invalid chromosomes are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCG\n")
            f.write(">CHRMT\n")  # Mitochondrial - should be skipped
            f.write("GCTAGCTAGCTA\n")
            input_file = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            pretrain_data_preprocess(
                f_ref=input_file, k=3, seq_len=12, f_output=output_file, num_cores=1
            )

            with open(output_file, "r") as f:
                lines = f.readlines()
                # Should only have sequences from CHR1
                self.assertEqual(len(lines), 1)

        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_default_output_filename(self):
        """Test that default output filename is generated correctly."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCG\n")
            input_file = f.name

        try:
            pretrain_data_preprocess(
                f_ref=input_file,
                k=3,
                seq_len=12,
                f_output=None,  # Let it generate default name
                num_cores=1,
            )

            # Check that output file was created with default name pattern
            expected_output = (
                input_file.replace("Raw", "Refined") + "_3mers_seqlen12.txt"
            )
            self.assertTrue(os.path.exists(expected_output))

            # Clean up
            if os.path.exists(expected_output):
                os.unlink(expected_output)

        finally:
            os.unlink(input_file)

    @parameterized.expand(
        [
            ("k2_len10", 2, 10),
            ("k3_len12", 3, 12),
            ("k4_len20", 4, 20),
        ]
    )
    def test_different_parameters(self, name, k, seq_len):
        """Test with different k and seq_len parameters."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCGATCGATCGATCG\n")
            input_file = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            pretrain_data_preprocess(
                f_ref=input_file,
                k=k,
                seq_len=seq_len,
                f_output=output_file,
                num_cores=1,
            )

            with open(output_file, "r") as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0)

                # Verify k-mer length
                for line in lines:
                    kmers = line.strip().split()
                    for kmer in kmers:
                        self.assertEqual(len(kmer), k)

        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_num_cores_validation(self):
        """Test that num_cores is validated against available CPUs."""
        import multiprocessing

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as f:
            f.write(">CHR1\n")
            f.write("ATCGATCGATCG\n")
            input_file = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            output_file = f.name

        try:
            # Try to use more cores than available
            available_cpus = multiprocessing.cpu_count()

            with self.assertRaises(AssertionError):
                pretrain_data_preprocess(
                    f_ref=input_file,
                    k=3,
                    seq_len=12,
                    f_output=output_file,
                    num_cores=available_cpus + 10,  # More than available
                )

        finally:
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)


if __name__ == "__main__":
    unittest.main()
