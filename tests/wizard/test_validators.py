import os
import tempfile
import unittest

from syto.app.wizard import validators as v


class TestValidators(unittest.TestCase):
    def test_path_exists_accepts_existing_file(self):
        with tempfile.NamedTemporaryFile() as f:
            self.assertIsNone(v.path_exists(f.name))

    def test_path_exists_rejects_missing(self):
        msg = v.path_exists("/no/such/path/xyz")
        self.assertIsInstance(msg, str)
        self.assertIn("does not exist", msg)

    def test_path_exists_or_blank_accepts_blank(self):
        self.assertIsNone(v.path_exists_or_blank(""))
        self.assertIsNone(v.path_exists_or_blank("   "))

    def test_path_exists_or_blank_rejects_missing_nonblank(self):
        self.assertIsInstance(v.path_exists_or_blank("/no/such/path"), str)

    def test_positive_int_accepts_positive(self):
        self.assertIsNone(v.positive_int(39))

    def test_positive_int_rejects_zero_and_negative(self):
        self.assertIsInstance(v.positive_int(0), str)
        self.assertIsInstance(v.positive_int(-5), str)

    def test_non_negative_int_accepts_zero_and_positive(self):
        self.assertIsNone(v.non_negative_int(0))
        self.assertIsNone(v.non_negative_int(5))

    def test_non_negative_int_rejects_negative(self):
        self.assertIsInstance(v.non_negative_int(-1), str)

    def test_positive_float_accepts_positive(self):
        self.assertIsNone(v.positive_float(0.001))

    def test_positive_float_rejects_nonpositive(self):
        self.assertIsInstance(v.positive_float(0.0), str)

    def test_non_negative_float_accepts_zero_and_positive(self):
        self.assertIsNone(v.non_negative_float(0.0))
        self.assertIsNone(v.non_negative_float(0.01))

    def test_non_negative_float_rejects_negative(self):
        self.assertIsInstance(v.non_negative_float(-0.5), str)

    def test_non_empty_rejects_blank(self):
        self.assertIsInstance(v.non_empty("  "), str)
        self.assertIsNone(v.non_empty("x"))


if __name__ == "__main__":
    unittest.main()
