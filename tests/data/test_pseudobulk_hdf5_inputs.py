"""Chunked writing of the input DataFrames into the consolidated pseudobulk HDF5."""

import logging
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from syto.data.pseudobulk_hdf5_utils import (
    GenerationMetadata,
    PseudobulkHDF5ConsolidationWriter,
)


def _df(n=11):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "input_ids": [f"ACGT{i}" for i in range(n)],
            "original_label": rng.integers(0, 39, n),
            "prediction_0": rng.random(n),
            "name": [f"chr1:{i}" for i in range(n)],
            "read_count": rng.integers(1, 9, n),
        }
    )


def _write(chunk_rows, path, df):
    writer = PseudobulkHDF5ConsolidationWriter(path, logging.getLogger("t"))
    writer.INPUT_WRITE_CHUNK_ROWS = chunk_rows
    metadata = GenerationMetadata(
        grg_id_column="dmr_ctype_label",
        labeling_scheme="Soft Labels",
        classifier="dismir",
        data_watermark="w",
        data_stats={"train": {"n_reads": 1}},
    )
    with h5py.File(path, "w") as f:
        writer._write_inputs(f, metadata, {"train": df})
    with h5py.File(path, "r") as f:
        g = f["inputs/train"]
        return (
            g["numeric_data"][()],
            g["string_data"].asstr()[()],
            {k: (list(v) if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in g.attrs.items()},
            g["numeric_data"].dtype,
        )


class TestChunkedInputWriting(unittest.TestCase):
    def test_chunked_equals_single_write(self):
        df = _df()
        with tempfile.TemporaryDirectory() as d:
            whole = _write(10**9, Path(d) / "a.h5", df)
            chunked = _write(3, Path(d) / "b.h5", df)  # 11 rows -> chunks of 3,3,3,2
        np.testing.assert_array_equal(whole[0], chunked[0])
        np.testing.assert_array_equal(whole[1], chunked[1])
        self.assertEqual(whole[2], chunked[2])
        self.assertEqual(whole[3], chunked[3])

    def test_content_matches_the_frame(self):
        df = _df()
        with tempfile.TemporaryDirectory() as d:
            numeric, strings, attrs, dtype = _write(4, Path(d) / "c.h5", df)
        num_cols = ["original_label", "prediction_0", "read_count"]
        self.assertEqual(attrs["numeric_columns"], num_cols)
        self.assertEqual(dtype, np.float64)  # mixed int/float upcast, as df.values does
        np.testing.assert_allclose(numeric, df[num_cols].to_numpy(dtype=float))
        np.testing.assert_array_equal(strings, df[["input_ids", "name"]].to_numpy())
        self.assertEqual(attrs["n_rows"], len(df))


if __name__ == "__main__":
    unittest.main()
