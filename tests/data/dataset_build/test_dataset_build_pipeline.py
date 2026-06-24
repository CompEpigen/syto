import json
import logging
import unittest
from pathlib import Path
import tempfile

import pandas as pd
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from App.dataset_build_pipeline import DatasetBuildPipeline


def _write_atlas(path):
    df = pd.DataFrame(
        {
            "chr": ["chr1", "chr1"],
            "start": [1001, 2001],
            "end": [1010, 2010],
            "startCpG": [10, 20],
            "endCpG": [12, 22],
            "target": ["Adipocytes", "Gallbladder"],
            "name": ["chr1:1001-1010", "chr1:2001-2010"],
            "direction": ["U", "U"],
        }
    )
    for c in UXMMethylationAtlas.EXPECTED_CTYPE_COLUMNS:
        df[c] = 0.0
    df.to_csv(path, sep="\t", index=False)


class TestDatasetBuildPipeline(unittest.TestCase):
    def test_run_all_produces_final_parquet(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_atlas(d / "atlas.tsv")
            (d / "labels.json").write_text(
                json.dumps({"0": "Adipocytes", "1": "Gallbladder"})
            )
            inp = d / "in"
            inp.mkdir()
            reads = pd.DataFrame(
                {
                    "ref_name": ["chr1"] * 6,
                    "ref_pos": [1000] * 6,
                    "original_seq": ["ACGTACGTAC"] * 6,
                    "methyl_seq": ["0101010101"] * 6,
                    "ctype": ["Adipocytes"] * 6,
                }
            )
            for s in ["s1", "s2", "s3"]:
                reads.to_csv(inp / f"{s}.csv", sep="\t", index=False)
            cfg = {
                "phase": "all",
                "input_dir": str(inp),
                "output_dir": str(d / "out"),
                "atlas_path": str(d / "atlas.tsv"),
                "reference_genome": "hg38",
                "labels_dict_path": str(d / "labels.json"),
                "n_buckets": 2,
                "n_workers": 1,
                "min_reads": 1,
                "max_distance": 0.7,
                "seed": 42,
            }
            summary = DatasetBuildPipeline(cfg, logging.getLogger("t")).run()
            finals = list((d / "out" / "final").rglob("*.parquet"))
            self.assertTrue(finals)
            out = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            for col in ["soft_label", "label", "split", "name", "region_bucket"]:
                self.assertIn(col, out.columns)
            self.assertTrue(set(out["split"]).issubset({"train", "valid", "test"}))
