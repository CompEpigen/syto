import json
import logging
import unittest
from pathlib import Path
import tempfile

import pandas as pd
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas
from syto.app.dataset_build_pipeline import DatasetBuildPipeline


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
                "seed": 42,
                "signature": {
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                "labelers": {
                    "soft_label": {
                        "type": "data_driven_soft",
                        "distance_name": "jaccard",
                        "perform_pooling": True,
                        "min_reads": 1,
                        "max_distance": 0.7,
                    },
                    "label": {"type": "hard_with_background"},
                },
            }
            summary = DatasetBuildPipeline(cfg, logging.getLogger("t")).run()
            finals = list((d / "out" / "final").rglob("*.parquet"))
            self.assertTrue(finals)
            out = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            for col in ["soft_label", "label", "split", "name", "region_bucket"]:
                self.assertIn(col, out.columns)
            self.assertNotIn("signature", out.columns)
            self.assertTrue(set(out["split"]).issubset({"train", "valid", "test"}))


class TestDatasetBuildSplitRemap(unittest.TestCase):
    def test_split_remap_relabels_output_splits(self):
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
                "seed": 42,
                "split_remap": {"train": "train", "valid": "train", "test": "valid"},
                "signature": {
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                "labelers": {
                    "label": {"type": "hard_with_background"},
                },
            }
            DatasetBuildPipeline(cfg, logging.getLogger("t")).run()
            finals = list((d / "out" / "final").rglob("*.parquet"))
            self.assertTrue(finals)
            out = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            self.assertTrue(set(out["split"]).issubset({"train", "valid"}))
            self.assertNotIn("test", set(out["split"]))


class TestFinalizeAtlasSubset(unittest.TestCase):
    def test_finalize_atlas_filters_regions(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write_atlas(d / "atlas.tsv")  # regions chr1:1001-1010, chr1:2001-2010
            # subset atlas keeps only the first region
            sub = pd.read_csv(d / "atlas.tsv", sep="\t")
            sub[sub["name"] == "chr1:1001-1010"].to_csv(
                d / "atlas_sub.tsv", sep="\t", index=False
            )
            (d / "labels.json").write_text(
                json.dumps({"0": "Adipocytes", "1": "Gallbladder"})
            )
            inp = d / "in"
            inp.mkdir()
            # reads overlapping the FIRST region only (ref_pos 1000)
            r1 = pd.DataFrame(
                {
                    "ref_name": ["chr1"] * 4,
                    "ref_pos": [1000] * 4,
                    "original_seq": ["ACGTACGTAC"] * 4,
                    "methyl_seq": ["0101010101"] * 4,
                    "ctype": ["Adipocytes"] * 4,
                }
            )
            # reads overlapping the SECOND region (ref_pos 2000) -> filtered out
            r2 = pd.DataFrame(
                {
                    "ref_name": ["chr1"] * 4,
                    "ref_pos": [2000] * 4,
                    "original_seq": ["ACGTACGTAC"] * 4,
                    "methyl_seq": ["0101010101"] * 4,
                    "ctype": ["Gallbladder"] * 4,
                }
            )
            for s in ["s1", "s2", "s3"]:
                pd.concat([r1, r2]).to_csv(inp / f"{s}.csv", sep="\t", index=False)

            base = {
                "input_dir": str(inp),
                "output_dir": str(d / "out"),
                "atlas_path": str(d / "atlas.tsv"),
                "reference_genome": "hg38",
                "labels_dict_path": str(d / "labels.json"),
                "n_buckets": 2,
                "n_workers": 1,
                "seed": 42,
                "signature": {
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                "labelers": {"label": {"type": "hard_with_background"}},
            }
            DatasetBuildPipeline(
                {**base, "phase": "stage"}, logging.getLogger("t")
            ).run()
            DatasetBuildPipeline(
                {
                    **base,
                    "phase": "finalize",
                    "finalize_atlas_path": str(d / "atlas_sub.tsv"),
                },
                logging.getLogger("t"),
            ).run()

            finals = list((d / "out" / "final").rglob("*.parquet"))
            out = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            self.assertEqual(set(out["name"]), {"chr1:1001-1010"})


class TestStagedSourceDir(unittest.TestCase):
    def test_finalize_reads_staged_from_separate_source_dir(self):
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

            base = {
                "input_dir": str(inp),
                "atlas_path": str(d / "atlas.tsv"),
                "reference_genome": "hg38",
                "labels_dict_path": str(d / "labels.json"),
                "n_buckets": 2,
                "n_workers": 1,
                "seed": 42,
                "signature": {
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                "labelers": {"label": {"type": "hard_with_background"}},
            }
            # Stage into dir A.
            DatasetBuildPipeline(
                {**base, "phase": "stage", "output_dir": str(d / "A")},
                logging.getLogger("t"),
            ).run()
            # Finalize into a different dir B, reading staged data from A.
            DatasetBuildPipeline(
                {
                    **base,
                    "phase": "finalize",
                    "output_dir": str(d / "B"),
                    "staged_source_dir": str(d / "A"),
                },
                logging.getLogger("t"),
            ).run()

            finals = list((d / "B" / "final").rglob("*.parquet"))
            self.assertTrue(finals)
            self.assertFalse((d / "A" / "final").exists())


class _FakeBam:
    references = ("1", "2")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDatasetBuildFromBams(unittest.TestCase):
    def test_bam_input_labels_samples_from_groups_file(self):
        from unittest.mock import patch

        from syto.data.dataset_build import stage

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            atlas = pd.DataFrame(
                {
                    "chr": ["chr1", "chr1"],
                    "start": [1001, 2001],
                    "end": [1010, 2010],
                    "startCpG": [10, 20],
                    "endCpG": [12, 22],
                    "target": ["Healthy CSF", "Medulloblastoma"],
                    "name": ["chr1:1001-1010", "chr1:2001-2010"],
                    "direction": ["U", "U"],
                    "Healthy CSF": [0.9, 0.1],
                    "Medulloblastoma": [0.1, 0.9],
                }
            )
            atlas.to_csv(d / "atlas.tsv", sep="\t", index=False)
            (d / "labels.json").write_text(
                json.dumps({"0": "Healthy CSF", "1": "Medulloblastoma"})
            )
            inp = d / "in"
            inp.mkdir()
            samples = {f"csf{i}_merged.mdup": "Healthy CSF" for i in range(3)}
            samples.update({f"mb{i}_merged.mdup": "Medulloblastoma" for i in range(3)})
            for name in [*samples, "outlier_merged.mdup"]:
                (inp / f"{name}.bam").touch()
            pd.DataFrame(
                {"name": list(samples), "group": list(samples.values())}
            ).to_csv(d / "groups.csv", index=False)

            bam_reads = pd.DataFrame(
                {
                    "read_name": [f"r{i}" for i in range(4)],
                    "chromosome": ["1"] * 4,
                    "read_start": [1000, 1000, 2000, 2000],
                    "read_end": [1010] * 2 + [2010] * 2,
                    "seq": ["ACGTACGTAC"] * 4,
                    "methylation_encoding": ["0101010101"] * 4,
                }
            )
            cfg = {
                "phase": "all",
                "input_type": "bam",
                "input_dir": str(inp),
                "sample_groups_path": str(d / "groups.csv"),
                "output_dir": str(d / "out"),
                "atlas_path": str(d / "atlas.tsv"),
                "atlas_cell_types": ["Healthy CSF", "Medulloblastoma"],
                "reference_genome": "hg19",
                "labels_dict_path": str(d / "labels.json"),
                "n_buckets": 2,
                "n_workers": 1,
                "seed": 42,
                "bam_processing": {
                    "data_type": "wgbs",
                    "reference_path": "ref.fa",
                    "merge_pairs": True,
                    "mate_fetch_padding": 100,
                },
                "signature": {
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                "labelers": {"label": {"type": "hard_with_background"}},
            }
            with patch.object(stage, "pysam") as pysam_mock, patch.object(
                stage, "read_bam_regions", return_value=bam_reads
            ) as read_mock:
                pysam_mock.AlignmentFile.return_value = _FakeBam()
                summary = DatasetBuildPipeline(cfg, logging.getLogger("t")).run()

            self.assertEqual(summary["stage"]["files"], 6)
            params, _ = read_mock.call_args.args
            self.assertEqual(params["require_flags"], 3)
            # One padded fetch per region bucket, with the chr prefix stripped.
            self.assertEqual(read_mock.call_count, 6 * 2)
            self.assertEqual(
                [call.args[1] for call in read_mock.call_args_list[:2]],
                [[("1", 900, 1110)], [("1", 1900, 2110)]],
            )

            finals = list((d / "out" / "final").rglob("*.parquet"))
            out = pd.concat([pd.read_parquet(p) for p in finals], ignore_index=True)
            labels = out.groupby("file")["original_label"].unique()
            self.assertEqual(
                {f: list(v) for f, v in labels.items()},
                {f: [0 if g == "Healthy CSF" else 1] for f, g in samples.items()},
            )
            self.assertTrue(set(out["split"]).issubset({"train", "valid", "test"}))


class TestRegionAnnotations(unittest.TestCase):
    """Atlas columns can be carried onto reads for a classifier head to key on."""

    def test_annotation_is_dense_encoded_and_mapping_saved(self):
        from syto.data.dataset_build.filters import load_region_annotations

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            atlas = pd.DataFrame(
                {
                    "name": ["r0", "r1", "r2"],
                    # deliberately non-contiguous: an embedding head needs 0..K-1
                    "cluster": [7, 42, 7],
                }
            )
            atlas.to_csv(d / "atlas.tsv", sep="\t", index=False)

            table, mappings = load_region_annotations(
                str(d / "atlas.tsv"), {"cluster": "cluster_label"}
            )
            self.assertEqual(list(table["cluster_label"]), [0, 1, 0])
            self.assertEqual(mappings["cluster_label"], {0: 7, 1: 42})
            self.assertEqual(sorted(table.columns), ["cluster_label", "name"])
