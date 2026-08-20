"""Chunked stages 2-4 must reproduce the single-shot pipeline exactly."""

import array
import copy
import json
import math
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pysam

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "App"))

from inference import InferencePipeline  # noqa: E402

from baselines.deconvolution.base import BaselineDeconvolver  # noqa: E402
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas  # noqa: E402

TARGETS = ["Blood-B", "Blood-T", "Colon-Ep"]
LABELS_DICT = {i: name for i, name in enumerate(TARGETS)}


class FakeClassifier:
    """Deterministic stand-in for a read classifier.

    Predictions depend only on the read's own content, so the same read gets
    the same scores no matter which chunk it arrives in.
    """

    def __init__(self):
        self.n_calls = 0
        self.max_rows = 0

    def predict_split(self, split_df, **kwargs):
        self.n_calls += 1
        self.max_rows = max(self.max_rows, len(split_df))
        df = split_df.copy()
        seed = (
            df["read_start"].to_numpy() * 31 + df["name"].str.len().to_numpy()
        ).astype(np.int64)
        for label in range(len(TARGETS)):
            df[f"prediction_{label}"] = (np.sin(seed + label).astype(float) + 1.0) / 2.0
        totals = sum(df[f"prediction_{i}"] for i in range(len(TARGETS)))
        for label in range(len(TARGETS)):
            df[f"prediction_{label}"] = df[f"prediction_{label}"] / totals
        return df


def build_atlas_frame(n_regions_per_target=4):
    """A miniature UXM atlas: interleaved regions across two chromosomes."""
    rows = []
    for region in range(n_regions_per_target):
        for target_idx, target in enumerate(TARGETS):
            index = region * len(TARGETS) + target_idx
            chromosome = "chr1" if index % 2 == 0 else "chr2"
            start = 1_000 + index * 500
            end = start + 300
            rows.append(
                {
                    "chr": chromosome,
                    "start": start,
                    "end": end,
                    "startCpG": index * 10,
                    "endCpG": index * 10 + 5,
                    "target": target,
                    "name": f"{chromosome}:{start}-{end}",
                    "direction": "U",
                }
            )
    atlas = pd.DataFrame(rows)
    # The UXM atlas format demands the full cell-type column set.
    for cell_type in UXMMethylationAtlas.EXPECTED_CTYPE_COLUMNS:
        atlas[cell_type] = np.where(atlas["target"] == cell_type, 0.9, 0.1)
    return atlas


def build_reads(atlas, n_reads=300, seed=7):
    """Reads scattered over the atlas: some miss, some span several regions."""
    rng = np.random.default_rng(seed)
    rows = []
    for read in range(n_reads):
        chromosome = "chr1" if read % 2 == 0 else "chr2"
        start = int(rng.integers(500, 1_000 + len(atlas) * 500))
        length = int(rng.integers(50, 900))
        pattern = "".join(rng.choice(list("012"), size=length + 1))
        rows.append(
            {
                "read_name": f"read_{read}",
                "chromosome": chromosome,
                "read_start": start,
                "read_end": start + length,
                "seq": "ACGT" * (length // 4 + 1),
                "methylation_encoding": pattern,
                "label": 0,
            }
        )
    return pd.DataFrame(rows)


class ChunkedInferenceTestBase(unittest.TestCase):
    """Builds the on-disk artefacts the pipeline constructor insists on."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.atlas_frame = build_atlas_frame()
        cls.atlas_path = cls.tmp / "atlas.tsv"
        cls.atlas_frame.to_csv(cls.atlas_path, sep="\t", index=False)

        cls.labels_path = cls.tmp / "labels_dict.json"
        cls.labels_path.write_text(
            json.dumps({str(k): v for k, v in LABELS_DICT.items()})
        )

        cls.mask_path = cls.tmp / "mask.npz"
        np.savez(cls.mask_path, features_mask=np.ones(len(TARGETS) ** 2, dtype=bool))

        cls.reads = build_reads(cls.atlas_frame)
        cls.logger = logging.getLogger("chunked-inference-test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build_pipeline(self, chunking=None, output_dir=None, extra_config=None):
        config = {
            "labels_dict_path": str(self.labels_path),
            "features_mask_path": str(self.mask_path),
            "checkpoint_path": "unused",
            "num_labels": len(TARGETS),
            "classifier": {"classifier_type": "lookup"},
            "input": {"type": "parsed_reads", "data_path": "unused.pkl"},
            "output_dir": str(output_dir or self.tmp / "out"),
            "output": {"save_predictions": False},
            # Filling missing labels needs a pseudobulk-derived prior on disk;
            # the chunked/unchunked equivalence is unaffected by that step,
            # which runs once on the finished matrix in both paths.
            "fill_in_missing_labels": False,
            "deconvolution": {
                "syto": {
                    "atlas_path": str(self.atlas_path),
                    "atlas_name": "test-atlas",
                    "methods": [{"name": "ls", "flavor": "nnls", "enabled": False}],
                },
                "baselines": [],
            },
        }
        if chunking is not None:
            config["chunked_inference"] = chunking
        if extra_config:
            config.update(extra_config)

        pipeline = InferencePipeline(config=config, logger=self.logger)
        pipeline.processed_reads = self.reads.copy()
        return pipeline


class TestChunkedMatchesUnchunked(ChunkedInferenceTestBase):
    """The chunked path is an optimisation, not a different computation."""

    def _unchunked_aggregation(self):
        pipeline = self.build_pipeline()
        classifier = FakeClassifier()
        with patch.object(pipeline, "_build_classifier", return_value=classifier):
            pipeline.prepared_reads = pipeline._prepare_reads()
            pipeline.predictions_df = pipeline._predict_classifier()
            return pipeline._aggregate_to_dmr()

    def _chunked_aggregation(self, chunking):
        pipeline = self.build_pipeline(chunking=chunking)
        classifier = FakeClassifier()
        with patch.object(pipeline, "_build_classifier", return_value=classifier):
            aggregated = pipeline._run_syto_stages_chunked()
        return aggregated, classifier

    def test_region_chunks_match_the_single_shot_matrix(self):
        """Any regions-per-chunk setting yields the same DMR matrix."""
        expected = self._unchunked_aggregation()
        for regions_per_chunk in (1, 2, 5, 100):
            with self.subTest(regions_per_chunk=regions_per_chunk):
                result, _ = self._chunked_aggregation(
                    {
                        "enabled": True,
                        "chunk_by": "region",
                        "regions_per_chunk": regions_per_chunk,
                        "progress_bar": False,
                    }
                )
                pd.testing.assert_frame_equal(result, expected)

    def test_grg_chunks_match_the_single_shot_matrix(self):
        """One chunk per GR group produces one feature-matrix row at a time."""
        expected = self._unchunked_aggregation()
        result, classifier = self._chunked_aggregation(
            {"enabled": True, "chunk_by": "grg", "progress_bar": False}
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertLessEqual(classifier.n_calls, len(TARGETS))

    def test_every_read_is_classified_exactly_once_per_region(self):
        """Chunk boundaries must neither drop nor duplicate read x region pairs."""
        expected = self._unchunked_aggregation()
        result, _ = self._chunked_aggregation(
            {
                "enabled": True,
                "chunk_by": "region",
                "regions_per_chunk": 1,
                "progress_bar": False,
            }
        )
        pd.testing.assert_series_equal(result["n_reads"], expected["n_reads"])


class TestChunkPlanningAndSelection(ChunkedInferenceTestBase):
    """Unit-level checks on chunk construction and read lookup."""

    def test_region_chunks_cover_the_atlas_without_overlap(self):
        pipeline = self.build_pipeline(
            {"enabled": True, "chunk_by": "region", "regions_per_chunk": 5}
        )
        chunks = pipeline._plan_chunks()
        names = pd.concat([chunk["name"] for chunk in chunks])
        self.assertEqual(len(names), len(pipeline.atlas.atlas))
        self.assertEqual(set(names), set(pipeline.atlas.atlas["name"]))

    def test_grg_chunks_are_one_per_target(self):
        pipeline = self.build_pipeline({"enabled": True, "chunk_by": "grg"})
        chunks = pipeline._plan_chunks()
        self.assertEqual(len(chunks), len(TARGETS))
        for chunk in chunks:
            self.assertEqual(chunk["target"].nunique(), 1)

    def test_read_lookup_returns_every_overlapping_read(self):
        """The index must not miss long reads that start far to the left."""
        pipeline = self.build_pipeline({"enabled": True})
        index = pipeline._build_read_index(pipeline.processed_reads)
        atlas = pipeline.atlas.atlas

        for _, region in atlas.iterrows():
            regions = atlas[atlas["name"] == region["name"]]
            selected = pipeline._reads_for_regions(regions, index)
            brute_force = pipeline.processed_reads[
                (pipeline.processed_reads["chromosome"] == region["chr"])
                & (pipeline.processed_reads["read_start"] < region["end"])
                & (pipeline.processed_reads["read_end"] >= region["start"] - 1)
            ]
            selected_names = set() if selected is None else set(selected["read_name"])
            self.assertEqual(selected_names, set(brute_force["read_name"]))

    def test_consumers_sharing_a_locus_land_in_the_same_chunk(self):
        """regions_per_chunk counts distinct loci, so reads are parsed once."""

        class FakeAccumulator:
            def __init__(self, atlas):
                self.atlas = atlas

        pipeline = self.build_pipeline(
            {"enabled": True, "chunk_by": "region", "regions_per_chunk": 3}
        )
        # A baseline whose atlas repeats the syto regions verbatim.
        chunks = pipeline._plan_chunks([FakeAccumulator(pipeline.atlas)])

        n_loci = len(pipeline.atlas.atlas)
        self.assertEqual(len(chunks), math.ceil(n_loci / 3))
        for chunk in chunks:
            loci = chunk[["chr", "start", "end"]].drop_duplicates()
            self.assertLessEqual(len(loci), 3)
            # Both consumers are present for every locus in the chunk.
            self.assertEqual(len(chunk), 2 * len(loci))
            self.assertEqual(set(chunk["consumer"]), {-1, 0})

    def test_invalid_chunk_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "chunk_by"):
            self.build_pipeline({"enabled": True, "chunk_by": "chromosome"})

    def test_zero_regions_per_chunk_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regions_per_chunk"):
            self.build_pipeline({"enabled": True, "regions_per_chunk": 0})


class TestChunkedPredictionStreaming(ChunkedInferenceTestBase):
    """save_predictions in chunked mode streams to parquet instead of pickling."""

    def test_predictions_are_streamed_to_parquet(self):
        output_dir = self.tmp / "streamed"
        pipeline = self.build_pipeline(
            {
                "enabled": True,
                "chunk_by": "region",
                "regions_per_chunk": 3,
                "progress_bar": False,
            },
            output_dir=output_dir,
        )
        pipeline.config["output"]["save_predictions"] = True

        with patch.object(pipeline, "_build_classifier", return_value=FakeClassifier()):
            aggregated = pipeline._run_syto_stages_chunked()

        path = output_dir / "predictions.parquet"
        self.assertTrue(path.exists())
        written = pd.read_parquet(path)
        self.assertEqual(len(written), int(aggregated["n_reads"].sum()))
        self.assertIn("prediction_0", written.columns)


class TestChunkedFailureModes(ChunkedInferenceTestBase):
    """Nothing overlapping is an error, exactly as in the unchunked path."""

    def test_no_overlapping_reads_raises(self):
        pipeline = self.build_pipeline(
            {"enabled": True, "chunk_by": "region", "progress_bar": False}
        )
        pipeline.processed_reads = pipeline.processed_reads.assign(chromosome="chr9")
        with patch.object(pipeline, "_build_classifier", return_value=FakeClassifier()):
            with self.assertRaisesRegex(RuntimeError, "No reads overlapped"):
                pipeline._run_syto_stages_chunked()


def write_ont_bam(path, atlas, n_reads_per_region=3, long_reads=6, seed=11):
    """Write a small coordinate-sorted, indexed ONT BAM with MM/ML tags.

    Mixes reads that sit inside a single region with long reads spanning
    several of them, plus reads that overlap nothing — the three cases the
    chunked reader has to get right.
    """
    rng = np.random.default_rng(seed)
    chrom_lengths = {"chr1": 200_000, "chr2": 200_000}

    specs = []
    for _, region in atlas.iterrows():
        for replicate in range(n_reads_per_region):
            start = int(region["start"]) - 30 + replicate * 20
            specs.append((region["chr"], start, 240))
    for index in range(long_reads):
        chromosome = "chr1" if index % 2 == 0 else "chr2"
        specs.append((chromosome, 900 + index * 300, 4_500))
    for index in range(4):  # overlapping no atlas region at all
        specs.append(("chr1", 100_000 + index * 1_000, 300))

    specs.sort(key=lambda spec: (spec[0], spec[1]))

    header = {
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": name, "LN": length} for name, length in chrom_lengths.items()],
    }
    references = list(chrom_lengths)
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for index, (chromosome, start, length) in enumerate(specs):
            n_cpgs = length // 3
            sequence = "ACG" * n_cpgs
            segment = pysam.AlignedSegment()
            segment.query_name = f"read_{index}"
            segment.query_sequence = sequence
            segment.flag = 0 if index % 3 else 16
            segment.reference_id = references.index(chromosome)
            segment.reference_start = start
            segment.mapping_quality = 60
            segment.cigartuples = [(0, len(sequence))]
            segment.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
            segment.set_tag("MM", "C+m," + ",".join(["0"] * n_cpgs))
            # Reads are drawn as mostly-unmethylated or mostly-methylated rather
            # than uniformly at random: uniform ML values put nearly every CpG in
            # the uncertain band, which leaves UXM with an all-zero mixture.
            low, high = (0, 60) if index % 2 else (200, 256)
            segment.set_tag(
                "ML",
                array.array("B", [int(v) for v in rng.integers(low, high, n_cpgs)]),
            )
            bam.write(segment)
    pysam.index(str(path))
    return path


class BamBackedTestBase(ChunkedInferenceTestBase):
    """Adds a real indexed ONT BAM and a pipeline wired to read it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bam_path = write_ont_bam(cls.tmp / "reads.bam", cls.atlas_frame)

    def build_bam_pipeline(self, chunking=None, chromosomes="all", **config_overrides):
        extra = {
            "input": {
                "type": "bam",
                "data_path": str(self.bam_path),
                "data_type": "ont",
                "chromosomes": chromosomes,
            },
            "bam_processing": {
                "n_jobs": 1,
                "min_mapq": 10,
                "exclude_flags": 1796,
                "min_cpgs": 1,
                "merge_pairs": False,
                "ont_methyl_tr": 180,
                "ont_unmethyl_tr": 75,
            },
        }
        extra.update(config_overrides)
        pipeline = self.build_pipeline(chunking=chunking, extra_config=extra)
        pipeline.processed_reads = None
        pipeline._read_classifier = FakeClassifier()
        return pipeline


class TestStreamedStageOne(BamBackedTestBase):
    """Parsing per chunk from the BAM must match parsing the whole file."""

    def _whole_file_aggregation(self):
        pipeline = self.build_bam_pipeline()
        pipeline.processed_reads = pipeline._process_bam()
        pipeline.prepared_reads = pipeline._prepare_reads()
        pipeline.predictions_df = pipeline._predict_classifier()
        return pipeline._aggregate_to_dmr(), len(pipeline.processed_reads)

    def test_streamed_chunks_match_whole_file_parsing(self):
        """Same DMR matrix whether stage 1 ran once or per chunk."""
        expected, _ = self._whole_file_aggregation()
        for chunk_by, regions_per_chunk in (("region", 1), ("region", 5), ("grg", 25)):
            with self.subTest(chunk_by=chunk_by, regions_per_chunk=regions_per_chunk):
                pipeline = self.build_bam_pipeline(
                    {
                        "enabled": True,
                        "stream_bam": True,
                        "chunk_by": chunk_by,
                        "regions_per_chunk": regions_per_chunk,
                        "progress_bar": False,
                    }
                )
                self.assertTrue(pipeline.stream_bam)
                result = pipeline._run_syto_stages_chunked()
                pd.testing.assert_frame_equal(result, expected)

    def test_streaming_never_materialises_the_read_table(self):
        """processed_reads stays empty; only chunk-sized frames are built."""
        pipeline = self.build_bam_pipeline(
            {
                "enabled": True,
                "stream_bam": True,
                "chunk_by": "region",
                "regions_per_chunk": 2,
                "progress_bar": False,
            }
        )
        pipeline._run_syto_stages_chunked()
        self.assertIsNone(pipeline.processed_reads)
        self.assertIsNone(pipeline.prepared_reads)
        self.assertIsNone(pipeline.predictions_df)

        _, n_whole_file_reads = self._whole_file_aggregation()
        self.assertLess(pipeline._read_classifier.max_rows, n_whole_file_reads)

    def test_chromosome_restriction_is_honoured(self):
        """Regions outside input.chromosomes are never fetched."""
        pipeline = self.build_bam_pipeline(
            {
                "enabled": True,
                "stream_bam": True,
                "chunk_by": "region",
                "regions_per_chunk": 100,
                "progress_bar": False,
            },
            chromosomes=["chr1"],
        )
        aggregated = pipeline._run_syto_stages_chunked()

        self.assertGreater(aggregated["n_reads"].sum(), 0)
        intervals = pipeline._fetch_intervals(pipeline.atlas.atlas)
        self.assertTrue(all(chromosome == "chr1" for chromosome, _, _ in intervals))

    def test_prediction_write_failure_does_not_abort_the_run(self):
        """The DMR matrix is the deliverable; a broken side output is not."""
        pipeline = self.build_bam_pipeline(
            {
                "enabled": True,
                "stream_bam": True,
                "chunk_by": "region",
                "regions_per_chunk": 2,
                "progress_bar": False,
            }
        )
        pipeline.config["output"]["save_predictions"] = True

        class BrokenWriter:
            def write(self, df):
                raise OSError("disk full")

            def close(self):
                pass

        with patch.object(
            pipeline, "_open_chunked_prediction_writer", return_value=BrokenWriter()
        ):
            aggregated = pipeline._run_syto_stages_chunked()
        self.assertGreater(aggregated["n_reads"].sum(), 0)

    def test_fetch_intervals_are_deduplicated(self):
        """One fetch per locus, however many consumers claim it."""

        class FakeAccumulator:
            def __init__(self, atlas):
                self.atlas = atlas

        pipeline = self.build_bam_pipeline(
            {"enabled": True, "stream_bam": True, "regions_per_chunk": 4}
        )
        chunk = pipeline._plan_chunks([FakeAccumulator(pipeline.atlas)])[0]
        intervals = pipeline._fetch_intervals(chunk)
        self.assertEqual(len(intervals), len(set(intervals)))
        self.assertEqual(len(intervals), len(chunk) // 2)

    def test_grg_chunking_with_baselines_is_rejected(self):
        """Baseline atlases force a genomically ordered union pass."""
        with self.assertRaisesRegex(ValueError, "chunk_by"):
            self.build_bam_pipeline(
                {"enabled": True, "stream_bam": True, "chunk_by": "grg"},
                deconvolution={
                    "syto": {
                        "atlas_path": str(self.atlas_path),
                        "atlas_name": "test-atlas",
                        "methods": [{"name": "ls", "flavor": "nnls", "enabled": False}],
                    },
                    "baselines": [
                        {
                            "model": "uxm",
                            "enabled": True,
                            "atlas_path": str(self.atlas_path),
                        }
                    ],
                },
            )


class TestBamRegionReader(unittest.TestCase):
    """Region-driven parsing: right reads, no duplicates, filters applied."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.atlas_frame = build_atlas_frame()
        cls.bam_path = write_ont_bam(cls.tmp / "reads.bam", cls.atlas_frame)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _reader(self, **kwargs):
        from syto.data.sequencing.bam_processing import BamRegionReader

        params = dict(
            bam_path=str(self.bam_path),
            data_type="ont",
            methyl_tr=180,
            unmethyl_tr=75,
            require_flags=0,
            min_cpgs=1,
        )
        params.update(kwargs)
        return BamRegionReader(**params)

    def test_returns_each_read_once_per_call(self):
        """A read spanning several requested regions is parsed once."""
        regions = [
            (row["chr"], int(row["start"]) - 1, int(row["end"]))
            for _, row in self.atlas_frame.iterrows()
        ]
        with self._reader() as reader:
            df = reader.read_regions(regions)
        self.assertFalse(df.empty)
        self.assertEqual(df["read_name"].duplicated().sum(), 0)

    def test_matches_whole_file_parsing_for_the_same_reads(self):
        """Region parsing agrees read-for-read with the whole-file walk."""
        from syto.data.sequencing.bam_processing import process_bam_with_chunking

        whole = process_bam_with_chunking(
            bam_path=str(self.bam_path),
            chromosomes=["chr1", "chr2"],
            n_jobs=1,
            data_type="ont",
            methyl_tr=180,
            unmethyl_tr=75,
            require_flags=0,
            min_cpgs=1,
            merge_pairs=False,
        )
        regions = [
            (row["chr"], int(row["start"]) - 1, int(row["end"]))
            for _, row in self.atlas_frame.iterrows()
        ]
        with self._reader() as reader:
            streamed = reader.read_regions(regions)

        shared = set(streamed["read_name"]) & set(whole["read_name"])
        self.assertGreater(len(shared), 0)
        columns = ["read_name", "read_start", "read_end", "methylation_encoding"]
        left = streamed[streamed["read_name"].isin(shared)].sort_values("read_name")
        right = whole[whole["read_name"].isin(shared)].sort_values("read_name")
        pd.testing.assert_frame_equal(
            left[columns].reset_index(drop=True),
            right[columns].reset_index(drop=True),
        )

    def test_empty_region_set_returns_empty_frame(self):
        with self._reader() as reader:
            self.assertTrue(reader.read_regions([("chr1", 150_000, 151_000)]).empty)
            self.assertTrue(reader.read_regions([("chrX", 0, 1_000)]).empty)

    def test_wgbs_without_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reference genome path"):
            self._reader(data_type="wgbs")


def build_cpg_count_atlas(atlas, path, cpgs_per_region=4):
    """Write a CelFiE/EpiDISH-style per-CpG METH/COV atlas for the same regions."""
    rng = np.random.default_rng(3)
    rows = []
    for _, region in atlas.iterrows():
        for index in range(cpgs_per_region):
            start = int(region["start"]) - 1 + 10 + index * 20
            row = {
                "CHROM": region["chr"],
                "START": start,
                "END": start + 1,
                "name": region["name"],
            }
            for cell_type in TARGETS:
                coverage = int(rng.integers(20, 60))
                methylated = int(
                    coverage * (0.1 if cell_type == region["target"] else 0.9)
                )
                row[f"{cell_type}_METH"] = methylated
                row[f"{cell_type}_COV"] = coverage
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


class TestStreamedBaselineInputs(ChunkedInferenceTestBase):
    """merge_inputs over chunks must reproduce a single-shot build_input."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cpg_atlas_path = build_cpg_count_atlas(
            cls.atlas_frame, cls.tmp / "cpg_counts.tsv"
        )

    def _deconvolvers(self):
        from baselines.deconvolution.celfie.celfie import CelFiEDeconvolver
        from baselines.deconvolution.celfieish.celfieish import CelFiEISHDeconvolver
        from baselines.deconvolution.epidish.epidish import EpiDishDeconvolver
        from baselines.deconvolution.uxm.uxm import UXMDeconvolver
        from syto.data.atlases.celfieish_atlases import CpGBetaCountsMethylationAtlas

        uxm_atlas = UXMMethylationAtlas(
            atlas_name="uxm", reference_genome="hg38", atlas_path=str(self.atlas_path)
        )
        cpg_atlas = CpGBetaCountsMethylationAtlas(
            atlas_name="cpg",
            reference_genome="hg38",
            atlas_path=str(self.cpg_atlas_path),
        )
        return [
            UXMDeconvolver(uxm_atlas, ref_cells=list(uxm_atlas.ref_cells)),
            CelFiEDeconvolver(cpg_atlas, num_iterations=5, convergence_criteria=0.01),
            CelFiEISHDeconvolver(
                cpg_atlas, num_iterations=5, convergence_criteria=0.01
            ),
            EpiDishDeconvolver(cpg_atlas),
        ]

    @staticmethod
    def _chunk_inputs(deconvolver, reads, regions_per_chunk):
        """Build one input per chunk of the deconvolver's own atlas."""
        atlas = deconvolver.atlas.atlas.sort_values(["chr", "start", "end"])
        parts = []
        for start in range(0, len(atlas), regions_per_chunk):
            names = atlas.iloc[start : start + regions_per_chunk]["name"]
            sliced = copy.copy(deconvolver)
            sliced._atlas = deconvolver.atlas.subset_regions(names)
            prepared = sliced.prepare_reads(BaselineDeconvolver._sort_reads(reads))
            if prepared.empty:
                continue
            built = sliced.build_input(prepared)
            if built:
                parts.append(built)
        return parts

    def _single_shot_input(self, deconvolver, reads):
        prepared = deconvolver.prepare_reads(BaselineDeconvolver._sort_reads(reads))
        return deconvolver.build_input(prepared)

    def test_merged_chunk_inputs_match_single_shot(self):
        """Every baseline: chunked input == whole-sample input, region order aside."""
        for deconvolver in self._deconvolvers():
            for regions_per_chunk in (1, 5):
                with self.subTest(model=deconvolver.name, chunk=regions_per_chunk):
                    expected = self._single_shot_input(deconvolver, self.reads)
                    merged = deconvolver.merge_inputs(
                        self._chunk_inputs(deconvolver, self.reads, regions_per_chunk)
                    )
                    self.assertIsNotNone(merged)
                    self._assert_inputs_equivalent(deconvolver.name, expected, merged)

    def _assert_inputs_equivalent(self, model, expected, merged):
        if "scaling_factors" in expected:  # UXM
            for key in ("scaling_factors", "counts"):
                left = expected[key].sort_values("name").reset_index(drop=True)
                right = merged[key].sort_values("name").reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right)
        elif "matrices" in expected:  # CelFiE-ISH
            self.assertEqual(
                sorted(expected["region_names"]), sorted(merged["region_names"])
            )
            by_region = dict(zip(merged["region_names"], merged["matrices"]))
            for name, matrix in zip(expected["region_names"], expected["matrices"]):
                np.testing.assert_array_equal(matrix, by_region[name])
        elif "x_meth" in expected:  # CelFiE
            self.assertEqual(
                sorted(expected["region_names"]), sorted(merged["region_names"])
            )
            for key in ("x_meth", "x_cov"):
                by_region = dict(zip(merged["region_names"], merged[key]))
                for name, values in zip(expected["region_names"], expected[key]):
                    np.testing.assert_array_equal(values, by_region[name])
        else:  # EpiDISH: rows are CpGs, identity dropped by build_input
            self.assertEqual(expected["ref_cells"], merged["ref_cells"])
            self.assertEqual(expected["mixture"].shape, merged["mixture"].shape)
            left = np.column_stack([expected["mixture"], expected["ref"]])
            right = np.column_stack([merged["mixture"], merged["ref"]])
            order_left = np.lexsort(left.T)
            order_right = np.lexsort(right.T)
            np.testing.assert_allclose(left[order_left], right[order_right])


class TestStreamedBaselineRun(BamBackedTestBase):
    """End-to-end: a streamed run deconvolutes baselines from the same pass."""

    def _uxm_config(self):
        return {
            "syto": {
                "atlas_path": str(self.atlas_path),
                "atlas_name": "test-atlas",
                "methods": [{"name": "ls", "flavor": "nnls", "enabled": False}],
            },
            "baselines": [
                {"model": "uxm", "enabled": True, "atlas_path": str(self.atlas_path)}
            ],
        }

    def test_streamed_uxm_matches_the_whole_file_baseline(self):
        """UXM proportions are identical whether reads were streamed or not."""
        whole = self.build_bam_pipeline(deconvolution=self._uxm_config())
        whole.processed_reads = whole._process_bam()
        expected = whole._run_baseline(whole._baseline_deconvolvers[0])

        streamed = self.build_bam_pipeline(
            {
                "enabled": True,
                "stream_bam": True,
                "chunk_by": "region",
                "regions_per_chunk": 3,
                "progress_bar": False,
            },
            deconvolution=self._uxm_config(),
        )
        streamed._run_syto_stages_chunked()
        results = streamed._streamed_baseline_results

        self.assertEqual(len(results), len(expected))
        self.assertEqual(results[0][0], expected[0][0])
        # A degenerate (all-zero) mixture would make the comparison vacuous.
        self.assertTrue(np.isfinite(expected[0][2]).all())
        np.testing.assert_allclose(results[0][2], expected[0][2], atol=1e-4)

    def test_streamed_baselines_reach_stage_five(self):
        """The streamed result is what stage 5 reports, without re-running reads."""
        pipeline = self.build_bam_pipeline(
            {
                "enabled": True,
                "stream_bam": True,
                "chunk_by": "region",
                "regions_per_chunk": 3,
                "progress_bar": False,
            },
            deconvolution=self._uxm_config(),
        )
        pipeline.dmr_aggregated = pipeline._run_syto_stages_chunked()
        results = pipeline._run_deconvolution()
        self.assertTrue(any(name == "uxm" for name, _, _ in results))


class TestStreamingPredictionWriter(unittest.TestCase):
    """Chunks whose column sets differ must not abort the run."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "predictions.parquet"
        self.logger = logging.getLogger("writer-test")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_later_chunks_are_aligned_to_the_first_schema(self):
        """merge_paired_reads only emits its extra columns for merged fragments."""
        from inference import _StreamingParquetWriter

        wide = pd.DataFrame(
            {
                "read_name": ["a"],
                "prediction_0": [0.5],
                "read_total_cpgs": [7],
                "is_merged_pair": [True],
            }
        )
        narrow = pd.DataFrame({"read_name": ["b"], "prediction_0": [0.25]})

        writer = _StreamingParquetWriter(str(self.path), self.logger)
        writer.write(wide)
        writer.write(narrow)  # missing columns -> null
        writer.close()

        written = pd.read_parquet(self.path)
        self.assertEqual(len(written), 2)
        self.assertEqual(list(written.columns), list(wide.columns))
        self.assertTrue(pd.isna(written.loc[1, "read_total_cpgs"]))

    def test_unexpected_columns_are_dropped_not_fatal(self):
        """A chunk that gains a column keeps writing, minus that column."""
        from inference import _StreamingParquetWriter

        writer = _StreamingParquetWriter(str(self.path), self.logger)
        writer.write(pd.DataFrame({"read_name": ["a"], "prediction_0": [0.5]}))
        writer.write(
            pd.DataFrame(
                {"read_name": ["b"], "prediction_0": [0.25], "mate1_start": [10]}
            )
        )
        writer.close()

        written = pd.read_parquet(self.path)
        self.assertEqual(len(written), 2)
        self.assertNotIn("mate1_start", written.columns)


class TestAtlasRestrictedStageOne(BamBackedTestBase):
    """Unchunked stage 1 parses only what the atlases can possibly use."""

    def test_restricted_parse_keeps_every_atlas_overlapping_read(self):
        """The restricted read table loses nothing stage 2 would have used."""
        whole = self.build_bam_pipeline(
            bam_processing={
                "n_jobs": 1,
                "min_mapq": 10,
                "exclude_flags": 1796,
                "min_cpgs": 1,
                "merge_pairs": False,
                "ont_methyl_tr": 180,
                "ont_unmethyl_tr": 75,
                "restrict_to_atlas": False,
            }
        )
        whole_reads = whole._process_bam()
        whole_prepared = whole._overlap_reads_with_atlas(whole_reads, whole.atlas)

        restricted = self.build_bam_pipeline()
        restricted_reads = restricted._process_bam()
        restricted_prepared = restricted._overlap_reads_with_atlas(
            restricted_reads, restricted.atlas
        )

        # Fewer reads parsed, identical atlas-overlapped result.
        self.assertLess(len(restricted_reads), len(whole_reads))
        self.assertEqual(len(restricted_prepared), len(whole_prepared))
        self.assertEqual(
            sorted(restricted_prepared["read_name"]),
            sorted(whole_prepared["read_name"]),
        )

    def test_restricted_parse_gives_the_same_feature_matrix(self):
        """Stage 4 cannot tell which stage 1 produced the reads."""
        whole = self.build_bam_pipeline(
            bam_processing={
                "n_jobs": 1,
                "min_mapq": 10,
                "exclude_flags": 1796,
                "min_cpgs": 1,
                "merge_pairs": False,
                "ont_methyl_tr": 180,
                "ont_unmethyl_tr": 75,
                "restrict_to_atlas": False,
            }
        )
        whole.processed_reads = whole._process_bam()
        whole.prepared_reads = whole._prepare_reads()
        whole.predictions_df = whole._predict_classifier()
        expected = whole._aggregate_to_dmr()

        restricted = self.build_bam_pipeline()
        restricted.processed_reads = restricted._process_bam()
        restricted.prepared_reads = restricted._prepare_reads()
        restricted.predictions_df = restricted._predict_classifier()
        pd.testing.assert_frame_equal(restricted._aggregate_to_dmr(), expected)

    def test_intervals_cover_baseline_atlases_and_are_merged(self):
        """Baselines read their own atlases; the fetch must cover those too."""
        pipeline = self.build_bam_pipeline(deconvolution=self._baseline_config())
        intervals = pipeline._atlas_fetch_intervals()

        self.assertTrue(intervals)
        for chromosome, start, end in intervals:
            self.assertLess(start, end)
        # Merged: sorted, non-overlapping within a chromosome.
        for left, right in zip(intervals, intervals[1:]):
            if left[0] == right[0]:
                self.assertLessEqual(left[2], right[1])

        atlas = pipeline.atlas.atlas
        for _, region in atlas.iterrows():
            covered = any(
                chromosome == region["chr"]
                and start <= region["start"] - 1
                and end >= region["end"]
                for chromosome, start, end in intervals
            )
            self.assertTrue(covered, f"{region['name']} not covered")

    def test_restriction_can_be_switched_off(self):
        """The whole-genome parse stays available for callers that need it."""
        pipeline = self.build_bam_pipeline(
            bam_processing={
                "n_jobs": 1,
                "min_mapq": 10,
                "min_cpgs": 1,
                "merge_pairs": False,
                "restrict_to_atlas": False,
            }
        )
        self.assertEqual(pipeline._atlas_fetch_intervals.__self__, pipeline)
        with patch.object(pipeline, "_process_bam_over_regions") as region_mock:
            pipeline._process_bam()
        region_mock.assert_not_called()

    def test_mate_padding_applies_only_when_merging(self):
        """WGBS mates outside the atlas are fetched so pairs still merge."""
        pipeline = self.build_bam_pipeline()
        self.assertEqual(pipeline._mate_fetch_padding({"merge_pairs": False}), 0)
        self.assertEqual(pipeline._mate_fetch_padding({"merge_pairs": True}), 1000)
        pipeline.config["bam_processing"]["mate_fetch_padding"] = 250
        self.assertEqual(pipeline._mate_fetch_padding({"merge_pairs": True}), 250)

    def _baseline_config(self):
        return {
            "syto": {
                "atlas_path": str(self.atlas_path),
                "atlas_name": "test-atlas",
                "methods": [{"name": "ls", "flavor": "nnls", "enabled": False}],
            },
            "baselines": [
                {"model": "uxm", "enabled": True, "atlas_path": str(self.atlas_path)}
            ],
        }


if __name__ == "__main__":
    unittest.main()
