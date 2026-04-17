import unittest
from unittest import mock

import numpy as np
import pandas as pd

from methyldl.data import pseudo_bulk_generation as pseudo_bulk_generation_module


def build_prediction_columns():
    """Return the prediction columns expected by the pseudo-bulk helpers."""
    return [f"prediction_{index}" for index in range(40)]


def build_target_columns():
    """Return the exact aggregated output columns used by pseudo-bulk generation."""
    return [
        "dmr_ctype_label",
        "dmr_ctype",
        "prediction_0_wavg",
        "prediction_1_wavg",
        "prediction_2_wavg",
        "prediction_3_wavg",
        "prediction_4_wavg",
        "prediction_5_wavg",
        "prediction_6_wavg",
        "prediction_7_wavg",
        "prediction_8_wavg",
        "prediction_9_wavg",
        "prediction_10_wavg",
        "prediction_11_wavg",
        "prediction_12_wavg",
        "prediction_13_wavg",
        "prediction_14_wavg",
        "prediction_15_wavg",
        "prediction_16_wavg",
        "prediction_17_wavg",
        "prediction_18_wavg",
        "prediction_19_wavg",
        "prediction_20_wavg",
        "prediction_21_wavg",
        "prediction_22_wavg",
        "prediction_23_wavg",
        "prediction_24_wavg",
        "prediction_25_wavg",
        "prediction_26_wavg",
        "prediction_27_wavg",
        "prediction_28_wavg",
        "prediction_29_wavg",
        "prediction_30_wavg",
        "prediction_31_wavg",
        "prediction_32_wavg",
        "prediction_33_wavg",
        "prediction_34_wavg",
        "prediction_35_wavg",
        "prediction_36_wavg",
        "prediction_37_wavg",
        "prediction_38_wavg",
        "methylation_level_wavg",
        "total_weight",
        "n_reads",
        "chromosome",
        "label",
    ]


def build_split_dataframe(labels, prepared=False):
    """Create a compact split dataframe containing every DMR label for each cell type."""
    rows = []
    prediction_columns = build_prediction_columns()

    for label in labels:
        for dmr_ctype_label in range(39):
            row = {
                "original_label": label,
                "dmr_ctype_label": dmr_ctype_label,
                "dmr_ctype": f"ctype_{dmr_ctype_label}",
                "name": f"region_{label}_{dmr_ctype_label}",
                "record_M": 1,
                "record_U": 1,
                "record_X": 0,
                "label": label,
            }

            for prediction_index, prediction_column in enumerate(prediction_columns):
                row[prediction_column] = (prediction_index + 1) / 100.0

            if prepared:
                row["chromosome"] = "chr1"
                row["direction"] = "U"
                row["total_marked_cpgs"] = 2
                row["methylation_level"] = 0.25
            else:
                row["chr"] = "chr1"
                row["NCPGS"] = 2
                row["M_rate"] = 0.25

            rows.append(row)

    return pd.DataFrame(rows)


def build_empty_prepared_dataframe():
    """Create an empty dataframe with the same schema as prepared split inputs."""
    return build_split_dataframe([0], prepared=True).iloc[0:0].copy()


class PseudoBulkGenerationTestBase(unittest.TestCase):
    """Provide reusable fixtures and helpers for pseudo-bulk generation tests."""

    def setUp(self):
        """Create fresh dataframe fixtures and clear module-level worker state."""
        pseudo_bulk_generation_module._worker_data.clear()

        self.raw_train = build_split_dataframe([0, 1], prepared=False)
        self.raw_valid = build_split_dataframe([0, 1], prepared=False)
        self.raw_test = build_split_dataframe([0, 1], prepared=False)

        self.prepared_train = build_split_dataframe([0], prepared=True)
        self.prepared_valid = build_split_dataframe([0], prepared=True)
        self.prepared_test = build_split_dataframe([0], prepared=True)
        self.empty_prepared = build_empty_prepared_dataframe()

        self.target_columns = build_target_columns()
        self.ref_cells = ["cell_a", "cell_b"]
        self.labels_dict_reversed = {"cell_a": 0, "cell_b": 1}
        self.atlas = pd.DataFrame({"unused": [1]})

    def tearDown(self):
        """Clear module-level worker state after each test method."""
        pseudo_bulk_generation_module._worker_data.clear()

    def make_progress_bar_context(self):
        """Build a mocked tqdm context manager and the progress bar it yields."""
        progress_bar = mock.MagicMock()
        progress_context = mock.MagicMock()
        progress_context.__enter__.return_value = progress_bar
        progress_context.__exit__.return_value = False
        return progress_context, progress_bar


class TestGeneratePseudoBulk(PseudoBulkGenerationTestBase):
    """Exercise the public pseudo-bulk generation helper."""

    @mock.patch.object(
        pseudo_bulk_generation_module,
        "rearange_uxm_deconvolution_results",
        return_value=[0.6, 0.4] + ([0.0] * 37),
    )
    @mock.patch.object(
        pseudo_bulk_generation_module,
        "uxm_deconvolution",
        return_value=[np.array([0.6, 0.4])],
    )
    def test_generate_pseudo_bulk_returns_reads_when_requested(
        self,
        mocked_uxm_deconvolution,
        mocked_rearrange,
    ):
        """Return read-level samples together with aggregated outputs when requested."""
        labels, proportions_full, subs, uxm_data, reads = (
            pseudo_bulk_generation_module.generate_pseudo_bulk(
                total_samples=39,
                labels=[0],
                proportions=[1.0],
                splits={
                    "train": self.raw_train.copy(deep=True),
                    "valid": self.raw_valid.copy(deep=True),
                    "test": self.raw_test.copy(deep=True)
                },
                atlas=self.atlas,
                ref_cells=self.ref_cells,
                labels_dict_reversed=self.labels_dict_reversed,
                return_reads=True,
            )
        )

        self.assertEqual(labels, [0])
        self.assertEqual(len(proportions_full), 39)
        self.assertEqual(proportions_full[0], 1.0)
        self.assertTrue(all(value == 0 for value in proportions_full[1:]))

        self.assertEqual(len(subs), 3)
        self.assertEqual(len(uxm_data), 3)
        self.assertEqual(len(reads), 3)

        for split_name, sub in subs.items():
            self.assertEqual(list(sub.columns), self.target_columns)
            self.assertEqual(len(sub), 39)

        for split_name, read_df in reads.items():
            # Each DMR label contributes exactly one sampled row in this setup.
            self.assertEqual(len(read_df), 39)
            self.assertIn("direction", read_df.columns)
            self.assertIn("chromosome", read_df.columns)

        for split_name, (sf, counts, uxm_results, aligned_results) in uxm_data.items():
            self.assertEqual(
                list(sf.columns), ["name", "direction", "pseudo_bulk_sample"]
            )
            self.assertEqual(
                list(counts.columns),
                ["name", "direction", "pseudo_bulk_sample"],
            )
            self.assertTrue((sf["pseudo_bulk_sample"] == 0.5).all())
            self.assertTrue((counts["pseudo_bulk_sample"] == 2).all())
            self.assertEqual(uxm_results, {"cell_a": 0.6, "cell_b": 0.4})
            self.assertEqual(len(aligned_results), 39)

        self.assertEqual(mocked_uxm_deconvolution.call_count, 3)
        self.assertEqual(mocked_rearrange.call_count, 3)

    @mock.patch.object(
        pseudo_bulk_generation_module,
        "rearange_uxm_deconvolution_results",
        return_value=[0.7, 0.3] + ([0.0] * 37),
    )
    @mock.patch.object(
        pseudo_bulk_generation_module,
        "uxm_deconvolution",
        return_value=[np.array([0.7, 0.3])],
    )
    def test_generate_pseudo_bulk_omits_reads_by_default(
        self,
        mocked_uxm_deconvolution,
        mocked_rearrange,
    ):
        """Return only labels, proportions, aggregated splits, and UXM metadata by default."""
        result = pseudo_bulk_generation_module.generate_pseudo_bulk(
            total_samples=39,
            labels=[0],
            proportions=[1.0],
            splits={
                "train": self.raw_train.copy(deep=True),
                "valid": self.raw_valid.copy(deep=True),
                "test": self.raw_test.copy(deep=True)
            },
            atlas=self.atlas,
            ref_cells=self.ref_cells,
            labels_dict_reversed=self.labels_dict_reversed,
            return_reads=False,
        )

        self.assertEqual(len(result), 4)
        labels, proportions_full, subs, uxm_data = result

        self.assertEqual(labels, [0])
        self.assertEqual(proportions_full[0], 1.0)
        self.assertEqual(len(subs), 3)
        self.assertEqual(len(uxm_data), 3)
        self.assertEqual(mocked_uxm_deconvolution.call_count, 3)
        self.assertEqual(mocked_rearrange.call_count, 3)

    def test_generate_pseudo_bulk_rejects_invalid_proportions(self):
        """Raise an assertion when the requested mixture proportions do not sum to one."""
        with self.assertRaises(AssertionError) as context:
            pseudo_bulk_generation_module.generate_pseudo_bulk(
                total_samples=39,
                labels=[0, 1],
                proportions=[0.4, 0.4],
                splits={
                    "train": self.raw_train.copy(deep=True),
                    "valid": self.raw_valid.copy(deep=True),
                    "test": self.raw_test.copy(deep=True)
                },
                atlas=self.atlas,
                ref_cells=self.ref_cells,
                labels_dict_reversed=self.labels_dict_reversed,
            )

        self.assertIn("Proportions must sum up to one", str(context.exception))


class TestInitWorker(PseudoBulkGenerationTestBase):
    """Verify worker initialization and cached shared state."""

    @mock.patch.object(pseudo_bulk_generation_module.np.random, "seed")
    @mock.patch.object(pseudo_bulk_generation_module.random, "seed")
    @mock.patch.object(pseudo_bulk_generation_module.mp, "current_process")
    def test_init_worker_prepares_grouped_data_and_metadata(
        self,
        mocked_current_process,
        mocked_random_seed,
        mocked_numpy_seed,
    ):
        """Populate worker cache after normalizing split dataframe columns."""
        mocked_current_process.return_value = mock.Mock(pid=12345)

        train = self.raw_train.copy(deep=True)
        valid = self.raw_valid.copy(deep=True)
        test = self.raw_test.copy(deep=True)

        # Duplicate a few groups so the grouped outputs prove that init_worker keeps
        # the full group contents rather than only a single representative row.
        train = pd.concat(
            [
                train,
                train.loc[
                    (train["original_label"] == 0) & (train["dmr_ctype_label"] == 0)
                ].copy(),
            ],
            ignore_index=True,
        )
        valid = pd.concat(
            [
                valid,
                valid.loc[
                    (valid["original_label"] == 1) & (valid["dmr_ctype_label"] == 38)
                ].copy(),
                valid.loc[
                    (valid["original_label"] == 1) & (valid["dmr_ctype_label"] == 38)
                ].copy(),
            ],
            ignore_index=True,
        )
        test = pd.concat(
            [
                test,
                test.loc[
                    (test["original_label"] == 0) & (test["dmr_ctype_label"] == 10)
                ].copy(),
                test.loc[
                    (test["original_label"] == 0) & (test["dmr_ctype_label"] == 10)
                ].copy(),
                test.loc[
                    (test["original_label"] == 0) & (test["dmr_ctype_label"] == 10)
                ].copy(),
            ],
            ignore_index=True,
        )

        pseudo_bulk_generation_module.init_worker(
            splits={
                "train": train,
                "valid": valid,
                "test": test
            },
            allowed_labels=[0, 1],
            n_cells_max=5,
            n_read_per_split=123,
        )

        mocked_random_seed.assert_called_once_with(12345)
        mocked_numpy_seed.assert_called_once_with(12345)

        self.assertEqual(
            pseudo_bulk_generation_module._worker_data["allowed_labels"], [0, 1]
        )
        self.assertEqual(pseudo_bulk_generation_module._worker_data["n_cells_max"], 5)
        self.assertEqual(
            pseudo_bulk_generation_module._worker_data["n_read_per_split"], 123
        )
        self.assertEqual(
            pseudo_bulk_generation_module._worker_data["target_columns"],
            self.target_columns,
        )

        grouped_splits = pseudo_bulk_generation_module._worker_data["grouped_splits"]
        grouped_train = grouped_splits["train"]
        grouped_valid = grouped_splits["valid"]
        grouped_test = grouped_splits["test"]

        self.assertEqual(len(grouped_train.get_group((0, 0))), 2)
        self.assertEqual(len(grouped_valid.get_group((1, 38))), 3)
        self.assertEqual(len(grouped_test.get_group((0, 10))), 4)

        self.assertEqual(
            pseudo_bulk_generation_module._worker_data["dmr_sampling_variants"],
            ["uniform"],
        )

        for df in [train, valid, test]:
            self.assertIn("total_marked_cpgs", df.columns)
            self.assertIn("methylation_level", df.columns)
            self.assertIn("chromosome", df.columns)
            self.assertIn("direction", df.columns)
            self.assertTrue((df["direction"] == "U").all())


class TestWorkerTask(PseudoBulkGenerationTestBase):
    """Test the per-process batch worker behavior."""

    def test_worker_task_returns_empty_lists_for_empty_batch(self):
        """Return empty result and exception collections when no indices are assigned."""
        results, exceptions = pseudo_bulk_generation_module.worker_task(([], None))

        self.assertEqual(results, [])
        self.assertEqual(exceptions, [])

    @mock.patch.object(
        pseudo_bulk_generation_module,
        "generate_pseudo_bulk_optimized",
        return_value=(
            ["ignored"],
            [1.0] + ([0.0] * 38),
            {"train": "sub"},
            {"train": "uxm"},
        ),
    )
    @mock.patch.object(
        pseudo_bulk_generation_module,
        "random_select_with_weights",
        return_value=([0], [1.0]),
    )
    def test_worker_task_processes_each_index_in_the_batch(
        self,
        mocked_random_select,
        mocked_generate,
    ):
        """Call the selector and optimized generator once per batch item."""
        pseudo_bulk_generation_module._worker_data.update(
            {
                "allowed_labels": [0, 1],
                "n_cells_max": 4,
                "n_read_per_split": 321,
                "grouped_splits": {
                    "train": "grouped_train",
                    "valid": "grouped_valid",
                    "test": "grouped_test"
                },
                "num_labels": 39,
                "generate_uxm_inputs": True,
                "target_columns": self.target_columns,
                "dmr_sampling_variants": ["uniform"],
            }
        )

        results, exceptions = pseudo_bulk_generation_module.worker_task(([0] * 3, None))

        self.assertEqual(len(results), 3)
        self.assertEqual(exceptions, [])
        self.assertEqual(
            results,
            [([1.0] + ([0.0] * 38), {"train": "sub"}, {"train": "uxm"}, "uniform")] * 3,
        )
        self.assertEqual(mocked_random_select.call_count, 3)
        self.assertEqual(mocked_generate.call_count, 3)


class TestGeneratePseudoBulkOptimized(PseudoBulkGenerationTestBase):
    """Exercise the optimized pseudo-bulk generation helper."""

    def test_generate_pseudo_bulk_optimized_returns_three_split_outputs(self):
        """Build aggregated outputs and UXM inputs for each grouped split."""
        grouped_train = self.prepared_train.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )
        grouped_valid = self.prepared_valid.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )
        grouped_test = self.prepared_test.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )

        labels, proportions_full, subs, uxm_data = (
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=39,
                labels=[0],
                proportions=[1.0],
                grouped_splits={
                    "train": grouped_train,
                    "valid": grouped_valid,
                    "test": grouped_test
                },
                target_columns=self.target_columns,
            )
        )

        self.assertEqual(labels, [0])
        self.assertEqual(len(proportions_full), 39)
        self.assertEqual(proportions_full[0], 1.0)
        self.assertEqual(len(subs), 3)
        self.assertEqual(len(uxm_data), 3)

        for split_name, sub in subs.items():
            self.assertEqual(list(sub.columns), self.target_columns)
            self.assertEqual(len(sub), 39)

        for split_name, (sf, counts) in uxm_data.items():
            self.assertEqual(
                list(sf.columns), ["name", "direction", "pseudo_bulk_sample"]
            )
            self.assertEqual(
                list(counts.columns),
                ["name", "direction", "pseudo_bulk_sample"],
            )
            self.assertTrue((sf["pseudo_bulk_sample"] == 0.5).all())
            self.assertTrue((counts["pseudo_bulk_sample"] == 2).all())

    def test_generate_pseudo_bulk_optimized_skips_empty_grouped_split(self):
        """Skip a split entirely when all requested grouped lookups are missing."""
        grouped_train = self.prepared_train.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )
        grouped_valid = self.empty_prepared.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )
        grouped_test = self.prepared_test.groupby(
            ["original_label", "dmr_ctype_label"],
            sort=False,
        )

        labels, proportions_full, subs, uxm_data = (
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=39,
                labels=[0],
                proportions=[1.0],
                grouped_splits={
                    "train": grouped_train,
                    "valid": grouped_valid,
                    "test": grouped_test
                },
                target_columns=self.target_columns,
            )
        )

        self.assertEqual(labels, [0])
        self.assertEqual(proportions_full[0], 1.0)
        self.assertEqual(len(subs), 2)
        self.assertEqual(len(uxm_data), 2)

    def test_generate_pseudo_bulk_optimized_rejects_invalid_proportions(self):
        """Raise an assertion when optimized generation receives invalid proportions."""
        with self.assertRaises(AssertionError) as context:
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=39,
                labels=[0, 1],
                proportions=[0.2, 0.2],
                grouped_splits={},
                target_columns=self.target_columns,
            )

        self.assertIn("Proportions must sum up to one", str(context.exception))


class TestRunIosGenerationParallel(PseudoBulkGenerationTestBase):
    """Verify the parallel wrapper around worker submission and checkpointing."""

    def test_run_ios_generation_parallel_uses_default_worker_count_and_collects_results(
        self,
    ):
        """Derive the default worker count, submit batches, and collect completed outputs."""
        future_one = mock.MagicMock()
        future_one.result.return_value = (
            [("io_1", {"train": "subs_1"}, {"train": "uxm_1"}), ("io_2", {"train": "subs_2"}, {"train": "uxm_2"})],
            [],
        )

        future_two = mock.MagicMock()
        future_two.result.return_value = (
            [("io_3", {"train": "subs_3"}, {"train": "uxm_3"})],
            [("labels", "weights", "error")],
        )

        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.submit.side_effect = [future_one, future_two]

        progress_context, progress_bar = self.make_progress_bar_context()

        with mock.patch.object(
            pseudo_bulk_generation_module.mp,
            "cpu_count",
            return_value=4,
        ), mock.patch.object(
            pseudo_bulk_generation_module,
            "ProcessPoolExecutor",
            return_value=executor,
        ) as mocked_executor_class, mock.patch.object(
            pseudo_bulk_generation_module,
            "as_completed",
            return_value=[future_two, future_one],
        ), mock.patch.object(
            pseudo_bulk_generation_module,
            "tqdm",
            return_value=progress_context,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(),
        ) as mocked_open, mock.patch.object(
            pseudo_bulk_generation_module.pickle,
            "dump",
        ) as mocked_pickle_dump:
            all_ios, all_exceptions = (
                pseudo_bulk_generation_module.run_ios_generation_parallel(
                    splits={
                        "train": self.raw_train.copy(deep=True),
                        "valid": self.raw_valid.copy(deep=True),
                        "test": self.raw_test.copy(deep=True)
                    },
                    file_name="pseudo_bulk.pkl",
                    n_io_examples=3,
                    batch_size=2,
                    checkpoint_interval=10,
                )
            )

        mocked_executor_class.assert_called_once_with(
            max_workers=3,
            initializer=pseudo_bulk_generation_module.init_worker,
            initargs=(
                mock.ANY,
                list(range(39)),
                10,
                int(4.75 * 1e5),
                39,
                True,
                None,
                ["uniform"],
            ),
        )
        self.assertEqual(
            executor.submit.call_args_list,
            [
                mock.call(pseudo_bulk_generation_module.worker_task, ([0, 1], None)),
                mock.call(pseudo_bulk_generation_module.worker_task, ([2], None)),
            ],
        )

        # The progress bar tracks both successful results and captured exceptions.
        progress_bar.update.assert_any_call(2)
        progress_bar.update.assert_any_call(2)

        # Final flush saves remaining results
        mocked_open.assert_called_once_with("pseudo_bulk_3.pkl", "wb")
        mocked_pickle_dump.assert_called_once()
        self.assertEqual(all_ios, [])
        self.assertEqual(all_exceptions, [("labels", "weights", "error")])

    def test_run_ios_generation_parallel_writes_checkpoint_and_resets_accumulator(
        self,
    ):
        """Write a checkpoint file when the accumulated result count hits the interval."""
        future = mock.MagicMock()
        future.result.return_value = (
            [("io_1", {"train": "subs_1"}, {"train": "uxm_1"}), ("io_2", {"train": "subs_2"}, {"train": "uxm_2"})],
            [],
        )

        executor = mock.MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.submit.return_value = future

        progress_context, progress_bar = self.make_progress_bar_context()

        with mock.patch.object(
            pseudo_bulk_generation_module,
            "ProcessPoolExecutor",
            return_value=executor,
        ), mock.patch.object(
            pseudo_bulk_generation_module,
            "as_completed",
            return_value=[future],
        ), mock.patch.object(
            pseudo_bulk_generation_module,
            "tqdm",
            return_value=progress_context,
        ), mock.patch(
            "builtins.open",
            mock.mock_open(),
        ) as mocked_open, mock.patch.object(
            pseudo_bulk_generation_module.pickle,
            "dump",
        ) as mocked_pickle_dump:
            all_ios, all_exceptions = (
                pseudo_bulk_generation_module.run_ios_generation_parallel(
                    splits={
                        "train": self.raw_train.copy(deep=True),
                        "valid": self.raw_valid.copy(deep=True),
                        "test": self.raw_test.copy(deep=True)
                    },
                    file_name="pseudo_bulk.pkl",
                    n_io_examples=2,
                    n_workers=1,
                    batch_size=2,
                    checkpoint_interval=2,
                    start_checkpoint_idx=4,
                )
            )

        progress_bar.update.assert_called_once_with(2)
        mocked_open.assert_called_once_with("pseudo_bulk_6.pkl", "wb")
        mocked_pickle_dump.assert_called_once()
        self.assertEqual(
            mocked_pickle_dump.call_args.args[0],
            [("io_1", {"train": "subs_1"}, {"train": "uxm_1"}), ("io_2", {"train": "subs_2"}, {"train": "uxm_2"})],
        )
        self.assertEqual(all_ios, [])
        self.assertEqual(all_exceptions, [])


class TestRandomSelectWithWeights(unittest.TestCase):
    """Exercise the random label sampler used by pseudo-bulk workers."""

    @mock.patch.object(pseudo_bulk_generation_module.random, "sample")
    @mock.patch.object(pseudo_bulk_generation_module.random, "randint")
    def test_random_select_with_weights_returns_empty_lists_for_empty_input_or_non_positive_n(
        self,
        mocked_randint,
        mocked_sample,
    ):
        """Return empty outputs immediately when sampling cannot proceed."""
        test_cases = [
            {"elements": [], "n": 3},
            {"elements": ["a", "b"], "n": 0},
            {"elements": ["a", "b"], "n": -1},
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                selected, weights = (
                    pseudo_bulk_generation_module.random_select_with_weights(
                        elements=test_case["elements"],
                        n=test_case["n"],
                    )
                )

                self.assertEqual(selected, [])
                self.assertEqual(weights, [])

        mocked_randint.assert_not_called()
        mocked_sample.assert_not_called()

    @mock.patch.object(
        pseudo_bulk_generation_module.random,
        "random",
        side_effect=[0.2, 0.8],
    )
    @mock.patch.object(
        pseudo_bulk_generation_module.random,
        "sample",
        return_value=["a", "c"],
    )
    @mock.patch.object(pseudo_bulk_generation_module.random, "randint", return_value=2)
    def test_random_select_with_weights_normalizes_weights(
        self,
        mocked_randint,
        mocked_sample,
        mocked_random,
    ):
        """Select labels and normalize their randomly generated weights to one."""
        selected, weights = pseudo_bulk_generation_module.random_select_with_weights(
            elements=["a", "b", "c"],
            n=3,
        )

        mocked_randint.assert_called_once_with(1, 3)
        mocked_sample.assert_called_once_with(["a", "b", "c"], 2)
        self.assertEqual(selected, ["a", "c"])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(weights[0], 0.2)
        self.assertAlmostEqual(weights[1], 0.8)

    @mock.patch.object(pseudo_bulk_generation_module.random, "randint", return_value=3)
    def test_random_select_with_weights_caps_requested_size_to_available_elements(
        self,
        mocked_randint,
    ):
        """Cap the requested sample size so selection never exceeds the available elements."""
        with mock.patch.object(
            pseudo_bulk_generation_module.random,
            "sample",
            return_value=["only_one"],
        ) as mocked_sample:
            selected, weights = (
                pseudo_bulk_generation_module.random_select_with_weights(
                    elements=["only_one"],
                    n=3,
                )
            )

        mocked_randint.assert_called_once_with(1, 1)
        mocked_sample.assert_called_once_with(["only_one"], 1)
        self.assertEqual(selected, ["only_one"])
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights[0], 1.0)


class TestDMRSamplingStrategies(PseudoBulkGenerationTestBase):
    """Test uniform and random DMR sampling in generate_pseudo_bulk_optimized."""

    def _make_grouped_splits(self):
        """Build grouped splits from prepared data."""
        return {
            "train": self.prepared_train.groupby(
                ["original_label", "dmr_ctype_label"], sort=False,
            ),
        }

    def test_uniform_dmr_sampling_distributes_equally(self):
        """Uniform sampling gives equal reads per DMR group."""
        labels, proportions_full, subs, uxm_data = (
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=39,
                labels=[0],
                proportions=[1.0],
                grouped_splits=self._make_grouped_splits(),
                target_columns=self.target_columns,
                dmr_sampling="uniform",
            )
        )
        self.assertEqual(labels, [0])
        self.assertIn("train", subs)
        self.assertEqual(len(subs["train"]), 39)

    def test_random_dmr_sampling_produces_output(self):
        """Random sampling produces valid output with non-uniform per-DMR counts."""
        labels, proportions_full, subs, uxm_data = (
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=390,
                labels=[0],
                proportions=[1.0],
                grouped_splits=self._make_grouped_splits(),
                target_columns=self.target_columns,
                dmr_sampling="random",
            )
        )
        self.assertEqual(labels, [0])
        self.assertIn("train", subs)
        # Random allocation should still produce some rows
        self.assertGreater(len(subs["train"]), 0)

    def test_invalid_dmr_sampling_raises(self):
        """An invalid dmr_sampling value raises AssertionError."""
        with self.assertRaises(AssertionError):
            pseudo_bulk_generation_module.generate_pseudo_bulk_optimized(
                total_samples=39,
                labels=[0],
                proportions=[1.0],
                grouped_splits=self._make_grouped_splits(),
                target_columns=self.target_columns,
                dmr_sampling="invalid",
            )


class TestComputeSamplesPerDmr(unittest.TestCase):
    """Test the uniform and random DMR allocation helpers."""

    def test_uniform_returns_dict_of_ints(self):
        result = pseudo_bulk_generation_module._compute_samples_per_dmr_uniform(
            labels=[0, 1], n_samples_list=[390, 195], num_labels=39,
        )
        self.assertEqual(result[0], 10)  # 390 / 39
        self.assertEqual(result[1], 5)   # 195 / 39

    def test_random_returns_dict_of_lists(self):
        result = pseudo_bulk_generation_module._compute_samples_per_dmr_random(
            labels=[0], n_samples_list=[390], num_labels=39,
        )
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), 39)
        # Total should be close to 390 (integer rounding may lose a few)
        self.assertLessEqual(sum(result[0]), 390)
        self.assertGreater(sum(result[0]), 350)  # not wildly off


class TestWorkerTaskMultipleVariants(PseudoBulkGenerationTestBase):
    """Test worker_task with multiple DMR sampling variants."""

    @mock.patch.object(
        pseudo_bulk_generation_module,
        "generate_pseudo_bulk_optimized",
        return_value=(
            ["ignored"],
            [1.0] + ([0.0] * 38),
            {"train": "sub"},
            {"train": "uxm"},
        ),
    )
    @mock.patch.object(
        pseudo_bulk_generation_module,
        "random_select_with_weights",
        return_value=([0], [1.0]),
    )
    def test_worker_task_produces_two_results_per_index_with_both_variants(
        self,
        mocked_random_select,
        mocked_generate,
    ):
        """Each index produces one result per variant when two are configured."""
        pseudo_bulk_generation_module._worker_data.update(
            {
                "allowed_labels": [0],
                "n_cells_max": 4,
                "n_read_per_split": 100,
                "grouped_splits": {"train": "grouped"},
                "num_labels": 39,
                "generate_uxm_inputs": False,
                "target_columns": self.target_columns,
                "dmr_sampling_variants": ["uniform", "random"],
            }
        )

        results, exceptions = pseudo_bulk_generation_module.worker_task(([0, 1], None))

        # 2 indices × 2 variants = 4 results
        self.assertEqual(len(results), 4)
        self.assertEqual(exceptions, [])

        # Check variant tags alternate
        variants = [r[3] for r in results]
        self.assertEqual(variants, ["uniform", "random", "uniform", "random"])

        # generate called once per variant per index = 4
        self.assertEqual(mocked_generate.call_count, 4)
        # random_select called once per index (not per variant) = 2
        self.assertEqual(mocked_random_select.call_count, 2)


class TestConsolidateVariantAware(PseudoBulkGenerationTestBase):
    """Test variant-aware consolidation of IO pickles."""

    def test_consolidate_variant_aware_separates_features_by_variant(self):
        """Variant-tagged tuples produce features_{split}_{variant} keys."""
        import tempfile
        import os

        pred_cols = [f"prediction_{i}_wavg" for i in range(39)]
        df = pd.DataFrame({c: [0.1] for c in pred_cols})

        ios = [
            ([1.0] + [0.0] * 38, {"train": df.copy()}, None, "uniform"),
            ([1.0] + [0.0] * 38, {"train": df.copy()}, None, "random"),
            ([0.0, 1.0] + [0.0] * 37, {"train": df.copy()}, None, "uniform"),
            ([0.0, 1.0] + [0.0] * 37, {"train": df.copy()}, None, "random"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "ios_1.pkl")
            import pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(ios, f)

            output_path = os.path.join(tmpdir, "result.npz")
            result = pseudo_bulk_generation_module.consolidate_ios_pickles(
                ios_dir=tmpdir,
                output_path=output_path,
                num_labels=39,
            )

        self.assertIn("features_train_uniform", result)
        self.assertIn("features_train_random", result)
        self.assertIn("proportions_train", result)

        # 2 proportion vectors (deduplicated from 4 ios with 2 variants)
        self.assertEqual(result["proportions_train"].shape[0], 2)
        # 2 feature matrices per variant
        self.assertEqual(result["features_train_uniform"].shape[0], 2)
        self.assertEqual(result["features_train_random"].shape[0], 2)

    def test_consolidate_legacy_uses_original_key_scheme(self):
        """Legacy 3-element tuples produce proportions + features_{split} keys."""
        import tempfile
        import os

        pred_cols = [f"prediction_{i}_wavg" for i in range(39)]
        df = pd.DataFrame({c: [0.1] for c in pred_cols})

        ios = [
            ([1.0] + [0.0] * 38, {"train": df.copy()}, None),
            ([0.0, 1.0] + [0.0] * 37, {"train": df.copy()}, None),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "ios_1.pkl")
            import pickle
            with open(pkl_path, "wb") as f:
                pickle.dump(ios, f)

            output_path = os.path.join(tmpdir, "result.npz")
            result = pseudo_bulk_generation_module.consolidate_ios_pickles(
                ios_dir=tmpdir,
                output_path=output_path,
                num_labels=39,
            )

        self.assertIn("proportions", result)
        self.assertIn("features_train", result)
        self.assertNotIn("proportions_train", result)
        self.assertEqual(result["proportions"].shape[0], 2)
        self.assertEqual(result["features_train"].shape[0], 2)
