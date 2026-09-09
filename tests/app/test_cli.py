"""The ``syto`` command layer: subcommand dispatch, aliases, and overrides.

Covers the parsing/dispatch surface only - each task's pipeline is exercised by
its own test module.  The one thing worth stating up front: task names keep
their underscore spelling everywhere below the CLI (``validate_config``, the
configs on disk), while the CLI itself speaks hyphens.
"""

import json
import logging
import os
import tempfile
import unittest
from unittest.mock import patch

from syto.app import cli


class TestTaskNameSpellings(unittest.TestCase):
    def test_every_task_is_accepted_in_both_spellings(self):
        parser = cli.build_parser()

        for task in cli.ALL_TASKS:
            for spelling in (task, task.replace("_", "-")):
                with self.subTest(spelling=spelling):
                    args = parser.parse_args([spelling])
                    self.assertEqual(cli.normalize_task(args.task), task)

    def test_command_table_matches_validate_config(self):
        """A task the CLI dispatches must be one validate_config knows."""
        for task in cli.COMMANDS:
            with self.subTest(task=task):
                with self.assertRaises(ValueError) as ctx:
                    cli.validate_config({}, task)
                # Missing-field complaint, not "Unknown task".
                self.assertNotIn("Unknown task", str(ctx.exception))

    def test_wizard_task_is_not_in_the_dispatch_table(self):
        # create_config takes no --config, so main() handles it separately.
        self.assertNotIn(cli.WIZARD_TASK, cli.COMMANDS)
        self.assertIn(cli.WIZARD_TASK, cli.ALL_TASKS)


class TestModelOverride(unittest.TestCase):
    def test_every_accepted_architecture_can_be_selected(self):
        """--model must offer whatever validate_config accepts, or the override
        cannot reach half the classifiers."""
        parser = cli.build_parser()

        for architecture in cli.MODEL_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                args = parser.parse_args(["classifier-fit", "--model", architecture])
                self.assertEqual(args.model, architecture)

                config = {"model": {"architecture": architecture}}
                cli.apply_overrides(config, args, logging.getLogger("test"))
                # Accepted by the validator, i.e. not "Unknown model architecture".
                with self.assertRaises(ValueError) as ctx:
                    cli.validate_config(config, "classifier_fit")
                self.assertNotIn("Unknown model architecture", str(ctx.exception))

    def test_an_unknown_architecture_is_rejected_at_the_command_line(self):
        parser = cli.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["classifier-fit", "--model", "nonesuch"])


class TestLegacyTaskAlias(unittest.TestCase):
    def test_separate_value_becomes_a_subcommand(self):
        argv, rewritten = cli._rewrite_task_alias(
            ["--task", "fit_calibration", "--config", "c.yaml"]
        )

        self.assertTrue(rewritten)
        self.assertEqual(argv, ["fit-calibration", "--config", "c.yaml"])

    def test_equals_form_becomes_a_subcommand(self):
        argv, rewritten = cli._rewrite_task_alias(
            ["--task=inference", "--config", "c.yaml"]
        )

        self.assertTrue(rewritten)
        self.assertEqual(argv, ["inference", "--config", "c.yaml"])

    def test_flags_before_the_task_are_preserved(self):
        argv, rewritten = cli._rewrite_task_alias(
            ["--verbose", "--task", "inference", "--config", "c.yaml"]
        )

        self.assertTrue(rewritten)
        self.assertEqual(argv, ["inference", "--verbose", "--config", "c.yaml"])

    def test_subcommand_form_is_left_alone(self):
        argv, rewritten = cli._rewrite_task_alias(["inference", "--config", "c.yaml"])

        self.assertFalse(rewritten)
        self.assertEqual(argv, ["inference", "--config", "c.yaml"])

    def test_dangling_task_flag_is_left_for_argparse_to_reject(self):
        argv, rewritten = cli._rewrite_task_alias(["--task"])

        self.assertFalse(rewritten)
        self.assertEqual(argv, ["--task"])


class TestApplyOverrides(unittest.TestCase):
    def _args(self, **kwargs):
        parser = cli.build_parser()
        return parser.parse_args(["inference"] + list(kwargs.pop("argv", [])))

    def test_nested_keys_are_created_when_absent(self):
        config = {}
        args = self._args(argv=["--bam", "s.bam", "--mlflow-uri", "http://mlflow"])

        cli.apply_overrides(config, args, logging.getLogger("test"))

        self.assertEqual(config["input"], {"type": "bam", "bam_path": "s.bam"})
        self.assertEqual(config["mlflow"]["tracking_uri"], "http://mlflow")

    def test_existing_sections_are_updated_not_replaced(self):
        config = {"input": {"reference_path": "ref.fa"}}
        args = self._args(argv=["--bam", "s.bam"])

        cli.apply_overrides(config, args, logging.getLogger("test"))

        self.assertEqual(config["input"]["reference_path"], "ref.fa")
        self.assertEqual(config["input"]["bam_path"], "s.bam")

    def test_unset_flags_leave_the_config_untouched(self):
        config = {"output_dir": "keep"}
        args = self._args()

        cli.apply_overrides(config, args, logging.getLogger("test"))

        self.assertEqual(config, {"output_dir": "keep"})


class TestMainDispatch(unittest.TestCase):
    def setUp(self):
        # main() reconfigures the root logger with basicConfig(force=True),
        # which is right for a CLI process and ruinous inside a test run - it
        # closes the handlers the test runner installed.  Hand it a plain
        # logger instead; the messages still propagate for assertLogs.
        logging_patch = patch.object(
            cli, "setup_logging", lambda *a, **k: logging.getLogger("syto.app.cli")
        )
        logging_patch.start()
        self.addCleanup(logging_patch.stop)

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = os.path.join(self.tmp.name, "out")
        self.config_path = os.path.join(self.tmp.name, "cfg.yaml")
        labels_path = os.path.join(self.tmp.name, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({"0": "ct_a"}, f)
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(
                "output_dir: {}\nlabels_dict_path: {}\n"
                "deconvolvers:\n  - name: epidish\n    predictions_dir: /tmp/p\n".format(
                    self.output_dir, labels_path
                )
            )

    def _run(self, argv, runner):
        with patch.dict(
            cli.COMMANDS, {"fit_calibration": (runner, "test")}, clear=False
        ):
            return cli.main(argv)

    def test_subcommand_runs_the_matching_pipeline(self):
        calls = []

        code = self._run(
            ["fit-calibration", "--config", self.config_path],
            lambda config, logger: calls.append(config),
        )

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["output_dir"], self.output_dir)

    def test_legacy_task_flag_reaches_the_same_pipeline(self):
        calls = []

        code = self._run(
            ["--task", "fit_calibration", "--config", self.config_path],
            lambda config, logger: calls.append(config),
        )

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)

    def test_legacy_task_flag_warns(self):
        with self.assertLogs("syto.app.cli", level="WARNING") as logs:
            self._run(
                ["--task", "fit_calibration", "--config", self.config_path],
                lambda config, logger: None,
            )

        self.assertIn("deprecated", "\n".join(logs.output))

    def test_config_is_copied_next_to_the_results(self):
        self._run(
            ["fit-calibration", "--config", self.config_path],
            lambda config, logger: None,
        )

        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "cfg.yaml")))

    def test_dry_run_validates_without_running(self):
        calls = []

        code = self._run(
            ["fit-calibration", "--config", self.config_path, "--dry-run"],
            lambda config, logger: calls.append(config),
        )

        self.assertEqual(code, 0)
        self.assertEqual(calls, [])

    def test_command_line_overrides_reach_the_pipeline(self):
        calls = []
        other_out = os.path.join(self.tmp.name, "elsewhere")

        self._run(
            [
                "fit-calibration",
                "--config",
                self.config_path,
                "--output-dir",
                other_out,
            ],
            lambda config, logger: calls.append(config),
        )

        self.assertEqual(calls[0]["output_dir"], other_out)

    def test_a_failing_pipeline_exits_nonzero(self):
        def boom(config, logger):
            raise RuntimeError("pipeline blew up")

        code = self._run(["fit-calibration", "--config", self.config_path], boom)

        self.assertEqual(code, 1)

    def test_invalid_config_exits_nonzero_without_running(self):
        calls = []
        empty = os.path.join(self.tmp.name, "empty.yaml")
        with open(empty, "w", encoding="utf-8") as f:
            f.write("output_dir: /tmp/x\n")  # no labels_dict_path

        code = self._run(
            ["fit-calibration", "--config", empty],
            lambda config, logger: calls.append(config),
        )

        self.assertEqual(code, 1)
        self.assertEqual(calls, [])

    def test_missing_config_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["fit-calibration"])

        self.assertEqual(ctx.exception.code, 2)

    def test_no_command_prints_help(self):
        self.assertEqual(cli.main([]), 1)

    def test_create_config_launches_the_wizard(self):
        with patch("syto.app.wizard.run_wizard") as wizard:
            code = cli.main(["create-config"])

        self.assertEqual(code, 0)
        wizard.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
