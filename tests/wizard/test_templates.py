import unittest

import yaml

from App.main import validate_config
from App.wizard import renderer, templates
from App.wizard.fields import FieldSpec
from App.wizard.tasks import TASK_REGISTRY


class TestTaskListing(unittest.TestCase):
    def test_pipeline_order_first_and_all_registered_present(self):
        tasks = templates.ordered_tasks()
        self.assertEqual(tasks[: len(templates.TASK_ORDER)], templates.TASK_ORDER)
        self.assertEqual(set(tasks), set(TASK_REGISTRY))

    def test_needs_classifier(self):
        self.assertTrue(templates.needs_classifier(["fit_calibration", "inference"]))
        self.assertFalse(
            templates.needs_classifier(["fit_calibration", "fit_deconvolution"])
        )


class TestTemplateValue(unittest.TestCase):
    def test_default_is_kept(self):
        spec = FieldSpec(key="k", label="k", kind="int", default=39)
        self.assertEqual(templates.template_value(spec), 39)

    def test_select_without_default_collapses_to_first_choice(self):
        spec = FieldSpec(key="k", label="k", kind="select", choices=["a", "b"])
        self.assertEqual(templates.template_value(spec), "a")

    def test_bool_without_default_is_false(self):
        spec = FieldSpec(key="k", label="k", kind="bool")
        self.assertIs(templates.template_value(spec), False)

    def test_path_without_default_is_placeholder(self):
        spec = FieldSpec(key="data_path", label="d", kind="path")
        self.assertEqual(templates.template_value(spec), "<data_path>")


class TestCountPlaceholders(unittest.TestCase):
    def test_counts_nested_lists_and_dicts(self):
        config = {
            "a": "<a>",
            "b": 1,
            "c": {"d": "<d>"},
            "e": [{"f": "<f>"}, {"f": "kept"}],
        }
        self.assertEqual(templates.count_placeholders(config), 3)


class TestHeader(unittest.TestCase):
    def test_includes_run_command(self):
        text = templates.header("inference", "cfg/inference.yaml")
        self.assertIn(
            "python App/main.py --task inference --config cfg/inference.yaml", text
        )

    def test_classifier_named_only_for_classifier_tasks(self):
        self.assertIn(
            "Target classifier: dismir",
            templates.header("inference", "x.yaml", "dismir"),
        )
        self.assertNotIn(
            "Target classifier",
            templates.header("fit_calibration", "x.yaml", "dismir"),
        )


class TestBuildTemplateAllTasks(unittest.TestCase):
    """Every registered task must yield a runnable-shaped, parseable template."""

    def test_round_trips_through_yaml(self):
        for task in templates.ordered_tasks():
            with self.subTest(task=task):
                config = templates.build_template(TASK_REGISTRY[task](), "methylbert")
                self.assertEqual(yaml.safe_load(renderer.dump_yaml(config)), config)

    def test_satisfies_main_validate_config(self):
        # Placeholders are still values, so the required-field check must pass:
        # a template is only missing content, never structure.
        for task in templates.ordered_tasks():
            with self.subTest(task=task):
                config = templates.build_template(TASK_REGISTRY[task](), "methylbert")
                validate_config(config, task)

    def test_every_task_writes_output_dir_at_the_root(self):
        # App/main.py reads config["output_dir"] for every task before dispatch,
        # so a nested-only output dir (e.g. under an `output:` section) fails
        # before the pipeline is even constructed.
        for task in templates.ordered_tasks():
            with self.subTest(task=task):
                config = templates.build_template(TASK_REGISTRY[task](), "methylbert")
                self.assertIn("output_dir", config)

    def test_has_something_to_fill_in(self):
        for task in templates.ordered_tasks():
            with self.subTest(task=task):
                config = templates.build_template(TASK_REGISTRY[task](), "methylbert")
                self.assertGreater(templates.count_placeholders(config), 0)


class TestClassifierPropagation(unittest.TestCase):
    def test_classifier_fit_materialises_the_chosen_architecture(self):
        cfg = templates.build_template(TASK_REGISTRY["classifier_fit"](), "dismir")
        self.assertEqual(cfg["model"]["architecture"], "dismir")
        # dismir-only training block, no HuggingFace training_args
        self.assertIn("epochs", cfg["training"])
        self.assertNotIn("training_args", cfg["training"])

    def test_inference_materialises_the_chosen_classifier(self):
        cfg = templates.build_template(TASK_REGISTRY["inference"](), "epigenbert2")
        self.assertEqual(cfg["classifier"]["classifier_type"], "epigenbert2")
        self.assertIn("foundation_model", cfg["classifier"])

    def test_unknown_classifier_falls_back_to_first_choice(self):
        cfg = templates.build_template(TASK_REGISTRY["classifier_fit"](), "nonesuch")
        self.assertIn(cfg["model"]["architecture"], templates.CLASSIFIERS)

    def test_every_offered_classifier_reaches_generate_pseudobulk(self):
        for classifier in templates.CLASSIFIERS:
            with self.subTest(classifier=classifier):
                cfg = templates.build_template(
                    TASK_REGISTRY["generate_pseudobulk"](), classifier
                )
                self.assertEqual(cfg["classifier_type"], classifier)

    def test_generate_pseudobulk_falls_back_on_unknown_classifier(self):
        from App.wizard.tasks.generate_pseudobulk import PB_CLASSIFIERS

        cfg = templates.build_template(
            TASK_REGISTRY["generate_pseudobulk"](), "nonesuch"
        )
        self.assertIn(cfg["classifier_type"], PB_CLASSIFIERS)


class TestSectionsArePrePopulated(unittest.TestCase):
    def test_inference_lists_every_method_and_baseline(self):
        from App.wizard.tasks.baselines_common import BASELINES
        from App.wizard.tasks.inference import SYTO_DECONVOLVERS

        cfg = templates.build_template(TASK_REGISTRY["inference"](), "methylbert")
        methods = cfg["deconvolution"]["syto"]["methods"]
        self.assertEqual(len(methods), len(SYTO_DECONVOLVERS))
        self.assertTrue(all(m["enabled"] for m in methods))
        self.assertEqual(
            [b["model"] for b in cfg["deconvolution"]["baselines"]], BASELINES
        )

    def test_fit_tasks_list_every_deconvolver_and_split(self):
        from App.wizard.tasks.deconvolver_common import (
            DECONVOLVER_ORDER,
            DEFAULT_SPLITS,
        )

        for task in ("fit_deconvolution", "fit_calibration"):
            with self.subTest(task=task):
                cfg = templates.build_template(TASK_REGISTRY[task]())
                self.assertEqual(
                    [d["name"] for d in cfg["deconvolvers"]], DECONVOLVER_ORDER
                )
                self.assertEqual(cfg["splits"], DEFAULT_SPLITS)

    def test_deconvolute_pseudobulk_lists_every_baseline(self):
        from App.wizard.tasks.baselines_common import BASELINES

        cfg = templates.build_template(TASK_REGISTRY["deconvolute_pseudobulk"]())
        self.assertEqual([b["model"] for b in cfg["baselines"]], BASELINES)

    def test_generate_pseudobulk_stubs_every_split(self):
        from App.wizard.tasks.generate_pseudobulk import SPLITS

        cfg = templates.build_template(TASK_REGISTRY["generate_pseudobulk"](), "dismir")
        self.assertEqual(sorted(cfg["split_information"]), sorted(SPLITS))
        for split, info in cfg["split_information"].items():
            self.assertEqual(
                info["target_proportions_path"],
                f"<{split}_target_proportions_path>",
            )

    def test_generate_pseudobulk_split_shape_follows_input_type(self):
        # raw_splits reads the reads from the shared top-level data_path, so a
        # per-split data_path belongs to pre_predicted templates only.
        cfg = templates.build_template(TASK_REGISTRY["generate_pseudobulk"](), "dismir")
        per_split_has_data_path = any(
            "data_path" in info for info in cfg["split_information"].values()
        )
        self.assertEqual(per_split_has_data_path, cfg["input_type"] == "pre_predicted")
        if cfg["input_type"] == "raw_splits":
            self.assertIn("data_path", cfg)

    def test_template_entries_match_the_interactive_defaults(self):
        # The uxm/celfie template entries must not drift from what the
        # interactive sub-flow would produce for the same answers.
        from App.wizard.tasks.baselines_common import template_baseline_entries

        by_model = {e["model"]: e for e in template_baseline_entries()}
        self.assertEqual(by_model["uxm"]["ignore_cells"], ["Megakaryocytes"])
        self.assertEqual(by_model["celfie"]["num_iterations"], 400)
        self.assertEqual(by_model["epidish"]["method"], "RPC")


if __name__ == "__main__":
    unittest.main()
