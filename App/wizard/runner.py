"""Top-level orchestration for ``--task create_config``.

Two ways in, chosen up front:

``template`` mode
    Pick any number of tasks, name the classifier they target, and get one
    editable skeleton per task with placeholders where paths go. Nothing is
    validated against the filesystem because nothing points at it yet.

``fine-grained`` mode
    The original flow: pick one task and answer every question, with paths
    validated as they are entered, producing a ready-to-run config.
"""

import os

from App.wizard import renderer, templates
from App.wizard.engine import WizardEngine
from App.wizard.tasks import TASK_REGISTRY

MODE_TEMPLATE = "Templates — generate configs with placeholders to fill in later"
MODE_FINE = "Fine-grained — answer every question now for a single task"


# ── Default (interactive) prompts; every one is injectable for tests ─────────


def _default_select_mode():
    import questionary

    return questionary.select(
        "How do you want to create configs?",
        choices=[MODE_TEMPLATE, MODE_FINE],
    ).ask()


def _default_select_task():
    import questionary

    return questionary.select(
        "Which task config do you want to create?",
        choices=sorted(TASK_REGISTRY.keys()),
    ).ask()


def _default_select_tasks(choices):
    import questionary

    return (
        questionary.checkbox(
            "Which task configs do you want templates for?", choices=choices
        ).ask()
        or []
    )


def _default_select_classifier(choices):
    import questionary

    return questionary.select(
        "Which classifier do these target?", choices=choices
    ).ask()


def _default_confirm_save():
    import questionary

    return questionary.confirm("Save this config?", default=True).ask()


def _default_ask_path(default: str):
    import questionary

    return questionary.path("Save to:", default=default).ask()


def _default_ask_dir(default: str):
    import questionary

    return questionary.path("Write the templates to:", default=default).ask()


def _default_confirm_overwrite(path: str):
    import questionary

    return questionary.confirm(f"{path} exists — overwrite?", default=False).ask()


# ── Modes ────────────────────────────────────────────────────────────────────


def run_task_wizard(engine=None, select_task=None, confirm_save=None, ask_path=None):
    """Fine-grained mode: one task, every question asked, one config written."""
    engine = engine or WizardEngine()
    select_task = select_task or _default_select_task
    confirm_save = confirm_save or _default_confirm_save
    ask_path = ask_path or _default_ask_path

    task_name = select_task()
    wizard = TASK_REGISTRY[task_name]()

    answers = engine.run(wizard.field_specs())
    config = wizard.build_config(answers)

    renderer.render_preview(config)
    if not confirm_save():
        print("Not saved.")
        return None

    path = ask_path(renderer.default_save_path(task_name))
    renderer.save_config(config, path)
    renderer.print_run_command(task_name, path)
    return path


def run_template_mode(
    select_tasks=None, select_classifier=None, ask_dir=None, confirm_overwrite=None
):
    """Template mode: several tasks at once, placeholders instead of paths.

    Returns the list of files written (empty when nothing was selected, or when
    every target already existed and the user declined to overwrite).
    """
    select_tasks = select_tasks or _default_select_tasks
    select_classifier = select_classifier or _default_select_classifier
    ask_dir = ask_dir or _default_ask_dir
    confirm_overwrite = confirm_overwrite or _default_confirm_overwrite

    tasks = select_tasks(templates.ordered_tasks())
    if not tasks:
        print("No task selected — nothing generated.")
        return []

    classifier = None
    if templates.needs_classifier(tasks):
        classifier = select_classifier(templates.CLASSIFIERS)

    out_dir = ask_dir(renderer.default_template_dir())
    written: list[tuple[str, str, int]] = []
    for task in tasks:
        config = templates.build_template(TASK_REGISTRY[task](), classifier)
        path = os.path.join(out_dir, f"{task}.yaml")
        if os.path.exists(path) and not confirm_overwrite(path):
            print(f"  · skipped {path} (already exists)")
            continue
        renderer.save_config(
            config, path, header=templates.header(task, path, classifier)
        )
        written.append((task, path, templates.count_placeholders(config)))

    _print_template_summary(out_dir, written)
    return [path for _, path, _ in written]


def _print_template_summary(out_dir: str, written: list[tuple[str, str, int]]) -> None:
    if not written:
        print("Nothing written.")
        return
    print(f"\nWrote {len(written)} template(s) to {out_dir}:")
    width = max(len(os.path.basename(path)) for _, path, _ in written)
    for _, path, n_placeholders in written:
        name = os.path.basename(path).ljust(width)
        print(f"  {name}  {n_placeholders} placeholder(s) to fill in")
    print("\nFill in every <placeholder>, then run e.g.:")
    task, path, _ = written[0]
    print(f"  python App/main.py --task {task} --config {path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def run_wizard(
    engine=None,
    select_task=None,
    confirm_save=None,
    ask_path=None,
    select_mode=None,
    select_tasks=None,
    select_classifier=None,
    ask_dir=None,
    confirm_overwrite=None,
):
    """Ask which mode to use, then hand off. See the module docstring."""
    select_mode = select_mode or _default_select_mode
    if select_mode() == MODE_TEMPLATE:
        return run_template_mode(
            select_tasks=select_tasks,
            select_classifier=select_classifier,
            ask_dir=ask_dir,
            confirm_overwrite=confirm_overwrite,
        )
    return run_task_wizard(
        engine=engine,
        select_task=select_task,
        confirm_save=confirm_save,
        ask_path=ask_path,
    )
