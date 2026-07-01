"""Top-level orchestration for ``--task create_config``."""

from App.wizard import renderer
from App.wizard.engine import WizardEngine
from App.wizard.tasks import TASK_REGISTRY


def _default_select_task():
    import questionary

    return questionary.select(
        "Which task config do you want to create?",
        choices=sorted(TASK_REGISTRY.keys()),
    ).ask()


def _default_confirm_save():
    import questionary

    return questionary.confirm("Save this config?", default=True).ask()


def _default_ask_path(default: str):
    import questionary

    return questionary.path("Save to:", default=default).ask()


def run_wizard(engine=None, select_task=None, confirm_save=None, ask_path=None):
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
