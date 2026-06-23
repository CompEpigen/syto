"""Interactive config-generation wizard.

``run_wizard`` is re-exported lazily so that importing individual submodules
(e.g. ``App.wizard.validators``) does not pull in ``questionary``/``rich``.
"""


def run_wizard() -> None:
    """Entry point for ``--task create_config``. See ``App.wizard.runner``."""
    from App.wizard.runner import run_wizard as _run

    return _run()
