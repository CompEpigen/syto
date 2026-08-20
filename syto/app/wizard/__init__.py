"""Interactive config-generation wizard.

``run_wizard`` is re-exported lazily so that importing individual submodules
(e.g. ``syto.app.wizard.validators``) does not pull in ``questionary``/``rich``.
"""


def run_wizard() -> None:
    """Entry point for ``syto create-config``. See ``syto.app.wizard.runner``."""
    from syto.app.wizard.runner import run_wizard as _run

    return _run()
