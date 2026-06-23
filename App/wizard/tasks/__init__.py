"""Task wizard registry."""

TASK_REGISTRY: dict = {}


def _load_tasks() -> None:
    """Import task modules so they self-register in TASK_REGISTRY."""
    from App.wizard.tasks import inference  # noqa: F401


_load_tasks()
