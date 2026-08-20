"""Reusable input validators for the config wizard.

Each validator takes an already-coerced value and returns ``None`` when the
value is acceptable, or a human-readable error message string otherwise. The
engine shows the message and re-prompts.
"""

import os


def path_exists(value: str) -> str | None:
    if not os.path.exists(value):
        return f"Path does not exist: {value}"
    return None


def path_exists_or_blank(value: str) -> str | None:
    if value is None or value.strip() == "":
        return None
    return path_exists(value)


def pseudobulk_store(value: str) -> str | None:
    """Accept either a consolidated ``.h5`` file or a columnar store directory."""
    from syto.data.pseudobulk_store import detect_store_format

    try:
        detect_store_format(value)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    return None


def positive_int(value: int) -> str | None:
    if value <= 0:
        return "Value must be a positive integer."
    return None


def non_negative_int(value: int) -> str | None:
    if value < 0:
        return "Value must be zero or a positive integer."
    return None


def positive_float(value: float) -> str | None:
    if value <= 0:
        return "Value must be a positive number."
    return None


def non_negative_float(value: float) -> str | None:
    if value < 0:
        return "Value must be zero or a positive number."
    return None


def non_empty(value: str) -> str | None:
    if value is None or value.strip() == "":
        return "Value must not be empty."
    return None
