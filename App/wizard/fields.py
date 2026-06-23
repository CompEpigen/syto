"""Declarative specification of a single wizard question."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FieldSpec:
    """One wizard question.

    Parameters
    ----------
    key:
        Dot-notation path written into the flat ``answers`` dict, e.g.
        ``"classifier.classifier_type"``. Nesting into the final YAML happens
        later in the task's ``build_config``.
    kind:
        One of ``"text"``, ``"select"``, ``"path"``, ``"bool"``, ``"int"``,
        ``"float"``, ``"list_section"``.
    default:
        Pre-populated value shown to the user; also used verbatim when the field
        is skipped (``when`` is False, or it is an ``expert`` field and the user
        declined the expert section).
    choices:
        Options for ``kind == "select"``.
    validate:
        Optional ``value -> str | None`` validator (see ``validators``).
    when:
        Optional ``answers -> bool`` predicate. When it returns False the field
        is skipped and ``default`` is recorded.
    tier:
        ``"core"`` (always asked) or ``"expert"`` (asked only if the user opts
        into the advanced section).
    help:
        Optional one-line hint.
    handler:
        For ``kind == "list_section"`` only: a ``(engine, answers) -> None``
        callable that drives a variable-length sub-flow and mutates ``answers``.
    """

    key: str
    label: str
    kind: str
    default: Any = None
    choices: list[str] = field(default_factory=list)
    validate: Callable[[Any], "str | None"] | None = None
    when: Callable[[dict], bool] | None = None
    tier: str = "core"
    help: "str | None" = None
    handler: Callable | None = None
