"""Drives a list of FieldSpecs as an interactive wizard.

I/O is isolated in ``_ask`` / ``_ask_expert_gate`` / ``ask_checkbox`` so the
control flow (conditional fields, expert gating, validation re-prompting) can be
unit-tested by subclassing and scripting those methods.
"""

from typing import Any

from App.wizard.fields import FieldSpec


class WizardEngine:
    """Iterate FieldSpecs, prompt the user, and collect a flat answers dict."""

    def run(self, specs: list[FieldSpec]) -> dict:
        answers: dict = {}
        expert_enabled: bool | None = None

        for spec in specs:
            if spec.kind == "list_section":
                spec.handler(self, answers)
                continue

            if spec.when is not None and not spec.when(answers):
                answers[spec.key] = spec.default
                continue

            if spec.tier == "expert":
                if expert_enabled is None:
                    expert_enabled = self._ask_expert_gate()
                if not expert_enabled:
                    answers[spec.key] = spec.default
                    continue

            answers[spec.key] = self.ask_one(spec, answers)

        return answers

    # ── Public prompt helpers (reused by list-section handlers) ──────────────

    def ask_one(self, spec: FieldSpec, answers: dict) -> Any:
        """Prompt a single field, coercing and validating with re-prompts."""
        while True:
            raw = self._ask(spec)
            try:
                value = self._coerce(spec, raw)
            except ValueError as exc:
                self._echo_error(str(exc))
                continue
            if spec.validate is not None:
                error = spec.validate(value)
                if error:
                    self._echo_error(error)
                    continue
            return value

    def ask_checkbox(self, label: str, choices: list[str]) -> list[str]:
        import questionary

        return questionary.checkbox(label, choices=choices).ask() or []

    # ── Coercion ─────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(spec: FieldSpec, raw: Any) -> Any:
        if spec.kind == "int":
            return int(str(raw).strip())
        if spec.kind == "float":
            return float(str(raw).strip())
        if spec.kind == "bool":
            return bool(raw)
        # text, select, path -> raw string as-is
        return raw

    # ── I/O (overridden in tests) ────────────────────────────────────────────

    def _ask(self, spec: FieldSpec) -> Any:
        import questionary

        if spec.kind == "select":
            return questionary.select(
                spec.label, choices=spec.choices, default=spec.default
            ).ask()
        if spec.kind == "bool":
            return questionary.confirm(
                spec.label, default=bool(spec.default)
            ).ask()
        if spec.kind == "path":
            return questionary.path(
                spec.label, default=str(spec.default or "")
            ).ask()
        # text, int, float -> free text entry (coerced afterwards)
        return questionary.text(
            spec.label, default="" if spec.default is None else str(spec.default)
        ).ask()

    def _ask_expert_gate(self) -> bool:
        import questionary

        return questionary.confirm(
            "Configure advanced settings? These are sensible defaults — you can "
            "always edit the generated YAML directly after saving.",
            default=False,
        ).ask()

    @staticmethod
    def _echo_error(message: str) -> None:
        print(f"  ✗ {message}")
