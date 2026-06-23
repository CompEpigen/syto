import unittest

from App.wizard.engine import WizardEngine
from App.wizard.fields import FieldSpec


class StubEngine(WizardEngine):
    """Engine whose I/O is scripted so tests need no TTY."""

    def __init__(self, raw_by_key, expert=False):
        super().__init__()
        self._raw_by_key = raw_by_key
        self._expert = expert
        self.gate_asked = 0

    def _ask(self, spec):
        return self._raw_by_key[spec.key]

    def _ask_expert_gate(self):
        self.gate_asked += 1
        return self._expert


class TestWizardEngine(unittest.TestCase):
    def test_collects_flat_answers(self):
        specs = [
            FieldSpec(key="a", label="A", kind="text"),
            FieldSpec(key="n", label="N", kind="int"),
        ]
        eng = StubEngine({"a": "hello", "n": "39"})
        answers = eng.run(specs)
        self.assertEqual(answers, {"a": "hello", "n": 39})

    def test_when_false_skips_and_records_default(self):
        specs = [
            FieldSpec(key="type", label="T", kind="text"),
            FieldSpec(
                key="flavor", label="F", kind="text", default="NA",
                when=lambda ans: ans.get("type") == "dismir",
            ),
        ]
        eng = StubEngine({"type": "lookup", "flavor": "SHOULD_NOT_ASK"})
        answers = eng.run(specs)
        self.assertEqual(answers["flavor"], "NA")

    def test_expert_gate_asked_once_and_skips_when_declined(self):
        specs = [
            FieldSpec(key="core1", label="C", kind="text"),
            FieldSpec(key="exp1", label="E1", kind="int", default=2, tier="expert"),
            FieldSpec(key="exp2", label="E2", kind="int", default=10, tier="expert"),
        ]
        eng = StubEngine({"core1": "x", "exp1": "999", "exp2": "999"}, expert=False)
        answers = eng.run(specs)
        self.assertEqual(eng.gate_asked, 1)
        self.assertEqual(answers["exp1"], 2)
        self.assertEqual(answers["exp2"], 10)

    def test_expert_fields_asked_when_accepted(self):
        specs = [
            FieldSpec(key="exp1", label="E1", kind="int", default=2, tier="expert"),
        ]
        eng = StubEngine({"exp1": "7"}, expert=True)
        answers = eng.run(specs)
        self.assertEqual(answers["exp1"], 7)

    def test_validation_reprompts_until_valid(self):
        from App.wizard import validators as v

        attempts = iter(["0", "-3", "5"])

        class RepromptEngine(WizardEngine):
            def _ask(self, spec):
                return next(attempts)

        spec = FieldSpec(
            key="n", label="N", kind="int", validate=v.positive_int
        )
        eng = RepromptEngine()
        answers = eng.run([spec])
        self.assertEqual(answers["n"], 5)

    def test_bad_numeric_input_reprompts(self):
        attempts = iter(["not_a_number", "12"])

        class RepromptEngine(WizardEngine):
            def _ask(self, spec):
                return next(attempts)

        eng = RepromptEngine()
        answers = eng.run([FieldSpec(key="n", label="N", kind="int")])
        self.assertEqual(answers["n"], 12)

    def test_list_section_handler_invoked(self):
        def handler(engine, answers):
            answers["deconv.methods"] = ["xgboost"]

        spec = FieldSpec(
            key="deconv", label="Deconv", kind="list_section", handler=handler
        )
        eng = StubEngine({})
        answers = eng.run([spec])
        self.assertEqual(answers["deconv.methods"], ["xgboost"])


if __name__ == "__main__":
    unittest.main()
