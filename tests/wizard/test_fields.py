import unittest

from syto.app.wizard.fields import FieldSpec


class TestFieldSpec(unittest.TestCase):
    def test_minimal_construction_defaults(self):
        fs = FieldSpec(key="a.b", label="A B", kind="text")
        self.assertEqual(fs.key, "a.b")
        self.assertEqual(fs.tier, "core")
        self.assertEqual(fs.choices, [])
        self.assertIsNone(fs.validate)
        self.assertIsNone(fs.when)
        self.assertIsNone(fs.handler)

    def test_choices_are_independent_per_instance(self):
        a = FieldSpec(key="a", label="A", kind="select")
        b = FieldSpec(key="b", label="B", kind="select")
        a.choices.append("x")
        self.assertEqual(b.choices, [])

    def test_when_predicate_is_callable(self):
        fs = FieldSpec(
            key="x",
            label="X",
            kind="text",
            when=lambda answers: answers.get("y") == "z",
        )
        self.assertTrue(fs.when({"y": "z"}))
        self.assertFalse(fs.when({"y": "q"}))


if __name__ == "__main__":
    unittest.main()
