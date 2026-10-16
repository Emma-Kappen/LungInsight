from __future__ import annotations

import sys
import unittest
from pathlib import Path

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from nodule_identity import stable_nodule_id


class StableNoduleIdTests(unittest.TestCase):
    def test_is_deterministic_and_ignores_float_serialization_noise(self):
        first = stable_nodule_id("p1", "study1", "series1", [-12.3456781, 3, 5], "Merged")
        second = stable_nodule_id("p1", "study1", "series1", [-12.3456784, 3, 5], "Merged")
        self.assertEqual(first, second)

    def test_identity_changes_with_each_required_identity_component(self):
        base = ("p1", "study1", "series1", [1, 2, 3], "ViTDet3D")
        expected = stable_nodule_id(*base)
        for index, replacement in (
            (0, "p2"),
            (1, "study2"),
            (2, "series2"),
            (3, [1, 2, 3.01]),
            (4, "Merged"),
        ):
            values = list(base)
            values[index] = replacement
            self.assertNotEqual(expected, stable_nodule_id(*values))

    def test_missing_or_invalid_inputs_are_rejected(self):
        for values in (
            ("", "study", "series", [0, 0, 0], "LoG"),
            ("p", "", "series", [0, 0, 0], "LoG"),
            ("p", "study", "series", [0, 0], "LoG"),
            ("p", "study", "series", [0, float("nan"), 0], "LoG"),
            ("p", "UNKNOWN", "series", [0, 0, 0], "LoG"),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stable_nodule_id(*values)

    def test_output_is_namespaced_and_fixed_width(self):
        value = stable_nodule_id("p", "s", "r", [0, 0, 0], "LoG")
        self.assertRegex(value, r"^nodule_[0-9a-f]{24}$")


if __name__ == "__main__":
    unittest.main()
