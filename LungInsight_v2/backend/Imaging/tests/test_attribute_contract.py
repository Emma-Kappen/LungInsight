from __future__ import annotations

import sys
import unittest
from pathlib import Path

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from attribute_contract import build_attribute_support_audit


class AttributeContractTests(unittest.TestCase):
    def test_checkpoint_group_count_does_not_define_head_count(self):
        targets = (
            "subtlety", "internal_structure", "calcification", "sphericity",
            "margin", "lobulation", "spiculation", "texture",
        )
        groups = {
            "subtlety": ["subtlety"],
            "sphericity": ["sphericity"],
            "margin": ["margin"],
            "lobulation_spiculation": ["lobulation", "spiculation"],
            "texture": ["texture"],
        }
        supported = [name for attributes in groups.values() for name in attributes]
        scales = {name: {"semantics": "expected reader rating"} for name in supported}
        audit = build_attribute_support_audit(
            model_targets=targets,
            supported_attributes=supported,
            checkpoint_groups=groups,
            attribute_scales=scales,
        )
        self.assertEqual(len(audit), 8)
        self.assertEqual(audit["subtlety"]["target_index"], 0)
        self.assertEqual(audit["texture"]["target_index"], 7)
        self.assertEqual(audit["internal_structure"]["status"], "unsupported_checkpoint_contract")
        self.assertIsNone(audit["calcification"]["checkpoint_group"])

    def test_supported_attribute_requires_checkpoint_and_schema_semantics(self):
        with self.assertRaisesRegex(ValueError, "no checkpoint group"):
            build_attribute_support_audit(
                model_targets=("subtlety",),
                supported_attributes=("subtlety",),
                checkpoint_groups={},
                attribute_scales={"subtlety": {"semantics": "rating"}},
            )

    def test_duplicate_model_targets_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_attribute_support_audit(
                model_targets=("subtlety", "subtlety"),
                supported_attributes=(),
                checkpoint_groups={},
                attribute_scales={},
            )


if __name__ == "__main__":
    unittest.main()
