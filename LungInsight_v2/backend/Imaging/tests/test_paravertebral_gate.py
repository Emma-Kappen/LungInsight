import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paravertebral_gate import EXCLUSION_REASON, apply_paravertebral_gate, locate_spine  # noqa: E402

SPACING = (2.5, 1.0, 1.0)


def _chest(with_spine: bool = True) -> tuple[np.ndarray, np.ndarray]:
    volume = np.full((40, 128, 128), -50.0, dtype=np.float32)
    lung = np.zeros(volume.shape, dtype=bool)
    lung[:, 30:100, 20:50] = True
    lung[:, 30:100, 78:108] = True
    volume[lung] = -800.0
    if with_spine:
        volume[:, 90:110, 58:70] = 400.0
    return volume, lung


def _candidate(name: str, start: list[float], end: list[float]) -> dict:
    return {"stable_nodule_id": name, "bbox_start_zyx": start, "bbox_end_zyx": end, "candidate_in_lung_mask": True}


class ParavertebralGateTest(unittest.TestCase):
    def test_rejects_only_candidates_touching_the_vertebral_column(self) -> None:
        volume, lung = _chest()
        near = _candidate("near", [10, 95, 72], [14, 99, 76])  # 2 mm from the column
        far = _candidate("far", [10, 40, 30], [14, 44, 34])
        kept, rejected, report = apply_paravertebral_gate([near, far], volume, lung, SPACING)
        self.assertEqual([row["stable_nodule_id"] for row in kept], ["far"])
        self.assertEqual([row["stable_nodule_id"] for row in rejected], ["near"])
        self.assertEqual(rejected[0]["exclusion_reason"], EXCLUSION_REASON)
        self.assertFalse(rejected[0]["candidate_in_lung_mask"])
        self.assertEqual(report["status"], "spine_located")
        self.assertEqual(report["rejected_ids"], ["near"])
        self.assertIsNone(kept[0]["distance_to_spine_mm"])  # farther than the search margin

    def test_candidate_just_outside_the_threshold_is_kept(self) -> None:
        volume, lung = _chest()
        row = _candidate("edge", [10, 95, 81], [14, 99, 85])  # 12 mm from the column
        kept, rejected, _ = apply_paravertebral_gate([row], volume, lung, SPACING)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])
        self.assertAlmostEqual(kept[0]["distance_to_spine_mm"], 12.0, places=1)

    def test_rejects_nothing_when_the_spine_is_not_in_the_scan(self) -> None:
        volume, lung = _chest(with_spine=False)
        spine, report = locate_spine(volume, lung, SPACING)
        self.assertIsNone(spine)
        self.assertEqual(report["status"], "spine_not_found")
        row = _candidate("any", [10, 95, 72], [14, 99, 76])
        kept, rejected, _ = apply_paravertebral_gate([row], volume, lung, SPACING)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
