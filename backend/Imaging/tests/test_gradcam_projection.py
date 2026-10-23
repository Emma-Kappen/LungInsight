from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from gradcam_projection import project_patch_cam_to_ct


class GradcamProjectionTests(unittest.TestCase):
    def test_projects_patch_with_negative_padding_offset(self):
        patch = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        result = project_patch_cam_to_ct(patch, (-1, 1, 2), (3, 4, 5))
        self.assertTrue(np.isnan(result[0, 0, 0]))
        self.assertEqual(result[0, 2, 3], patch[1, 1, 1])
        self.assertEqual(result[1, 3, 4], patch[2, 2, 2])

    def test_nonintersecting_patch_returns_all_nan(self):
        result = project_patch_cam_to_ct(np.ones((2, 2, 2)), (4, 4, 4), (3, 3, 3))
        self.assertTrue(np.isnan(result).all())

    def test_rejects_nonfinite_cam_and_bad_geometry(self):
        with self.assertRaises(ValueError):
            project_patch_cam_to_ct(np.full((2, 2, 2), np.nan), (0, 0, 0), (2, 2, 2))
        with self.assertRaises(ValueError):
            project_patch_cam_to_ct(np.ones((2, 2, 2)), (0, 0), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
