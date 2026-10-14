from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart_rod_tracking import compute_heart_mask_rod, associate_large_hole_components  # noqa: E402


class HeartRodTrackingTests(unittest.TestCase):
    def fixture(self, *, heart_center_y=30, heart_center_x=55, heart_z=(8, 29)):
        z_size, y_size, x_size = 36, 96, 96
        hu = np.full((z_size, y_size, x_size), -1000.0, dtype=np.float32)
        lung = np.zeros(hu.shape, dtype=bool)
        yy, xx = np.ogrid[:y_size, :x_size]
        left_lung = ((yy - 48) ** 2 / 22**2 + (xx - 26) ** 2 / 12**2) <= 1
        right_lung = ((yy - 48) ** 2 / 22**2 + (xx - 70) ** 2 / 12**2) <= 1
        for z in range(2, 34):
            lung[z] = left_lung | right_lung
            hu[z, lung[z]] = -800.0
        heart = ((yy - heart_center_y) ** 2 / 7**2 + (xx - heart_center_x) ** 2 / 14**2) <= 1
        for z in range(heart_z[0], heart_z[1]):
            hu[z, heart] = 60.0
        return hu, lung, heart

    def tracker(self, hu, lung, *, spacing=(0.7, 0.7), z_spacing=2.5, max_radius=35.0, min_span=25.0):
        return compute_heart_mask_rod(
            hu,
            lung,
            pixel_spacing_mm=spacing,
            slice_spacing_mm=z_spacing,
            min_span_mm=min_span,
            seed_radius_mm=2.0,
            radius_step_mm=4.0,
            max_radius_mm=max_radius,
            soft_tissue_hu_min=-250.0,
            soft_tissue_hu_max=500.0,
            min_lung_slice_fraction=0.08,
            axial_margin_mm=5.0,
        )

    def test_expanding_rod_tracks_off_center_component_in_3d(self):
        hu, lung, expected_heart = self.fixture(heart_center_y=30, heart_center_x=55)
        actual, diagnostics = self.tracker(hu, lung)
        self.assertTrue(actual.any())
        self.assertEqual(diagnostics["status"], "tracked_central_3d_component")
        self.assertGreater(diagnostics["selected_radius_mm"], 2.0)
        self.assertGreaterEqual(diagnostics["selected_component_span_mm"], 25.0)
        self.assertGreater(np.count_nonzero(actual & np.broadcast_to(expected_heart, actual.shape)), 0)

    def test_component_must_persist_over_required_z_span(self):
        hu, lung, _ = self.fixture(heart_center_y=30, heart_center_x=55, heart_z=(8, 14))
        actual, diagnostics = self.tracker(hu, lung)
        self.assertFalse(actual.any())
        self.assertEqual(diagnostics["status"], "not_found")
        self.assertEqual(diagnostics["failure_reason"], "no_3d_component_met_rod_contact_and_z_span_criteria")

    def test_tracks_tissue_even_if_current_lung_mask_swallowed_it(self):
        hu, lung, heart2d = self.fixture(heart_center_y=30, heart_center_x=55)
        # Simulate the failure mode: parts of the soft-tissue heart are labelled lung.
        lung[12:23, heart2d] = True
        actual, diagnostics = self.tracker(hu, lung)
        self.assertTrue(actual.any())
        self.assertEqual(diagnostics["status"], "tracked_central_3d_component")

    def test_empty_tissue_returns_empty_fail_closed(self):
        hu = np.full((30, 64, 64), -1000.0, dtype=np.float32)
        lung = np.zeros_like(hu, dtype=bool)
        yy, xx = np.ogrid[:64, :64]
        lungs = (((yy - 32) ** 2 / 18**2 + (xx - 18) ** 2 / 8**2) <= 1) | (((yy - 32) ** 2 / 18**2 + (xx - 46) ** 2 / 8**2) <= 1)
        for z in range(3, 27):
            lung[z] = lungs
            hu[z, lung[z]] = -800
        mask, diag = self.tracker(hu, lung, max_radius=20)
        self.assertFalse(mask.any())
        self.assertEqual(diag["status"], "not_found")

    def test_geometry_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.tracker(np.zeros((5, 8, 8)), np.zeros((5, 8, 7), dtype=bool))

    def test_invalid_spacing_fails(self):
        hu, lung, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_heart_mask_rod(hu, lung, pixel_spacing_mm=(0.7, 0), slice_spacing_mm=2.5)

    def test_associates_whole_3d_hole_only_when_it_overlaps_track(self):
        holes = np.zeros((15, 20, 20), dtype=bool)
        holes[3:9, 5:12, 5:12] = True
        holes[2:10, 15:18, 15:18] = True
        track = np.zeros_like(holes)
        track[5, 8, 8] = True
        associated, diagnostics = associate_large_hole_components(holes, track)
        self.assertEqual(int(associated.sum()), int(holes[3:9, 5:12, 5:12].sum()))
        self.assertEqual(diagnostics["large_hole_component_count"], 2)
        self.assertEqual(len(diagnostics["associated_large_hole_component_labels"]), 1)

    def test_association_rejects_geometry_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same 3D ZYX shape"):
            associate_large_hole_components(np.zeros((2, 2, 2)), np.zeros((3, 2, 2)))


if __name__ == "__main__":
    unittest.main()
