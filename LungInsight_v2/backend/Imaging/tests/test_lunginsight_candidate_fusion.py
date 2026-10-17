from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from lunginsight_candidate_fusion import (  # noqa: E402
    filter_detections_by_lung_mask,
    load_canonical_contract,
    merge_detections,
    rank_filtered_log_candidates,
    select_fused_candidates,
)


SPACE = "stage01_final_cropped_ct"


def vit(cid, center, score=0.8, low=None, high=None):
    center = list(center)
    row = {
        "candidate_id": cid,
        "source": "ViTDet3D",
        "detection_source": "ViTDet3D",
        "detector_score": score,
        "coordinate_order": "ZYX",
        "center_zyx": center,
        "space": SPACE,
    }
    if low is not None:
        row["bbox_start_zyx"] = low
        row["bbox_end_zyx"] = high
    return row


def log(cid, center, response=2.0, scale=8.0):
    return {
        "diagnostic_id": cid,
        "purpose": "research_comparison_only",
        "coordinate_order": "ZYX",
        "center_zyx": list(center),
        "log_response": response,
        "scale_mm": scale,
        "space": SPACE,
    }


class MergeTests(unittest.TestCase):
    def merge(self, logs, vits, spacing=(2.5, 0.7, 0.7), **kwargs):
        return merge_detections(logs, vits, spacing_zyx_mm=spacing, expected_space=SPACE, **kwargs)

    def test_empty_detector_outputs(self):
        records, report = self.merge([], [])
        self.assertEqual(records, [])
        self.assertEqual(report["union_count_before_deduplication"], 0)
        self.assertEqual(report["deduplicated_count"], 0)

    def test_empty_log_keeps_vit(self):
        records, report = self.merge([], [vit("v1", (4, 10, 10))])
        self.assertEqual(len(records), 1)
        self.assertEqual(report["vitdet3d_count"], 1)
        self.assertEqual(records[0]["detector_provenance"], ["ViTDet3D"])

    def test_duplicate_log_and_vit_merge_and_keep_provenance_scores(self):
        records, report = self.merge(
            [log("l1", (4.0, 10.0, 10.0), response=3.4)],
            [vit("v1", (4.2, 10.0, 10.0), score=0.91, low=[2, 8, 8], high=[7, 13, 13])],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(report["union_count_before_deduplication"], 2)
        self.assertEqual(report["deduplicated_count"], 1)
        self.assertEqual(records[0]["primary_geometry_source"], "equal_weight_fusion")
        self.assertEqual(records[0]["center_zyx"], [4.1, 10.0, 10.0])
        self.assertEqual(set(records[0]["detector_provenance"]), {"LoG", "ViTDet3D"})
        self.assertAlmostEqual(records[0]["detector_scores"]["ViTDet3D"][0]["value"], 0.91)
        self.assertAlmostEqual(records[0]["detector_scores"]["LoG"][0]["value"], 3.4)
        evidence = {record["detector"]: record["record"] for record in records[0]["provenance_records"]}
        self.assertEqual(evidence["ViTDet3D"]["candidate_id"], "v1")
        self.assertEqual(evidence["LoG"]["diagnostic_id"], "l1")
        self.assertAlmostEqual(evidence["LoG"]["log_response"], 3.4)

    def test_single_detector_candidates_survive_fusion(self):
        records, _ = self.merge(
            [log("log-alone", (3, 5, 7))],
            [vit("vit-alone", (15, 5, 7))],
        )
        self.assertEqual(len(records), 2)
        self.assertEqual({row["source"] for row in records}, {"LoG", "ViTDet3D"})

    def test_geometry_fusion_weights_detectors_not_duplicate_record_count(self):
        records, _ = self.merge(
            [
                log("log-1", (10, 10, 10), scale=2),
                log("log-2", (10, 10, 11), scale=2),
            ],
            [vit("vit-1", (10, 10, 12), low=[9, 9, 11], high=[11, 11, 13])],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["center_zyx"], [10.0, 10.0, 11.25])

    def test_distinct_nearby_nodules_not_merged_by_small_bbox_overlap_alone(self):
        records, report = self.merge(
            [],
            [
                vit("v1", (10, 20, 20), low=[8, 18, 18], high=[12, 22, 22]),
                vit("v2", (10, 26, 20), low=[8, 24, 18], high=[12, 28, 22]),
            ],
            center_distance_mm=2.0, max_center_distance_mm=3.0, min_overlap_iou=0.05,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(report["deduplicated_count"], 2)

    def test_anisotropic_spacing_used_for_physical_distance(self):
        # 2 z voxels at 3 mm = 6 mm, so this is not a 2 mm near-center match.
        records, _ = self.merge(
            [log("l1", (10, 10, 10), scale=2)],
            [vit("v1", (12, 10, 10), low=[11, 9, 9], high=[13, 11, 11])],
            spacing=(3.0, 0.5, 0.5), center_distance_mm=2.0,
            max_center_distance_mm=4.0, min_overlap_iou=0.2,
        )
        self.assertEqual(len(records), 2)

    def test_coordinate_frame_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "convert geometry"):
            self.merge([], [dict(vit("v1", (1, 2, 3)), space="patch_local_ct")])

    def test_invalid_spacing_rejected(self):
        with self.assertRaises(ValueError):
            self.merge([], [], spacing=(1, 0, 1))

    def test_merge_stable_ids_and_order(self):
        logs = [log("l2", (20, 20, 20)), log("l1", (5, 5, 5))]
        vits = [vit("v2", (20.2, 20, 20)), vit("v1", (5.2, 5, 5))]
        first, _ = self.merge(logs, vits)
        second, _ = self.merge(list(reversed(logs)), list(reversed(vits)))
        self.assertEqual([x["nodule_id"] for x in first], [x["nodule_id"] for x in second])
        self.assertEqual([x["center_zyx"] for x in first], [x["center_zyx"] for x in second])

    def test_physical_stable_id_and_source_members_survive_deduplication(self):
        kwargs = {
            "patient_id": "p1",
            "study_uid": "study1",
            "series_uid": "series1",
            "origin_lps_mm": [-100.0, -200.0, -300.0],
            "direction_lps": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        }
        first, _ = self.merge(
            [{**log("log-1", (10, 20, 30), response=2.5), "center_lps_mm": [-86.0, -179.0, -275.0]}],
            [{**vit("vit-1", (10.2, 20, 30), score=0.9), "center_lps_mm": [-86.0, -179.0, -274.5]}],
            **kwargs,
        )
        second, _ = self.merge(
            [{**log("log-1", (10, 20, 30), response=2.5), "center_lps_mm": [-86.0, -179.0, -275.0]}],
            [{**vit("vit-1", (10.2, 20, 30), score=0.9), "center_lps_mm": [-86.0, -179.0, -274.5]}],
            **kwargs,
        )
        self.assertEqual(first[0]["stable_nodule_id"], second[0]["stable_nodule_id"])
        self.assertEqual(first[0]["candidate_id"], first[0]["stable_nodule_id"])
        self.assertEqual(first[0]["source"], "Merged")
        self.assertEqual({item["candidate_id"] for item in first[0]["source_members"]}, {"vit-1", "log-1"})
        self.assertEqual(first[0]["vitdet_score"], 0.9)
        self.assertEqual(first[0]["log_score"], 2.5)

    def test_stable_identity_does_not_change_with_detector_agreement(self):
        kwargs = {
            "patient_id": "p1",
            "study_uid": "study1",
            "series_uid": "series1",
            "origin_lps_mm": [-100.0, -200.0, -300.0],
            "direction_lps": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        }
        physical = {"center_lps_mm": [-86.0, -179.0, -275.0]}
        log_only, _ = self.merge(
            [{**log("l1", (10, 20, 30)), **physical}], [], **kwargs
        )
        vit_only, _ = self.merge(
            [], [{**vit("v1", (10, 20, 30)), **physical}], **kwargs
        )
        self.assertEqual(log_only[0]["stable_nodule_id"], vit_only[0]["stable_nodule_id"])

    def test_partial_stable_identity_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires patient_id"):
            self.merge(
                [],
                [vit("v1", (4, 10, 10))],
                patient_id="p1",
            )

    def test_filtered_log_candidates_are_ranked_before_limit(self):
        rows = [log(f"l{score}", (score, 10, 10), response=score) for score in range(60)]
        selected = rank_filtered_log_candidates(rows, limit=50)
        self.assertEqual(len(selected), 50)
        self.assertEqual(selected[0]["log_response"], 59)
        self.assertEqual(selected[-1]["log_response"], 10)
        self.assertEqual(rank_filtered_log_candidates(list(reversed(rows)), limit=50), selected)

    def test_log_ranking_rejects_missing_score(self):
        with self.assertRaisesRegex(ValueError, "no finite log_response"):
            rank_filtered_log_candidates([{"diagnostic_id": "missing"}])

    def test_selection_is_detector_neutral_and_keeps_single_source_candidates(self):
        fused, _ = self.merge(
            [log("log-only", (30, 30, 30), response=100.0)],
            [
                vit("vit-only-1", (4, 4, 4), score=0.2),
                vit("vit-only-2", (8, 8, 8), score=0.1),
            ],
        )
        selected = select_fused_candidates(fused, limit=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source"] for row in selected}, {"LoG", "ViTDet3D"})
        self.assertIn(
            "log-only",
            {member["candidate_id"] for row in selected for member in row["source_members"]},
        )
        self.assertTrue(all(0.0 <= row["fusion_rank_percentile"] <= 1.0 for row in fused))


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.lung = np.zeros((12, 16, 16), dtype=bool)
        self.lung[2:10, 2:14, 2:14] = True
        self.heart = np.zeros_like(self.lung)
        self.heart[4:8, 6:11, 6:11] = True

    def test_empty_input(self):
        accepted, rejected, report = filter_detections_by_lung_mask([], self.lung, self.heart, expected_space=SPACE)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual(report["input_count"], 0)

    def test_center_inside_heart_rejected_even_if_lung_mask_overlaps(self):
        self.lung[5, 8, 8] = True  # simulate erroneous overlap in the upstream lung mask
        candidate = vit("heart_fp", (5, 8, 8))
        accepted, rejected, report = filter_detections_by_lung_mask([candidate], self.lung, self.heart, expected_space=SPACE)
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["exclusion_reason"], "center_inside_heart_mask")
        self.assertEqual(report["rejection_counts"]["center_inside_heart_mask"], 1)

    def test_outside_lung_rejected(self):
        accepted, rejected, _ = filter_detections_by_lung_mask([vit("outside", (1, 1, 1))], self.lung, self.heart, expected_space=SPACE)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["exclusion_reason"], "center_outside_lung_mask")

    def test_center_inside_non_heart_lung_accepted(self):
        accepted, rejected, _ = filter_detections_by_lung_mask([vit("good", (3, 3, 3))], self.lung, self.heart, expected_space=SPACE)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_empty_heart_mask_required_fails_closed_for_all_candidates(self):
        empty_heart = np.zeros_like(self.lung)
        candidates = [vit("inside", (3, 3, 3)), vit("outside", (1, 1, 1))]
        accepted, rejected, report = filter_detections_by_lung_mask(
            candidates, self.lung, empty_heart, expected_space=SPACE, require_nonempty_heart_mask=True
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(row["exclusion_reason"] == "heart_mask_empty_unverified" for row in rejected))
        self.assertEqual(report["gate_status"], "blocked_empty_heart_mask")
        self.assertFalse(report["heart_mask_found"])

    def test_nonempty_heart_mask_passes_without_visual_review(self):
        accepted, rejected, report = filter_detections_by_lung_mask(
            [vit("good", (3, 3, 3))], self.lung, self.heart,
            expected_space=SPACE,
            require_nonempty_heart_mask=True,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(report["gate_status"], "passed")
        self.assertTrue(report["heart_mask_found"])

    def test_empty_heart_mask_can_be_reported_for_nonproduction_comparison_only(self):
        empty_heart = np.zeros_like(self.lung)
        accepted, rejected, report = filter_detections_by_lung_mask(
            [vit("inside", (3, 3, 3))], self.lung, empty_heart, expected_space=SPACE
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertFalse(report["heart_mask_found"])
        # Production callers opt into the fail-closed nonempty heart-mask requirement.

    def test_invalid_center_rejected(self):
        accepted, rejected, _ = filter_detections_by_lung_mask([vit("bad", (float("nan"), 2, 3))], self.lung, self.heart, expected_space=SPACE)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["exclusion_reason"], "missing_or_invalid_center_zyx")

    def test_mask_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            filter_detections_by_lung_mask([], self.lung, np.zeros((2, 2, 2), dtype=bool), expected_space=SPACE)

    def test_occupancy_threshold_is_additional_and_configurable(self):
        candidate = vit("box", (3, 3, 3), low=[2, 2, 2], high=[10, 12, 12])
        accepted, rejected, report = filter_detections_by_lung_mask(
            [candidate], self.lung, self.heart, min_bbox_lung_occupancy=0.95, expected_space=SPACE
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["exclusion_reason"], "bbox_lung_mask_occupancy_below_threshold")
        self.assertLess(report["rejected_count"], 2)

    def test_physical_center_must_match_zyx_geometry(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            filter_detections_by_lung_mask(
                [{**vit("bad-transform", (3, 3, 3)), "center_lps_mm": [99, 99, 99]}],
                self.lung,
                self.heart,
                spacing_zyx_mm=(2, 1, 1),
                origin_lps_mm=(0, 0, 0),
                direction_lps=np.eye(3),
                expected_space=SPACE,
            )

    def test_missing_candidate_coordinate_frame_fails_closed(self):
        candidate = vit("missing-space", (3, 3, 3))
        del candidate["space"]
        with self.assertRaisesRegex(ValueError, "coordinate space is missing"):
            filter_detections_by_lung_mask(
                [candidate], self.lung, self.heart, expected_space=SPACE
            )


class SchemaContractTests(unittest.TestCase):
    def schema(self, version="2.6"):
        return {
            "schema_version": version,
            "array_axes": ["z", "y", "x"],
            "coordinate_system": "DICOM_LPS",
            "coordinate_format": {
                "array_axes": ["z", "y", "x"],
                "coordinate_system": "DICOM_LPS",
                "ct_frame_space": SPACE,
                "index_to_physical_formula": "origin_mm + direction_lps @ (zyx * spacing_zyx_mm)",
            },
            "stage_02_detect_and_patch": {"space": SPACE},
        }

    def test_accepts_schema_26_when_coordinate_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(self.schema("2.6")), encoding="utf-8")
            contract = load_canonical_contract(path)
        self.assertEqual(contract["schema_version"], "2.6")
        self.assertEqual(contract["ct_space"], SPACE)
        self.assertEqual(contract["coordinate_system"], "DICOM_LPS")

    def test_accepts_schema_27_when_coordinate_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(self.schema("2.7")), encoding="utf-8")
            contract = load_canonical_contract(path)
        self.assertEqual(contract["schema_version"], "2.7")
        self.assertEqual(contract["ct_space"], SPACE)
        self.assertEqual(contract["coordinate_system"], "DICOM_LPS")

    def test_accepts_schema_28_when_coordinate_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(self.schema("2.8")), encoding="utf-8")
            contract = load_canonical_contract(path)
        self.assertEqual(contract["schema_version"], "2.8")
        self.assertEqual(contract["ct_space"], SPACE)
        self.assertEqual(contract["coordinate_system"], "DICOM_LPS")

    def test_accepts_schema_210_when_coordinate_contract_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(self.schema("2.10")), encoding="utf-8")
            contract = load_canonical_contract(path)
        self.assertEqual(contract["schema_version"], "2.10")
        self.assertEqual(contract["ct_space"], SPACE)
        self.assertEqual(contract["coordinate_system"], "DICOM_LPS")

    def test_accepts_schema_23_contract_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(self.schema("2.3")), encoding="utf-8")
            contract = load_canonical_contract(path)
        self.assertEqual(contract["schema_version"], "2.3")

    def test_rejects_conflicting_ct_frame_definitions(self):
        schema = self.schema("2.6")
        schema["coordinate_format"]["ct_frame_space"] = "patch_local_ct"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CT-frame conflict"):
                load_canonical_contract(path)

    def test_rejects_unsupported_coordinate_axes(self):
        schema = self.schema("2.6")
        schema["array_axes"] = ["x", "y", "z"]
        schema["coordinate_format"]["array_axes"] = ["x", "y", "z"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta_schema.json"
            path.write_text(json.dumps(schema), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported canonical array axes"):
                load_canonical_contract(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
