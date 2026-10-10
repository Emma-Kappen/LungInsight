from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import joblib
import numpy as np

from evaluate_xml_malignancy import evaluate
from predict_xml_malignancy import predict_record
from train_xml_malignancy import _load_nodule_rows
from xml_malignancy_common import (
    CumulativeOrdinalModel,
    FEATURE_COLUMNS,
    RATING_FEATURES,
    RATING_LIMITS,
    _rating_signature,
    match_reader_records,
    parse_integer_rating,
    parse_xml_file,
    validate_canonical_split,
)


def _labels_for_signature(signature: dict[str, int], target: int) -> np.ndarray:
    names = [
        "calcification", "internalStructure", "lobulation", "malignancy", "margin",
        "sphericity", "spiculation", "subtlety", "texture",
    ]
    values = {
        **signature,
        "malignancy": target,
    }
    return np.asarray([values[name] for name in names], dtype=np.float32)


def _write_h5(path: Path, signatures: list[dict[str, int]], patients: list[str] | None = None) -> None:
    patients = patients or [f"P{index}" for index in range(len(signatures))]
    with h5py.File(path, "w") as h5:
        h5.attrs["reader_feature_names"] = json.dumps([
            "calcification", "internalStructure", "lobulation", "malignancy", "margin",
            "sphericity", "spiculation", "subtlety", "texture",
        ])
        h5.create_dataset("reader_mask", data=np.ones((len(signatures), 1), dtype=bool))
        h5.create_dataset(
            "reader_labels",
            data=np.stack([_labels_for_signature(sig, 3) for sig in signatures])[:, None, :],
        )
        h5.create_dataset("patient_id", data=np.asarray(patients, dtype="S16"))
        h5.create_dataset("study_instance_uid", data=np.asarray(["study"] * len(signatures), dtype="S16"))
        h5.create_dataset("series_instance_uid", data=np.asarray(["series"] * len(signatures), dtype="S16"))
        h5.create_dataset("nodule_id", data=np.asarray([f"Nodule {i + 1:04d}" for i in range(len(signatures))], dtype="S32"))


def _signature(seed: int) -> dict[str, int]:
    values = {}
    for index, name in enumerate(RATING_FEATURES):
        values[name] = 1 + ((seed + index) % RATING_LIMITS[name])
    return values


def _xml_record(signature: dict[str, int], target: int = 3) -> dict:
    return {
        "study_instance_uid": "study",
        "series_instance_uid": "series",
        "feature_values": dict(signature),
        "malignancy_rating": target,
        "match_status": "unmatched",
        "match_reason": "not_evaluated",
    }


class XmlParsingTests(unittest.TestCase):
    def test_xml_reader_records_and_contour_geometry_are_extracted(self) -> None:
        content = """<LidcReadMessage>
          <ResponseHeader><StudyInstanceUID>study</StudyInstanceUID><SeriesInstanceUid>series</SeriesInstanceUid></ResponseHeader>
          <readingSession><servicingRadiologistID>reader-a</servicingRadiologistID>
            <unblindedReadNodule><noduleID>Nodule 001</noduleID><characteristics>
              <subtlety>5</subtlety><internalStructure>2</internalStructure><calcification>6</calcification>
              <sphericity>3</sphericity><margin>4</margin><lobulation>1</lobulation><spiculation>2</spiculation>
              <texture>5</texture><malignancy>4</malignancy>
            </characteristics><roi><edgeMap><xCoord>1</xCoord><yCoord>2</yCoord></edgeMap>
              <edgeMap><xCoord>8</xCoord><yCoord>12</yCoord></edgeMap></roi></unblindedReadNodule>
          </readingSession></LidcReadMessage>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid.xml"
            path.write_text(content, encoding="utf-8")
            rows, status = parse_xml_file(path)
        self.assertEqual(status["status"], "parsed")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["feature_values"]["calcification"], 6)
        self.assertEqual(rows[0]["malignancy_rating"], 4)
        self.assertEqual(rows[0]["geometry"]["roi_count"], 1)
        self.assertEqual(rows[0]["geometry"]["contour_points"], 2)
        self.assertEqual(rows[0]["geometry"]["contour_bbox_width"], 7.0)
        self.assertEqual(rows[0]["reader_id"], "reader-a")
        self.assertEqual(status["tag_counts"]["diameter"], 0)
        self.assertEqual(status["tag_counts"]["roi"], 1)

    def test_malformed_xml_is_reported_without_partial_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.xml"
            path.write_text("<LidcReadMessage>", encoding="utf-8")
            rows, status = parse_xml_file(path)
        self.assertEqual(rows, [])
        self.assertEqual(status["status"], "malformed_xml")
        self.assertTrue(status["error"])

    def test_missing_zero_and_out_of_range_ratings_are_not_valid_categories(self) -> None:
        self.assertIsNone(parse_integer_rating(None, "subtlety", maximum=5))
        self.assertIsNone(parse_integer_rating("0", "subtlety", maximum=5))
        self.assertIsNone(parse_integer_rating("6", "subtlety", maximum=5))
        self.assertIsNone(parse_integer_rating("2.5", "subtlety", maximum=5))
        self.assertEqual(parse_integer_rating("6", "calcification", maximum=6), 6)

    def test_malignancy_is_not_part_of_predictor_signature_or_feature_order(self) -> None:
        features = _signature(1)
        first = _rating_signature(features)
        features["malignancy"] = 5
        self.assertEqual(first, _rating_signature(features))
        self.assertFalse(any("malignancy" in feature.lower() for feature in FEATURE_COLUMNS))


class MatchingAndSplitTests(unittest.TestCase):
    def test_distinct_nodules_match_by_unique_full_signature_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            h5_path = Path(temporary) / "source.h5"
            signatures = [_signature(1), _signature(8)]
            _write_h5(h5_path, signatures)
            rows = [_xml_record(signature) for signature in signatures]
            report = match_reader_records(rows, h5_path)
        self.assertEqual(report["matched_record_count"], 2)
        self.assertEqual([row["h5_sample_index"] for row in rows], [0, 1])
        self.assertEqual([row["match_status"] for row in rows], ["matched", "matched"])
        self.assertTrue(all(row["target_crosscheck"] for row in rows))

    def test_duplicate_xml_signatures_are_rejected_as_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            h5_path = Path(temporary) / "source.h5"
            signature = _signature(2)
            _write_h5(h5_path, [signature])
            rows = [_xml_record(signature), _xml_record(signature)]
            report = match_reader_records(rows, h5_path)
        self.assertEqual(report["matched_record_count"], 0)
        self.assertEqual(report["unmatched_reason_counts"], {"ambiguous_signature": 2})

    def test_incomplete_feature_signatures_are_dropped_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            h5_path = Path(temporary) / "source.h5"
            signature = _signature(3)
            _write_h5(h5_path, [signature])
            incomplete = dict(signature)
            incomplete["texture"] = None
            rows = [_xml_record(incomplete)]
            report = match_reader_records(rows, h5_path)
        self.assertEqual(report["matched_record_count"], 0)
        self.assertEqual(rows[0]["match_reason"], "incomplete_non_malignancy_signature")

    def test_canonical_split_is_patient_disjoint_and_exactly_partitions_h5(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            h5_path = Path(temporary) / "source.h5"
            _write_h5(h5_path, [_signature(i) for i in range(3)], ["A", "B", "C"])
            split_path = Path(temporary) / "split.json"
            split_path.write_text(json.dumps({
                "train": {"sample_indices": [0], "patients": ["A"]},
                "val": {"sample_indices": [1], "patients": ["B"]},
                "test": {"sample_indices": [2], "patients": ["C"]},
            }), encoding="utf-8")
            split, index_split, metadata = validate_canonical_split(h5_path, split_path)
        self.assertEqual(metadata["sample_count"], 3)
        self.assertTrue(metadata["patient_disjoint"])
        self.assertEqual(index_split, {0: "train", 1: "val", 2: "test"})
        self.assertEqual(split["test"], [2])

    def test_patient_overlap_in_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            h5_path = Path(temporary) / "source.h5"
            _write_h5(h5_path, [_signature(i) for i in range(3)], ["A", "A", "C"])
            split_path = Path(temporary) / "split.json"
            split_path.write_text(json.dumps({
                "train": {"sample_indices": [0], "patients": ["A"]},
                "val": {"sample_indices": [1], "patients": ["A"]},
                "test": {"sample_indices": [2], "patients": ["C"]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Patient leakage"):
                validate_canonical_split(h5_path, split_path)


class InferenceAndArtifactTests(unittest.TestCase):
    @staticmethod
    def _fitted_bundle() -> dict:
        ratings = np.tile(np.arange(1, 6), 5)
        features = np.zeros((len(ratings), len(FEATURE_COLUMNS)), dtype=float)
        features[:, 0] = ratings.astype(float)
        features[:, 1] = np.nan
        model = CumulativeOrdinalModel(seed=7).fit(features, ratings)
        return {
            "model": model,
            "model_name": "ordinal",
            "feature_order": list(FEATURE_COLUMNS),
        }

    @staticmethod
    def _schema() -> dict:
        return {
            "features": {
                name: {"observed_range": [0, 5], "rating_range": [0, 5]}
                for name in FEATURE_COLUMNS
            }
        }

    def test_missing_and_out_of_range_features_warn_and_inference_is_deterministic(self) -> None:
        bundle = self._fitted_bundle()
        record = {FEATURE_COLUMNS[0]: 99, "malignancy_mean": 5}
        first = predict_record(record, bundle, self._schema())
        second = predict_record(record, bundle, self._schema())
        self.assertEqual(first, second)
        self.assertEqual(sum(first["probabilities_by_rating"].values()), 1.0)
        self.assertTrue(any("Missing" in warning for warning in first["warnings"]))
        self.assertTrue(any("outside" in warning for warning in first["warnings"]))
        self.assertIn("not a cancer diagnosis", first["interpretation"])

    def test_unseen_non_numeric_feature_is_treated_as_missing(self) -> None:
        bundle = self._fitted_bundle()
        result = predict_record({FEATURE_COLUMNS[0]: "new-category"}, bundle, self._schema())
        self.assertTrue(any("not numeric" in warning for warning in result["warnings"]))

    def test_model_save_reload_predictions_are_equivalent(self) -> None:
        bundle = self._fitted_bundle()
        features = np.zeros((2, len(FEATURE_COLUMNS)), dtype=float)
        features[:, 0] = [1.5, 4.5]
        features[:, 1] = np.nan
        expected = bundle["model"].predict_proba(features)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pkl"
            joblib.dump(bundle, path)
            actual = joblib.load(path)["model"].predict_proba(features)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_test_evaluation_is_blocked_until_validation_report_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            (data_dir / "model_selection.json").write_text("{}", encoding="utf-8")
            (data_dir / "best_model.pkl").write_bytes(b"placeholder")
            with self.assertRaisesRegex(RuntimeError, "blocked until validation_report"):
                evaluate(data_dir, "test")
            self.assertFalse((data_dir / "test_report.json").exists())

    def test_training_loader_excludes_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.csv"
            columns = ["split", "target_ratings_json", *FEATURE_COLUMNS]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for split_name in ("train", "val", "test"):
                    writer.writerow({
                        "split": split_name,
                        "target_ratings_json": "[1,2,3,4,5]",
                        **{name: 1 for name in FEATURE_COLUMNS},
                    })
            rows = _load_nodule_rows(path, {"train", "val"})
        self.assertEqual({row["split"] for row in rows}, {"train", "val"})


if __name__ == "__main__":
    unittest.main()