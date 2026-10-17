from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

IMAGING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMAGING_DIR))

from fastapi.testclient import TestClient
from api import app as api_module
from api import dicom_upload


def make_dicom(series_uid: str, position: float, pixel_value: int = 10) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = "patient-upload"
    dataset.StudyInstanceUID = "1.2.3.4"
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.InstanceNumber = int(position) + 1
    dataset.ImagePositionPatient = [0, 0, position]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.PixelSpacing = [0.7, 0.7]
    dataset.SliceThickness = 1.0
    dataset.RescaleSlope = 2.0
    dataset.RescaleIntercept = -1000.0
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelData = (np.full((2, 2), pixel_value, dtype=np.int16)).tobytes()
    output = __import__("io").BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


class DicomUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.original_root = api_module.OUTPUT_ROOT
        api_module.OUTPUT_ROOT = Path(self.temp.name).resolve()
        self.client = TestClient(api_module.app)

    def tearDown(self):
        api_module.OUTPUT_ROOT = self.original_root
        self.temp.cleanup()

    def test_reconstructs_one_series_in_hu_and_blocks_unrun_pipeline(self):
        files = [
            ("files", ("scan/one.dcm", make_dicom("1.2.3.5", 0), "application/dicom")),
            ("files", ("scan/two.dcm", make_dicom("1.2.3.5", 1), "application/dicom")),
        ]
        response = self.client.post(
            "/api/v1/imaging/dicom/upload",
            data={"patient_id": "patient-upload"},
            files=files,
        )
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(body["status"], "preprocessing_complete")
        self.assertEqual(body["analysis_status"], "blocked")
        self.assertIn("automated Stage 01 mask/geometry validation", body["next_step"])
        self.assertIn("Visual mask review is optional", body["next_step"])
        self.assertEqual(body["series"]["spacing_zyx_mm"], [1.0, 0.7, 0.7])
        volume_path = (
            api_module.OUTPUT_ROOT
            / "patient-upload"
            / "dicom_uploads"
            / body["job_id"]
            / "volume_hu.npy"
        )
        np.testing.assert_array_equal(np.load(volume_path), np.full((2, 2, 2), -980.0))
        stage01_input = (
            api_module.OUTPUT_ROOT
            / "patient-upload"
            / "dicom_uploads"
            / body["job_id"]
            / "stage01_input"
            / "patient-upload"
        )
        self.assertEqual(len(list(stage01_input.rglob("*.dcm"))), 2)
        self.assertIn("-InputPatientPath", body["runner_command"])
        self.assertIn("stage01_input", body["pipeline_input"])
        status = self.client.get(f"/api/v1/imaging/dicom/jobs/{body['job_id']}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["job_id"], body["job_id"])
        dicom_upload._JOBS.pop(body["job_id"], None)
        restored_status = self.client.get(f"/api/v1/imaging/dicom/jobs/{body['job_id']}")
        self.assertEqual(restored_status.status_code, 200)
        self.assertEqual(restored_status.json()["status"], "preprocessing_complete")

    def test_rejects_multiple_series_and_unsafe_relative_path(self):
        files = [
            ("files", ("one.dcm", make_dicom("1.2.3.5", 0), "application/dicom")),
            ("files", ("two.dcm", make_dicom("1.2.3.6", 1), "application/dicom")),
        ]
        response = self.client.post(
            "/api/v1/imaging/dicom/upload",
            data={"patient_id": "patient-upload"},
            files=files,
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("multiple CT studies", response.json()["detail"])

        response = self.client.post(
            "/api/v1/imaging/dicom/upload",
            data={"patient_id": "patient-upload"},
            files=[("files", ("../unsafe.dcm", make_dicom("1.2.3.5", 0), "application/dicom"))],
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("unsafe path", response.json()["detail"])

    def test_rejects_empty_upload(self):
        response = self.client.post(
            "/api/v1/imaging/dicom/upload",
            data={"patient_id": "patient-upload"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
