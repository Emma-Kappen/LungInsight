from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


STAGE01 = Path(__file__).resolve().parents[1] / "stages" / "01_hu_and_crop.py"
SPEC = importlib.util.spec_from_file_location("lunginsight_stage01_geometry", STAGE01)
STAGE01_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STAGE01_MODULE
SPEC.loader.exec_module(STAGE01_MODULE)


def slice_record(position, orientation=(1, 0, 0, 0, 1, 0), pixel_spacing=(0.7, 0.8)):
    return SimpleNamespace(
        ImageOrientationPatient=orientation,
        ImagePositionPatient=position,
        PixelSpacing=pixel_spacing,
        Rows=2,
        Columns=3,
        SliceThickness=1.0,
        PatientID="p1",
        StudyInstanceUID="study",
        SeriesInstanceUID="series",
        Manufacturer="test",
        KVP=None,
        ConvolutionKernel="test",
    )


class DicomGeometryTests(unittest.TestCase):
    def test_orders_by_oblique_plane_normal_not_patient_z(self):
        orientation = (0, 1, 0, 0, 0, 1)
        slices = [
            slice_record((2, 0, 0), orientation),
            slice_record((0, 0, 0), orientation),
            slice_record((1, 0, 0), orientation),
        ]
        ordered, spacing = STAGE01_MODULE.order_slices(slices)
        self.assertEqual([row.ImagePositionPatient[0] for row in ordered], [0, 1, 2])
        self.assertEqual(spacing, 1.0)
        metadata = STAGE01_MODULE.extract_metadata(ordered, spacing)
        np.testing.assert_allclose(
            metadata["direction_lps"],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
        )
        self.assertEqual(metadata["spacing_zyx_mm"], [1.0, 0.7, 0.8])

    def test_rejects_duplicate_slice_positions(self):
        with self.assertRaisesRegex(ValueError, "duplicate or irregular"):
            STAGE01_MODULE.order_slices(
                [slice_record((0, 0, 0)), slice_record((0, 0, 0))]
            )

    def test_rejects_inconsistent_orientation(self):
        with self.assertRaisesRegex(ValueError, "inconsistent image orientation"):
            STAGE01_MODULE.order_slices(
                [
                    slice_record((0, 0, 0)),
                    slice_record((0, 0, 1), orientation=(0, 1, 0, 1, 0, 0)),
                ]
            )

    def test_hu_conversion_applies_padding_and_rescale_once(self):
        first = SimpleNamespace(
            Rows=1,
            Columns=3,
            pixel_array=np.array([[0, 1, 2]], dtype=np.int16),
            RescaleSlope=2,
            RescaleIntercept=-1000,
            PixelPaddingValue=0,
            PixelPaddingRangeLimit=None,
        )
        volume = STAGE01_MODULE.slices_to_hu_volume([first])
        np.testing.assert_array_equal(volume[0], [[-1000, -998, -996]])

    def test_discovers_nested_extensionless_dicom_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp)
            study_uid = generate_uid()
            series_uid = generate_uid()
            for index in range(2):
                path = patient / "study" / "series" / f"slice_{index:04d}"
                path.parent.mkdir(parents=True, exist_ok=True)
                file_meta = FileMetaDataset()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                file_meta.MediaStorageSOPClassUID = CTImageStorage
                file_meta.MediaStorageSOPInstanceUID = generate_uid()
                dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=None)
                dataset.Modality = "CT"
                dataset.PatientID = "LIDC-IDRI-0001"
                dataset.StudyInstanceUID = study_uid
                dataset.SeriesInstanceUID = series_uid
                dataset.SOPClassUID = CTImageStorage
                dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                dataset.Rows = 1
                dataset.Columns = 1
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = 16
                dataset.BitsStored = 16
                dataset.HighBit = 15
                dataset.PixelRepresentation = 0
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, index]
                dataset.PixelSpacing = [0.7, 0.7]
                dataset.PixelData = np.array([[index]], dtype=np.uint16).tobytes()
                dataset.save_as(path)
            (patient / "study" / "annotations.xml").write_text("<annotation/>", encoding="utf-8")

            series = STAGE01_MODULE.load_ct_series(str(patient))
            self.assertEqual(len(series.datasets), 2)
            self.assertEqual(series.study_uid, study_uid)
            self.assertEqual(series.series_uid, series_uid)


if __name__ == "__main__":
    unittest.main()
