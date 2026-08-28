"""DICOM/NIfTI ingestion, HU conversion, windowing, resampling and previews."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import base64, io, shutil, tarfile, zipfile
import numpy as np
from PIL import Image
import pydicom
import SimpleITK as sitk

@dataclass
class CTVolume:
    volume_hu: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float] | None
    series_uid: str | None
    scan_date: str | None

def unpack_input(src: str | Path, work_dir: str | Path) -> Path:
    src, work_dir = Path(src), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        return src
    name = src.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            z.extractall(work_dir)
        return work_dir
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(src, "r:gz") as t:
            t.extractall(work_dir)
        return work_dir
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return src
    if name.endswith(".dcm"):
        out = work_dir / "dicom"
        out.mkdir(exist_ok=True)
        shutil.copy2(src, out / src.name)
        return out
    raise ValueError("CT input must be a DICOM folder/file, ZIP/TAR.GZ, or NIfTI.")

def _dicom_files(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".dcm"]

def _load_dicom(folder: Path) -> CTVolume:
    items = []
    for p in _dicom_files(folder):
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if not hasattr(ds, "PixelData") or getattr(ds, "Modality", "CT") != "CT":
                continue
            z = None
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "SliceLocation"):
                z = float(ds.SliceLocation)
            if z is None:
                z = float(getattr(ds, "InstanceNumber", len(items)))
            items.append((z, ds))
        except Exception:
            continue
    if not items:
        raise ValueError("No readable CT DICOM slices were found.")
    items.sort(key=lambda x: x[0])
    uid = str(getattr(items[0][1], "SeriesInstanceUID", "")) or None
    arrays = []
    for _, ds in items:
        a = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(a * slope + intercept)
    vol = np.stack(arrays, axis=0)
    ps = getattr(items[0][1], "PixelSpacing", [1.0, 1.0])
    if len(items) > 1:
        dz = float(np.median(np.abs(np.diff([x[0] for x in items]))))
    else:
        dz = float(getattr(items[0][1], "SliceThickness", 1.0))
    return CTVolume(vol, (dz, float(ps[0]), float(ps[1])), tuple(getattr(items[0][1], "ImagePositionPatient", [])) or None,
                    uid, str(getattr(items[0][1], "StudyDate", "")) or None)

def _load_nifti(path: Path) -> CTVolume:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()
    return CTVolume(arr, (float(sp[2]), float(sp[1]), float(sp[0])), tuple(img.GetOrigin()), None, None)

def load_ct(src: str | Path, work_dir: str | Path) -> CTVolume:
    path = unpack_input(src, work_dir)
    if path.is_file() and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz")):
        return _load_nifti(path)
    return _load_dicom(path)

def resample_isotropic(ct: CTVolume, spacing: float = 1.0) -> CTVolume:
    # SimpleITK preserves physical geometry and handles arbitrary source spacing.
    img = sitk.GetImageFromArray(ct.volume_hu)
    img.SetSpacing((ct.spacing_zyx[2], ct.spacing_zyx[1], ct.spacing_zyx[0]))
    size = img.GetSize()
    old = img.GetSpacing()
    new_size = [max(1, int(round(size[i] * old[i] / spacing))) for i in range(3)]
    f = sitk.ResampleImageFilter()
    f.SetInterpolator(sitk.sitkLinear)
    f.SetOutputSpacing((spacing, spacing, spacing))
    f.SetSize(new_size)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    out = f.Execute(img)
    arr = sitk.GetArrayFromImage(out).astype(np.float32)
    return CTVolume(arr, (spacing, spacing, spacing), tuple(out.GetOrigin()), ct.series_uid, ct.scan_date)

def window_hu(volume: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((volume - lo) / ww, 0, 1)

def preview_slice(volume: np.ndarray, index: int, wl: float = -600, ww: float = 1500, max_side: int = 768) -> str:
    index = max(0, min(int(index), volume.shape[0] - 1))
    img = (window_hu(volume[index], wl, ww) * 255).astype(np.uint8)
    pil = Image.fromarray(img).convert("L")
    pil.thumbnail((max_side, max_side))
    b = io.BytesIO()
    pil.save(b, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

def representative_slice(volume: np.ndarray) -> str:
    return preview_slice(volume, volume.shape[0] // 2)
