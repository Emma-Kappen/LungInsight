import os
import numpy as np
import pandas as pd
import pylidc as pl
import feather # for writing data frame to disk (works with R)


def flatten_annotation(ann):
    '''
    Flattens annotations into a single row that can be added to a pandas DataFrame
    '''
    
    id_vals = np.array([
        ann.scan.patient_id,
        ann._nodule_id,
        ann.id,
        ann.scan_id], 
        dtype = '<U14')
    feature_vals = ann.feature_vals()
    return(id_vals, feature_vals)


def flatten_annotations(annotations):
    '''
    Take a list of annotations, return a pandas DataFrame
    '''
    if not isinstance(annotations, list):
        # makes sure that anns is a list, even if it is of length 1
        annotations = [annotations]

    # instantiate empty arrays for the values
    id_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[0].shape[0]), dtype = "<U14")
    feature_values = np.zeros((len(annotations), 
                       flatten_annotation(annotations[0])[1].shape[0]), dtype = "int64")
    
    # loop over list of annotations
    for i, ann in enumerate(annotations):
        id_vals, feature_vals = flatten_annotation(ann)
        id_values[i,:] = id_vals
        feature_values[i,:] = feature_vals
    
    # combine together in a pandas DataFrame
    df_ids = pd.DataFrame(id_values, columns = ["patient_id", "nodule_id", "annotation_id", "scan_id"])
    df_feat= pd.DataFrame(feature_values, columns = [
                                         'sublety', 'internalstructure', 'calcification',
                                         'sphericity', 'margin', 'lobulation', 'spiculation',
                                         'texture', 'malignancy'])
    df = pd.concat([df_ids, df_feat], axis = 1)
    return(df)


def flatten_annotations_by_nodule(scans):
    '''
    take a list of scans, return a pandas DataFrame
    '''
    
    # instantiate DataFrame
    df = flatten_annotations(scans[0].annotations[0]).iloc[0:0]
    df.assign(nodule_number = np.empty(0, dtype = "int32"))
    
    # loop over scans
    for scan in scans:
        # loop over nodules within a scan
        for i, nodule_annotations in enumerate(scan.cluster_annotations()):
            if not isinstance(nodule_annotations, list):
                # makes sure that anns is a list, even if it is of length 1
                nodule_annotations = [nodule_annotations]
            nodule_df = flatten_annotations(nodule_annotations)
            nodule_df = nodule_df.assign(nodule_number = i+1)
            df = pd.concat([df, nodule_df], axis = 0)
    return(df)

def get_intercept_and_slope(scan, verbose = False):
    ''' 
    scan is the results of a pydicom query
    returns the intercept and slope
    adapted from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    imgs = scan.load_all_dicom_images(verbose = verbose)
    slice0 = imgs[0]
    intercept = slice0.RescaleIntercept
    slope = slice0.RescaleSlope
    return(intercept, slope)

FEATURE_NAMES = [
    'subtlety', 'internalStructure', 'calcification', 'sphericity',
    'margin', 'lobulation', 'spiculation', 'texture', 'malignancy'
]

def to_hu(volume, intercept, slope):
    volume = volume.astype(np.float64)
    if slope != 1:
        volume *= slope
    volume += intercept
    return volume.astype(np.int16)


def get_annotation_center(annotation):
    for attr in ('centroid', 'center'):
        if hasattr(annotation, attr):
            center = getattr(annotation, attr)
            if callable(center):
                center = center()
            if center is not None:
                return tuple(np.asarray(center, dtype=float))
    raise AttributeError('Annotation object has no centroid or center attribute')


def _get_feature_value(annotation, feature_name):
    if hasattr(annotation, feature_name):
        return getattr(annotation, feature_name)
    lower_name = feature_name.lower()
    if hasattr(annotation, lower_name):
        return getattr(annotation, lower_name)
    return None


def get_normalized_semantic_labels(nodule_annotations):
    labels = {}
    for feature in FEATURE_NAMES:
        values = []
        for ann in nodule_annotations:
            value = _get_feature_value(ann, feature)
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
        if values:
            avg_score = float(np.mean(values))
            labels[feature] = float(np.clip((avg_score - 1.0) / 4.0, 0.0, 1.0))
        else:
            labels[feature] = np.nan
    return labels


def extract_nodule_patches_and_manifest(scans, output_dir, patch_size=64, manifest_name='dataset_manifest.csv'):
    '''
    Extract a fixed 64x64x64 isotropic patch for each clustered nodule and build a manifest.
    '''
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []
    for scan in scans:
        patient_id = scan.patient_id
        nodules = scan.cluster_annotations()
        intercept, slope = get_intercept_and_slope(scan)

        for nodule_idx, nodule_annotations in enumerate(nodules, start=1):
            if not isinstance(nodule_annotations, list):
                nodule_annotations = [nodule_annotations]

            nodule_id = f"{patient_id}_{nodule_idx:03d}"
            filename = f"nodule_{nodule_id}.npy"
            file_path = os.path.join(output_dir, filename)

            try:
                ann = nodule_annotations[0]
                center = get_annotation_center(ann)
                vol, _ = ann.uniform_cubic_resample(side_length=patch_size, verbose=False)

                if vol.ndim != 3 or vol.shape != (patch_size, patch_size, patch_size):
                    raise ValueError(f'Unexpected patch shape {vol.shape}, expected {(patch_size,)*3}')

                vol = to_hu(vol, intercept, slope)
                np.save(file=file_path, arr=vol)

            except Exception as exc:
                print(f"Skipping {patient_id} nodule {nodule_idx}: {exc}")
                continue

            labels = get_normalized_semantic_labels(nodule_annotations)
            row = {
                'file_path': os.path.abspath(file_path),
                'PatientID': patient_id,
                'NoduleID': nodule_id,
                'center_x': center[0],
                'center_y': center[1],
                'center_z': center[2],
            }
            row.update({f'{feature}_confidence': labels[feature] for feature in FEATURE_NAMES})
            manifest_rows.append(row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, manifest_name)
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def flatten_multiindex_columns(df, sep = "_"):
    '''
    If a pandas DataFrame has a hierarchical index,
    flatten to single level
    '''
    col_vals = df.columns.values
    flattened = [sep.join(x) for x in col_vals]
    stripped = [x[:-1] if sep == x[-1] else x for x in flattened]
    df.columns = stripped
    return df
              