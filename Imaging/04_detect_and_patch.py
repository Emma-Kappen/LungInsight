import numpy as np
from scipy import ndimage


def characteristic_candidate_detect(
    volume_hu_masked: np.ndarray,
    lung_mask: np.ndarray,
    spacing_zyx,
    min_diameter_mm: float = 4.0,
    max_diameter_mm: float = 40.0,
    background_sigma_mm: float = 5.0,
    min_contrast_hu: float = 40.0,
    min_component_voxels: int = 20,
    max_raw_candidates: int = 8000,
):
    """
    Generate high-recall pulmonary lesion candidate regions using
    suspicious morphological/intensity characteristics rather than
    spherical LoG blob detection.

    IMPORTANT
    ---------
    This function DOES NOT classify candidates as malignant or benign.

    It only answers:

        "Does this region contain an abnormal pulmonary structure
         sufficiently unlike normal aerated lung that it should be
         examined by the downstream classifier?"

    Candidate generation deliberately favours recall over precision.

    Characteristics used:
        1. Local attenuation abnormality
        2. Local contrast against lung parenchyma
        3. Heterogeneous internal attenuation
        4. Boundary irregularity
        5. Non-spherical / elongated morphology is allowed
        6. Physical size plausibility
        7. Multi-slice persistence

    Returns
    -------
    list[dict]
        Candidate dictionaries compatible with the downstream NMS /
        patch-extraction pipeline.
    """

    if volume_hu_masked.shape != lung_mask.shape:
        raise ValueError(
            f"Shape mismatch: volume {volume_hu_masked.shape} "
            f"vs lung_mask {lung_mask.shape}"
        )

    sz, sy, sx = spacing_zyx

    # ------------------------------------------------------------------
    # 1. Build a local-background contrast map
    # ------------------------------------------------------------------
    #
    # Instead of asking "is this voxel bright?", ask:
    #
    #     "is this voxel substantially denser than its local lung
    #      background?"
    #
    # This is important for ground-glass / relatively low-density
    # lesions which may not cross a hard HU threshold.
    # ------------------------------------------------------------------

    sigma_vox = (
        max(background_sigma_mm / sz, 0.5),
        max(background_sigma_mm / sy, 0.5),
        max(background_sigma_mm / sx, 0.5),
    )

    lung_values = volume_hu_masked[lung_mask]

    if lung_values.size == 0:
        return []

    # Broad smoothing estimates local parenchymal background.
    background = ndimage.gaussian_filter(
        volume_hu_masked,
        sigma=sigma_vox,
        mode="nearest",
    )

    contrast = volume_hu_masked - background

    # Only consider the lung.
    contrast[~lung_mask] = -np.inf

    # ------------------------------------------------------------------
    # 2. Two complementary foreground mechanisms
    # ------------------------------------------------------------------
    #
    # A. Absolute soft-tissue component:
    #       useful for solid nodules/masses.
    #
    # B. Local contrast component:
    #       useful for lower-density / ground-glass abnormalities.
    #
    # Their UNION is intentional.
    # ------------------------------------------------------------------

    # Broad threshold, deliberately permissive.
    #
    # This is NOT a "cancer threshold". It merely identifies structures
    # substantially denser than normal aerated lung.
    soft_tissue_mask = (
        (volume_hu_masked > -650.0)
        & lung_mask
    )

    contrast_mask = (
        (contrast > min_contrast_hu)
        & lung_mask
    )

    candidate_foreground = (
        soft_tissue_mask
        | contrast_mask
    )

    # ------------------------------------------------------------------
    # 3. Remove isolated noise without enforcing spherical geometry
    # ------------------------------------------------------------------

    # 6-connected structure is intentionally conservative.
    structure = ndimage.generate_binary_structure(3, 1)

    candidate_foreground = ndimage.binary_opening(
        candidate_foreground,
        structure=structure,
        iterations=1,
    )

    candidate_foreground = ndimage.binary_closing(
        candidate_foreground,
        structure=structure,
        iterations=1,
    )

    # ------------------------------------------------------------------
    # 4. Connected components
    # ------------------------------------------------------------------

    labels, num_labels = ndimage.label(
        candidate_foreground,
        structure=ndimage.generate_binary_structure(3, 3),
    )

    if num_labels == 0:
        return []

    voxel_volume_mm3 = sz * sy * sx

    candidates = []

    # Physical size limits.
    min_volume_mm3 = (np.pi / 6.0) * (min_diameter_mm ** 3)
    max_volume_mm3 = (np.pi / 6.0) * (max_diameter_mm ** 3)

    # ------------------------------------------------------------------
    # 5. Analyse each abnormal region
    # ------------------------------------------------------------------

    for label_id in range(1, num_labels + 1):

        component = labels == label_id

        voxel_count = int(component.sum())

        if voxel_count < min_component_voxels:
            continue

        volume_mm3 = voxel_count * voxel_volume_mm3

        # Reject components that are obviously enormous structures
        # (e.g. large connected vessels / hilar structures).
        #
        # This is only a sanity bound, not a malignancy criterion.
        if volume_mm3 > max_volume_mm3 * 8.0:
            continue

        coords = np.argwhere(component)

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        extent_z_mm = (z_max - z_min + 1) * sz
        extent_y_mm = (y_max - y_min + 1) * sy
        extent_x_mm = (x_max - x_min + 1) * sx

        max_extent_mm = max(
            extent_z_mm,
            extent_y_mm,
            extent_x_mm,
        )

        # Reject tiny objects that cannot plausibly correspond to
        # the minimum candidate size.
        if max_extent_mm < min_diameter_mm:
            continue

        # ------------------------------------------------------------------
        # 6. Internal intensity characteristics
        # ------------------------------------------------------------------

        lesion_values = volume_hu_masked[component]
        lesion_contrast = contrast[component]

        mean_hu = float(np.mean(lesion_values))
        std_hu = float(np.std(lesion_values))

        mean_contrast = float(np.mean(lesion_contrast))
        max_contrast = float(np.max(lesion_contrast))

        # Heterogeneity:
        #
        # We don't use this to say "heterogeneous = cancer".
        # It simply gives heterogeneous regions higher candidate priority.
        heterogeneity = np.clip(
            std_hu / 200.0,
            0.0,
            1.0,
        )

        contrast_score = np.clip(
            mean_contrast / 200.0,
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # 7. Shape characteristics
        # ------------------------------------------------------------------
        #
        # Do NOT use sphericity as a rejection criterion.
        #
        # An irregular / lobulated / spiculated mass is precisely the
        # type of lesion the new detector is intended to recover.
        # ------------------------------------------------------------------

        bbox_volume_mm3 = (
            extent_z_mm
            * extent_y_mm
            * extent_x_mm
        )

        fill_ratio = (
            volume_mm3 / bbox_volume_mm3
            if bbox_volume_mm3 > 0
            else 0.0
        )

        irregularity = 1.0 - np.clip(
            fill_ratio,
            0.0,
            1.0,
        )

        # Aspect ratio captures elongated structures, but does NOT
        # reject them. It is retained as a feature because irregular
        # extensions can be useful for candidate prioritisation.
        min_extent = max(
            min(extent_z_mm, extent_y_mm, extent_x_mm),
            1e-6,
        )

        elongation = max_extent_mm / min_extent

        # ------------------------------------------------------------------
        # 8. Boundary / spiculation proxy
        # ------------------------------------------------------------------
        #
        # A simple geometric proxy:
        #
        #   dilated component - component
        #
        # gives the immediate surrounding shell. If that shell contains
        # strong positive contrast, the lesion has a strong/irregular
        # interface with surrounding lung.
        #
        # This is NOT a true clinical spiculation detector, but it gives
        # us a useful candidate-generation feature without requiring a
        # spherical model.
        # ------------------------------------------------------------------

        boundary_shell = (
            ndimage.binary_dilation(
                component,
                structure=ndimage.generate_binary_structure(3, 1),
                iterations=2,
            )
            & lung_mask
            & ~component
        )

        if boundary_shell.any():
            boundary_contrast = contrast[boundary_shell]

            positive_boundary_fraction = float(
                np.mean(boundary_contrast > min_contrast_hu)
            )

            boundary_mean_contrast = float(
                np.mean(boundary_contrast)
            )
        else:
            positive_boundary_fraction = 0.0
            boundary_mean_contrast = 0.0

        # ------------------------------------------------------------------
        # 9. Slice persistence
        # ------------------------------------------------------------------

        slice_presence = component.any(axis=(1, 2))
        occupied_slices = int(slice_presence.sum())

        persistence = np.clip(
            (
                occupied_slices
                * sz
                / max(min_diameter_mm, 1.0)
            ),
            0.0,
            1.0,
        )

        # ------------------------------------------------------------------
        # 10. Find the best seed INSIDE the component
        # ------------------------------------------------------------------
        #
        # We don't simply use the component centroid.
        #
        # For an irregular mass, centroid can fall in a low-density
        # region or even outside the visually strongest part.
        #
        # Use local contrast to select the seed that should anchor
        # downstream patch extraction.
        # ------------------------------------------------------------------

        component_indices = np.argwhere(component)

        component_contrasts = contrast[component]

        best_idx = int(
            np.argmax(component_contrasts)
        )

        seed_z, seed_y, seed_x = component_indices[best_idx]

        # ------------------------------------------------------------------
        # 11. Candidate priority
        # ------------------------------------------------------------------
        #
        # This is NOT a cancer score.
        #
        # It only determines which candidate regions are retained first
        # when there are many possible regions.
        # ------------------------------------------------------------------

        size_score = np.clip(
            (
                np.log1p(max_extent_mm)
                - np.log1p(min_diameter_mm)
            )
            /
            (
                np.log1p(max_diameter_mm)
                - np.log1p(min_diameter_mm)
                + 1e-8
            ),
            0.0,
            1.0,
        )

        boundary_score = np.clip(
            (
                0.6 * positive_boundary_fraction
                +
                0.4 * np.clip(
                    boundary_mean_contrast / 200.0,
                    0.0,
                    1.0,
                )
            ),
            0.0,
            1.0,
        )

        # Candidate priority deliberately favours multiple independent
        # characteristics rather than spherical shape.
        candidate_score = (
            0.30 * contrast_score
            + 0.20 * heterogeneity
            + 0.20 * boundary_score
            + 0.15 * irregularity
            + 0.10 * persistence
            + 0.05 * size_score
        )

        candidates.append({
            "z": int(seed_z),
            "y": int(seed_y),
            "x": int(seed_x),

            # This is a candidate-priority score ONLY.
            "candidate_score": float(candidate_score),

            "volume_mm3": float(volume_mm3),

            "extent_z_mm": float(extent_z_mm),
            "extent_y_mm": float(extent_y_mm),
            "extent_x_mm": float(extent_x_mm),

            "mean_hu": float(mean_hu),
            "std_hu": float(std_hu),

            "mean_contrast_hu": float(mean_contrast),
            "max_contrast_hu": float(max_contrast),

            "fill_ratio": float(fill_ratio),
            "irregularity": float(irregularity),
            "elongation": float(elongation),

            "boundary_contrast_hu": float(
                boundary_mean_contrast
            ),
            "boundary_positive_fraction": float(
                positive_boundary_fraction
            ),

            "occupied_slices": int(occupied_slices),
            "persistence": float(persistence),
        })

    # ------------------------------------------------------------------
    # 12. Limit raw candidate count
    # ------------------------------------------------------------------

    candidates.sort(
        key=lambda c: c["candidate_score"],
        reverse=True,
    )

    if len(candidates) > max_raw_candidates:
        print(
            f"[warn] {len(candidates)} characteristic candidates "
            f"exceed --max-raw-candidates ({max_raw_candidates}); "
            f"keeping the highest-priority {max_raw_candidates}."
        )

        candidates = candidates[:max_raw_candidates]

    return candidates