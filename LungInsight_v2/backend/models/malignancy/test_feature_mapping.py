from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from generate_malignancy_features import (
    ATTRIBUTE_NAMES,
    GROUPS,
    atomic_save_npz,
    assemble_attribute_slots,
    build_group_channel_mapping,
    load_fold_cache,
    pack_feature_matrix,
    resolve_checkpoint_path,
)
from model_objective import MODEL_TARGETS


class FeatureMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_count = 4
        self.group_outputs = {}
        self.group_values = {}
        for group_index, group in enumerate(GROUPS):
            values = np.empty((self.sample_count, len(MODEL_TARGETS), 2), dtype=np.float32)
            for channel in range(len(MODEL_TARGETS)):
                values[:, channel, 0] = group_index * 100 + channel
                values[:, channel, 1] = group_index * 1000 + channel
            self.group_outputs[group] = values
            self.group_values[group] = group_index

    def test_all_attributes_use_their_named_model_channels_once(self) -> None:
        slots = assemble_attribute_slots(self.group_outputs, self.sample_count)
        self.assertEqual(slots.shape, (self.sample_count, 6, 2))
        mapping = build_group_channel_mapping()
        assignments = [entry for entries in mapping.values() for entry in entries]
        self.assertEqual(sorted(entry[1] for entry in assignments), list(range(6)))
        self.assertEqual(len(assignments), 6)

        for feature_slot, name in enumerate(ATTRIBUTE_NAMES):
            owner = next(group for group, targets in GROUPS.items() if name in targets)
            group_index = self.group_values[owner]
            channel = MODEL_TARGETS.index(name)
            np.testing.assert_array_equal(slots[:, feature_slot, 0], group_index * 100 + channel)
            np.testing.assert_array_equal(slots[:, feature_slot, 1], group_index * 1000 + channel)

    def test_joint_group_uses_lobulation_and_spiculation_channels(self) -> None:
        slots = assemble_attribute_slots(self.group_outputs, self.sample_count)
        joint_values = self.group_outputs["lobulation_spiculation"]
        for name in ("lobulation", "spiculation"):
            feature_slot = ATTRIBUTE_NAMES.index(name)
            output_channel = MODEL_TARGETS.index(name)
            np.testing.assert_array_equal(
                slots[:, feature_slot, :], joint_values[:, output_channel, :]
            )

    def test_oof_and_validation_paths_share_the_same_packing(self) -> None:
        oof_slots = assemble_attribute_slots(self.group_outputs, self.sample_count)
        val_test_slots = assemble_attribute_slots(self.group_outputs, self.sample_count)
        oof_features = pack_feature_matrix(oof_slots)
        val_test_features = pack_feature_matrix(val_test_slots)
        self.assertEqual(oof_features.shape, (self.sample_count, 12))
        np.testing.assert_array_equal(oof_features, val_test_features)
        np.testing.assert_array_equal(oof_features[:, :6], oof_slots[:, :, 0])
        np.testing.assert_array_equal(oof_features[:, 6:], oof_slots[:, :, 1])

    def test_unknown_target_raises_descriptive_mapping_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model target.*bogus"):
            build_group_channel_mapping(groups={"broken": ("bogus",)})

    def test_incompatible_output_shape_raises_descriptive_error(self) -> None:
        outputs = dict(self.group_outputs)
        outputs["lobulation_spiculation"] = outputs["lobulation_spiculation"][:, :6, :]
        with self.assertRaisesRegex(ValueError, "group 'lobulation_spiculation'.*actual output dimensions"):
            assemble_attribute_slots(outputs, self.sample_count)

    def test_missing_duplicate_and_nonfinite_slots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing malignancy attribute mappings"):
            build_group_channel_mapping(groups={"only_subtlety": ("subtlety",)})
        with self.assertRaisesRegex(ValueError, "duplicate assignment"):
            build_group_channel_mapping(groups={
                "group_a": ("subtlety", "sphericity", "margin", "lobulation", "spiculation"),
                "group_b": ("spiculation", "texture"),
            })
        outputs = dict(self.group_outputs)
        outputs["texture"][0, MODEL_TARGETS.index("texture"), 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite values for target 'texture'"):
            assemble_attribute_slots(outputs, self.sample_count)

    def test_short_checkpoint_path_uses_canonical_group_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            checkpoint_root = Path(temporary_directory) / "checkpoints" / "hybrid"
            checkpoint = checkpoint_root / "subtlety" / "best.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")

            resolved = resolve_checkpoint_path(
                "checkpoints/hybrid/subtlety/best.pth",
                "subtlety",
                checkpoint_root,
            )

        self.assertEqual(resolved, checkpoint.resolve())

    def test_missing_checkpoint_error_names_group_and_canonical_path(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(FileNotFoundError, "group 'texture'.*canonical checkpoint path"):
                resolve_checkpoint_path(
                    "checkpoints/hybrid/texture/best.pth",
                    "texture",
                    Path(temporary_directory) / "checkpoints" / "hybrid",
                )

    def test_fold_cache_round_trip_and_provenance_validation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fold_00.npz"
            metadata = {"seed": 123, "split_sha256": "fixed-split"}
            positions = np.asarray([0, 2, 5], dtype=np.int64)
            h5_indices = np.asarray([10, 12, 15], dtype=np.int64)
            features = np.arange(36, dtype=np.float32).reshape(3, 12)
            atomic_save_npz(
                path,
                X=features,
                train_positions=positions,
                h5_indices=h5_indices,
                metadata=np.asarray(__import__("json").dumps(metadata, sort_keys=True)),
            )

            loaded = load_fold_cache(path, metadata, positions, h5_indices)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded, features)
            self.assertIsNone(load_fold_cache(path, {"seed": 456}, positions, h5_indices))


if __name__ == "__main__":
    unittest.main()