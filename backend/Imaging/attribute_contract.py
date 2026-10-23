"""Validation and audit helpers for checkpoint-backed attribute outputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_attribute_support_audit(
    *,
    model_targets: Sequence[str],
    supported_attributes: Sequence[str],
    checkpoint_groups: Mapping[str, Sequence[str]],
    attribute_scales: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    targets = tuple(str(name) for name in model_targets)
    if len(targets) != len(set(targets)):
        raise ValueError("Model target names must be unique.")
    target_index = {name: index for index, name in enumerate(targets)}
    checkpoint_by_attribute: dict[str, str] = {}
    for group, attributes in checkpoint_groups.items():
        for attribute in attributes:
            name = str(attribute)
            if name in checkpoint_by_attribute:
                raise ValueError(f"Attribute {name!r} is assigned to multiple checkpoint groups.")
            checkpoint_by_attribute[name] = str(group)

    audit: dict[str, dict[str, Any]] = {}
    for attribute in targets:
        supported = attribute in supported_attributes
        if supported and attribute not in target_index:
            raise ValueError(f"Supported attribute {attribute!r} has no model target.")
        group = checkpoint_by_attribute.get(attribute)
        if supported and group is None:
            raise ValueError(f"Supported attribute {attribute!r} has no checkpoint group.")
        if supported and attribute not in attribute_scales:
            raise ValueError(f"Supported attribute {attribute!r} has no schema rating semantics.")
        audit[attribute] = {
            "status": "supported" if supported else "unsupported_checkpoint_contract",
            "checkpoint_group": group,
            "head_output": "expected_rating" if supported else None,
            "target_index": target_index[attribute],
            "score_semantics": (
                attribute_scales[attribute].get("semantics") if supported else None
            ),
        }
    extra_supported = set(supported_attributes) - set(targets)
    if extra_supported:
        raise ValueError(f"Supported attributes are missing from the model target list: {sorted(extra_supported)}.")
    return audit
