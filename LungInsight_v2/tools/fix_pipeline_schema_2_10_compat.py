#!/usr/bin/env python3
"""Add schema 2.10 support to an existing LungInsight pipeline schema loader.

Unlike the previous repair, this searches AST literals throughout pipeline_schema.py,
including locals inside load_schema() and inline allowlists. It edits only literal
containers whose entries are all version strings and include 2.6, 2.7 and 2.8.
It fails closed if that contract cannot be found.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

TARGET_VERSION = "2.10"
REQUIRED_EXISTING_VERSIONS = {"2.6", "2.7", "2.8"}
_VERSION_RE = re.compile(r"^\d+\.\d+$")


class CompatibilityPatchError(RuntimeError):
    """The local source does not match the expected safe-to-patch contract."""


def _natural_version_key(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in v.split("."))
    except ValueError:
        return (999999,)


def _get_literal_container(node: ast.AST) -> tuple[list[str], str] | None:
    """Extract a literal version list/set/tuple; ignore any non-version data."""
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return None
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    if not isinstance(value, (set, frozenset, list, tuple)) or not value:
        return None
    values = [str(item) for item in value]
    if not all(isinstance(item, str) and _VERSION_RE.fullmatch(item) for item in value):
        return None
    if not REQUIRED_EXISTING_VERSIONS.issubset(set(values)):
        return None
    if isinstance(node, ast.Set):
        kind = "set"
    elif isinstance(node, ast.List):
        kind = "list"
    else:
        kind = "tuple"
    return values, kind


def _render(values: list[str], kind: str) -> str:
    values = sorted(set(values), key=_natural_version_key)
    items = ", ".join(repr(v) for v in values)
    if kind == "set":
        return "{" + items + "}"
    if kind == "list":
        return "[" + items + "]"
    return "(" + items + ("," if len(values) == 1 else "") + ")"


def _absolute_byte_offsets(source: str) -> list[int]:
    """Start byte offset for each 1-indexed source line (AST columns are UTF-8 bytes)."""
    raw_lines = source.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in raw_lines:
        starts.append(offset)
        offset += len(line.encode("utf-8"))
    return starts


def patch_loader_source(source: str) -> tuple[str, bool]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CompatibilityPatchError(
            f"pipeline_schema.py has a syntax error at line {exc.lineno}: {exc.msg}"
        ) from exc

    matches: list[tuple[ast.AST, list[str], str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            continue
        found = _get_literal_container(node)
        if found is not None:
            values, kind = found
            matches.append((node, values, kind))

    if not matches:
        raise CompatibilityPatchError(
            "Could not locate a literal supported-version collection containing all of "
            "'2.6', '2.7', and '2.8' anywhere in pipeline_schema.py. No changes were made. "
            "Inspect the schema-version check in load_schema(); do not edit checkpoints."
        )

    if all(TARGET_VERSION in values for _, values, _ in matches):
        return source, False

    starts = _absolute_byte_offsets(source)
    replacements: list[tuple[int, int, bytes]] = []
    for node, values, kind in matches:
        if TARGET_VERSION in values:
            continue
        if not hasattr(node, "end_lineno") or node.end_lineno is None or node.end_col_offset is None:
            raise CompatibilityPatchError("AST source positions unavailable; no changes were made.")
        start = starts[node.lineno - 1] + node.col_offset
        end = starts[node.end_lineno - 1] + node.end_col_offset
        replacement = _render(values + [TARGET_VERSION], kind).encode("utf-8")
        replacements.append((start, end, replacement))

    # Replace from the end to preserve offsets. Parent containers may also match a nested
    # literal, so remove any replacement fully contained in another replacement span.
    filtered: list[tuple[int, int, bytes]] = []
    for item in sorted(replacements, key=lambda x: (x[0], -x[1])):
        start, end, _ = item
        if any(parent_start <= start and end <= parent_end for parent_start, parent_end, _ in filtered):
            continue
        filtered.append(item)

    raw = source.encode("utf-8")
    for start, end, replacement in sorted(filtered, key=lambda x: x[0], reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    patched = raw.decode("utf-8")
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        raise CompatibilityPatchError(
            f"Proposed loader patch did not parse at line {exc.lineno}: {exc.msg}. No changes were made."
        ) from exc
    return patched, True


def patch_schema_document(schema_path: Path) -> tuple[str, bool]:
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise CompatibilityPatchError(f"Could not read schema JSON at {schema_path}: {exc}") from exc
    if not isinstance(schema, dict):
        raise CompatibilityPatchError("meta_schema.json must be a JSON object; no changes made.")
    if str(schema.get("schema_version")) != TARGET_VERSION:
        raise CompatibilityPatchError(
            f"Expected root schema_version={TARGET_VERSION!r}; found {schema.get('schema_version')!r}. No changes made."
        )
    policy = schema.get("metadata_policy")
    if not isinstance(policy, dict):
        raise CompatibilityPatchError("meta_schema.json has no metadata_policy object; no changes made.")
    supported = policy.get("supported_schema_versions")
    if not isinstance(supported, list) or not all(isinstance(v, (str, int, float)) for v in supported):
        raise CompatibilityPatchError("metadata_policy.supported_schema_versions must be a list; no changes made.")
    versions = [str(v) for v in supported]
    if not REQUIRED_EXISTING_VERSIONS.issubset(set(versions)):
        raise CompatibilityPatchError(
            f"Schema metadata versions {versions!r} do not contain the expected existing contract "
            f"{sorted(REQUIRED_EXISTING_VERSIONS)!r}; no changes made."
        )
    if TARGET_VERSION in versions:
        return json.dumps(schema, indent=2, ensure_ascii=False) + "\n", False
    versions.append(TARGET_VERSION)
    versions = sorted(set(versions), key=_natural_version_key)
    policy["supported_schema_versions"] = versions
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n", True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="LungInsight_v2 project root")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    loader = root / "backend" / "Imaging" / "pipeline_schema.py"
    schema_path = root / "meta_schema.json"
    for path in (loader, schema_path):
        if not path.is_file():
            print(f"ERROR: required file does not exist: {path}", file=sys.stderr)
            return 2

    try:
        loader_src = loader.read_text(encoding="utf-8-sig")
        schema_src = schema_path.read_text(encoding="utf-8-sig")
        patched_loader, loader_changed = patch_loader_source(loader_src)
        # Patch the on-disk schema through the same strict schema check; we only read the path.
        patched_schema = _patch_schema_text(schema_src)
        schema_changed = patched_schema != schema_src
        ast.parse(patched_loader)
        json.loads(patched_schema)
    except CompatibilityPatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not loader_changed and not schema_changed:
        print(f"Already compatible: schema {TARGET_VERSION} is accepted by loader and schema metadata.")
        return 0

    backup_dir = root / "backups" / ("schema_2_10_compat_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(loader, backup_dir / "pipeline_schema.py")
    shutil.copy2(schema_path, backup_dir / "meta_schema.json")
    try:
        if loader_changed:
            loader.write_text(patched_loader, encoding="utf-8", newline="")
        if schema_changed:
            schema_path.write_text(patched_schema, encoding="utf-8", newline="")
    except Exception:
        shutil.copy2(backup_dir / "pipeline_schema.py", loader)
        shutil.copy2(backup_dir / "meta_schema.json", schema_path)
        raise

    print(json.dumps({
        "status": "patched",
        "schema_version": TARGET_VERSION,
        "loader_changed": loader_changed,
        "schema_metadata_changed": schema_changed,
        "patched_version_literal_count": "all matching literal collections in loader",
        "backup_directory": str(backup_dir),
        "files": [str(loader), str(schema_path)],
        "note": "Only the supported-schema allowlist and metadata policy are adjusted. Model checkpoints are untouched."
    }, indent=2))
    return 0


def _patch_schema_text(text: str) -> str:
    """Version-independent internal form used by main; uses same schema contract validation."""
    # Keep the patch function path-based for tests/API consistency, but avoid temporary files here.
    try:
        schema = json.loads(text)
    except Exception as exc:
        raise CompatibilityPatchError(f"Could not parse meta_schema.json: {exc}") from exc
    if not isinstance(schema, dict) or str(schema.get("schema_version")) != TARGET_VERSION:
        version = schema.get("schema_version") if isinstance(schema, dict) else None
        raise CompatibilityPatchError(f"Expected root schema_version={TARGET_VERSION!r}; found {version!r}.")
    policy = schema.get("metadata_policy")
    if not isinstance(policy, dict):
        raise CompatibilityPatchError("meta_schema.json has no metadata_policy object.")
    supported = policy.get("supported_schema_versions")
    if not isinstance(supported, list):
        raise CompatibilityPatchError("metadata_policy.supported_schema_versions must be a list.")
    versions = [str(v) for v in supported]
    if not REQUIRED_EXISTING_VERSIONS.issubset(set(versions)):
        raise CompatibilityPatchError(f"Unexpected schema metadata version contract: {versions!r}.")
    if TARGET_VERSION not in versions:
        versions.append(TARGET_VERSION)
    policy["supported_schema_versions"] = sorted(set(versions), key=_natural_version_key)
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
