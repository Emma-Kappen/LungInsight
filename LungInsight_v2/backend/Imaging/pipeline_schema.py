"""Shared accessors and validation for LungInsight's authoritative meta_schema.json.

The schema file is configuration, not model weights. This helper intentionally
never writes to checkpoints or other trained model artifacts.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

class PipelineSchemaError(RuntimeError):
    """Configuration or contract error with a user-facing remediation hint."""


def repository_root(script_file: str | Path | None = None) -> Path:
    """Find the canonical LungInsight_v2 root from a Stage script location.

    Prefer the repository root over a nearby ``meta_schema.json``. Some older
    checkouts contain a stale copy beside the Imaging scripts; choosing the
    nearest file first can silently select that outdated schema.
    """
    anchor = Path(script_file).resolve() if script_file else Path(__file__).resolve()
    if anchor.is_file():
        anchor = anchor.parent
    ancestors = (anchor, *anchor.parents)

    # First, prefer the explicitly named project root even if a nested legacy
    # schema file exists. This is the layout used by the supported repository.
    for candidate in ancestors:
        if candidate.name.casefold() == "lunginsight_v2":
            return candidate

    # For renamed/forked checkouts, use the closest schema-bearing ancestor as
    # a compatibility fallback after checking for the canonical root name.
    for candidate in ancestors:
        if (candidate / "meta_schema.json").is_file():
            return candidate

    return Path(__file__).resolve().parents[2]


def load_schema(
    script_file: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate meta_schema.json with actionable errors."""
    env_path = os.environ.get("LUNGINSIGHT_META_SCHEMA")
    candidates: list[Path] = []
    if schema_path is not None:
        candidates.append(Path(schema_path).expanduser())
    elif env_path:
        candidates.append(Path(env_path).expanduser())
    else:
        root = repository_root(script_file)
        candidates.extend([root / "meta_schema.json", Path(__file__).resolve().parent / "meta_schema.json"])

    schema_file = next((p.resolve() for p in candidates if p.is_file()), None)
    if schema_file is None:
        tried = "\n  ".join(str(p.resolve()) for p in candidates)
        raise FileNotFoundError(
            "Required metadata schema 'meta_schema.json' was not found. "
            f"Looked in:\n  {tried}\n"
            "Fix: place the canonical meta_schema.json in the LungInsight_v2 repository root, "
            "or set LUNGINSIGHT_META_SCHEMA to its absolute path."
        )

    try:
        with schema_file.open("r", encoding="utf-8-sig") as handle:
            schema = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PipelineSchemaError(
            f"Malformed metadata schema at {schema_file}, line {exc.lineno}, column {exc.colno}: {exc.msg}. "
            "Fix the JSON syntax before running the pipeline."
        ) from exc
    except OSError as exc:
        raise PipelineSchemaError(f"Could not read metadata schema {schema_file}: {exc}") from exc

    if not isinstance(schema, dict):
        raise PipelineSchemaError(f"Metadata schema {schema_file} must contain a JSON object at its top level.")
    version = schema.get("schema_version")
    if not isinstance(version, str) or not version.strip():
        raise PipelineSchemaError(
            f"Metadata schema {schema_file} is missing a non-empty 'schema_version'. "
            "Fix: add the schema's version to meta_schema.json."
        )
    policy = schema.get("metadata_policy")
    if not isinstance(policy, dict):
        raise PipelineSchemaError(
            f"Metadata schema {schema_file} is missing the 'metadata_policy' object. "
            "Fix: restore the metadata policy from the matching schema bundle."
        )
    declared_supported = policy.get("supported_schema_versions")
    if (not isinstance(declared_supported, list) or not declared_supported
            or any(not isinstance(item, str) or not item.strip() for item in declared_supported)):
        raise PipelineSchemaError(
            "meta_schema.json metadata_policy.supported_schema_versions must be a non-empty list "
            "of version strings. Fix the schema version list; scripts do not maintain a separate list."
        )
    supported = set(declared_supported)
    if version not in supported:
        raise PipelineSchemaError(
            f"Unsupported meta_schema.json version {version!r} in {schema_file}; "
            f"this pipeline bundle supports {sorted(supported)}. "
            "Use the matching pipeline scripts/schema bundle; do not change checkpoint files to resolve this."
        )
    schema["_schema_file"] = str(schema_file)
    schema["_repository_root"] = str(repository_root(script_file))
    return schema


def get_required(data: dict[str, Any], dotted_path: str, *, expected_type: type | tuple[type, ...] | None = None) -> Any:
    """Fetch a required schema value and validate its type/non-emptiness."""
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PipelineSchemaError(
                f"meta_schema.json is missing required key '{dotted_path}'. "
                "Fix: restore that key from the matching schema bundle; scripts must not hardcode a fallback."
            )
        current = current[part]
    if current is None or current == "":
        raise PipelineSchemaError(
            f"meta_schema.json key '{dotted_path}' has an empty value. "
            "Fix: provide a valid configuration value in the schema."
        )
    if expected_type is not None and not isinstance(current, expected_type):
        raise PipelineSchemaError(
            f"meta_schema.json key '{dotted_path}' has type {type(current).__name__}; "
            f"expected {expected_type}. Fix the schema value's type."
        )
    return current


def check_meta_schema_version(
    payload: Any,
    schema: dict[str, Any],
    *,
    context: str,
    require: bool = False,
) -> None:
    """Validate a stage artifact's schema provenance; accept untagged legacy data with a warning."""
    if not isinstance(payload, dict):
        raise PipelineSchemaError(f"{context} must be a JSON object to validate schema provenance.")
    expected = str(schema["schema_version"])
    found = payload.get("meta_schema_version")
    if found is None:
        if require:
            raise PipelineSchemaError(
                f"{context} is missing 'meta_schema_version'. Re-run the upstream stage with this pipeline bundle "
                "to produce schema-tagged artifacts."
            )
        warnings.warn(
            f"{context} has no meta_schema_version; accepting it as a legacy artifact. "
            "Re-run the upstream stage for an explicit schema compatibility check.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if str(found) != expected:
        raise PipelineSchemaError(
            f"Schema version mismatch in {context}: artifact has meta_schema_version={found!r}, "
            f"but this pipeline requires {expected!r}. Re-run the upstream stage with the same meta_schema.json. "
            "Trained checkpoints do not need to be retrained or modified."
        )


def validate_file(path: str | Path, *, description: str, fix: str | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        suffix = f" Fix: {fix}" if fix else ""
        raise FileNotFoundError(f"Missing {description}: {resolved}.{suffix}")
    return resolved


def validate_stage_version(found: Any, expected: str, *, context: str, field: str = "schema_version") -> None:
    if found is None:
        return  # Legacy input may be accepted when the metadata-schema version is also absent.
    if str(found) != str(expected):
        raise PipelineSchemaError(
            f"Incompatible {context} {field}: found {found!r}, expected {expected!r}. "
            "Regenerate the upstream artifact with the Stage 01–06 scripts in this bundle."
        )


def resolve_template(template: str, *, root: str | Path, patient_id: str) -> Path:
    """Resolve a schema path template such as {output_root}/{patient_id}/..."""
    text = template.format(output_root=str(Path(root).resolve()), patient_id=patient_id)
    return Path(text).expanduser().resolve()
