"""Console policy shared by the imaging stages."""
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from typing import Any

_verbose = os.environ.get("LOG_LEVEL", "INFO").strip().upper() == "DEBUG"


def add_verbose_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed stage diagnostics (also enabled by LOG_LEVEL=DEBUG).",
    )


def configure_console(verbose: bool = False) -> None:
    global _verbose
    _verbose = bool(verbose) or os.environ.get("LOG_LEVEL", "INFO").strip().upper() == "DEBUG"


def pipeline_print(*values: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
    text = sep.join(str(value) for value in values)
    if _verbose or _is_operational_message(text):
        print(text, end=end, file=file, flush=flush)


def stage_complete(stage_number: int, stage_name: str) -> None:
    if not 1 <= stage_number <= 7:
        raise ValueError("stage_number must be between 1 and 7.")
    pipeline_print(f"[{stage_number:02d}/07] Stage {stage_name} Complete")


def _is_operational_message(text: str) -> bool:
    normalized = text.strip()
    lowered = normalized.lower()
    return (
        normalized.startswith("[") and "/07]" in normalized and "complete" in lowered
    ) or any(token in lowered for token in (
        "warning", "error", "failed", "failure", "blocked", "final summary", "status': 'completed",
    ))
