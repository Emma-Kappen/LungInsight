#!/usr/bin/env python3
"""Print the file tree of the folder that contains this script.

Place the script in a folder and run it; it traces the files and subfolders
inside that folder, regardless of where you run it from. The tree is printed
and saved to results.txt (next to the script) by default.

Noise is skipped automatically: __pycache__ and .pyc files, virtual
environments (any folder containing pyvenv.cfg, plus venv/.venv/env),
.git, IDE folders, tool caches, node_modules, OS junk files, and
*.egg-info / *.dist-info folders. Skipped folders are summarised in a
single line and are never descended into, so huge environments are fast.

Usage:
    python filetree.py                    # filtered tree -> results.txt
    python filetree.py -d 2               # limit depth to 2 levels
    python filetree.py -a                 # include hidden files/folders
    python filetree.py -s                 # show file sizes
    python filetree.py -i data "*.npy"    # also skip these names/patterns
    python filetree.py -o tree.txt        # save somewhere else
    python filetree.py --no-default-ignore   # disable the built-in filters
    python filetree.py --show-ignored     # list what was skipped
"""

import argparse
import fnmatch
import sys
from pathlib import Path

BRANCH = "├── "
LAST = "└── "
PIPE = "│   "
SPACE = "    "

DEFAULT_OUTPUT = "results.txt"

# Exact names (case-insensitive) skipped by default.
DEFAULT_IGNORE_NAMES = {
    "__pycache__", ".git", ".svn", ".hg", ".idea", ".vscode", ".vs",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints",
    ".tox", ".nox", ".eggs", ".cache", "node_modules", "site-packages",
    "venv", ".venv", "env", ".env", "virtualenv",
    ".ds_store", "thumbs.db", "desktop.ini",
}

# Glob patterns (case-insensitive) skipped by default.
DEFAULT_IGNORE_PATTERNS = [
    "*.pyc", "*.pyo", "*.egg-info", "*.dist-info", "*.egg",
    "*.swp", "*.tmp", "~$*", ".coverage", "*.log",
]


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class Filter:
    """Decides which entries to skip."""

    def __init__(self, extra, use_defaults, show_hidden):
        self.show_hidden = show_hidden
        self.names = set()
        self.patterns = []
        if use_defaults:
            self.names |= DEFAULT_IGNORE_NAMES
            self.patterns += DEFAULT_IGNORE_PATTERNS
        for item in extra:
            if any(ch in item for ch in "*?["):
                self.patterns.append(item.lower())
            else:
                self.names.add(item.lower())
        self.use_defaults = use_defaults

    def reason(self, entry: Path):
        """Return a skip reason string, or None if the entry should be shown."""
        name = entry.name
        lower = name.lower()
        if self.use_defaults and entry.is_dir() and (entry / "pyvenv.cfg").exists():
            return "virtual environment"
        if lower in self.names:
            return "ignored name"
        if any(fnmatch.fnmatch(lower, p) for p in self.patterns):
            return "ignored pattern"
        if not self.show_hidden and name.startswith("."):
            return "hidden"
        return None


def list_entries(directory: Path) -> list:
    """Sorted entries: folders first, then files, case-insensitive."""
    try:
        entries = list(directory.iterdir())
    except (PermissionError, OSError):
        return []
    return sorted(entries, key=lambda e: (not e.is_dir(), e.name.lower()))


def build_tree(directory, prefix, depth, opts, flt, state, lines):
    if opts.max_depth is not None and depth >= opts.max_depth:
        return
    shown = []
    for entry in list_entries(directory):
        why = flt.reason(entry)
        if why:
            state["skipped"] += 1
            state["skipped_log"].append(
                f"{entry.relative_to(state['root']).as_posix()}"
                f"{'/' if entry.is_dir() else ''}  ({why})")
        else:
            shown.append(entry)

    for index, entry in enumerate(shown):
        is_last = index == len(shown) - 1
        connector = LAST if is_last else BRANCH
        is_dir = entry.is_dir() and not entry.is_symlink()

        label = entry.name + ("/" if is_dir else "")
        if entry.is_symlink():
            try:
                label += f" -> {entry.readlink()}"
            except OSError:
                pass
        elif opts.size and not is_dir:
            try:
                label += f"  ({human_size(entry.stat().st_size)})"
            except OSError:
                pass
        if entry.resolve() == state["script"]:
            label += "  <- this script"
        lines.append(prefix + connector + label)

        if is_dir:
            state["dirs"] += 1
            build_tree(entry, prefix + (SPACE if is_last else PIPE), depth + 1,
                       opts, flt, state, lines)
        else:
            state["files"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print and save the file tree of the folder containing "
                    "this script, skipping caches, virtual environments, "
                    "and other extraneous files.")
    parser.add_argument("-d", "--max-depth", type=int, default=None,
                        help="maximum depth to descend (default: unlimited)")
    parser.add_argument("-a", "--all", action="store_true",
                        help="include hidden files and folders")
    parser.add_argument("-s", "--size", action="store_true",
                        help="show file sizes")
    parser.add_argument("-i", "--ignore", nargs="*", default=[],
                        help="extra names or glob patterns to skip")
    parser.add_argument("--no-default-ignore", action="store_true",
                        help="disable the built-in cache/venv/junk filters")
    parser.add_argument("--show-ignored", action="store_true",
                        help="append a list of everything that was skipped")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help=f"output file (default: {DEFAULT_OUTPUT} next to "
                             "the script)")
    opts = parser.parse_args()

    script_path = Path(__file__).resolve()
    root = script_path.parent
    out_path = opts.output or (root / DEFAULT_OUTPUT)

    extra = list(opts.ignore)
    # Never list the results file inside its own tree.
    extra.append(out_path.name)

    flt = Filter(extra, not opts.no_default_ignore, opts.all)
    state = {"dirs": 0, "files": 0, "skipped": 0, "skipped_log": [],
             "root": root, "script": script_path}

    lines = [f"{root.name}/"]
    build_tree(root, "", 0, opts, flt, state, lines)
    lines.append("")
    lines.append(f"{state['dirs']} directories, {state['files']} files "
                 f"({state['skipped']} entries skipped)")
    if opts.show_ignored and state["skipped_log"]:
        lines.append("")
        lines.append("Skipped:")
        lines.extend(f"  {item}" for item in state["skipped_log"])

    output = "\n".join(lines)

    try:
        out_path.write_text(output + "\n", encoding="utf-8")
        saved = f"Saved to {out_path}"
    except OSError as exc:
        saved = f"Could not save to {out_path}: {exc}"

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print(output)
    print(f"\n{saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())