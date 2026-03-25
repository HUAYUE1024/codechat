"""File scanning - discover and read code files from a project."""

import fnmatch
from pathlib import Path

import pathspec

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, CONFIG_EXTENSIONS, MAX_FILE_SIZE, SKIP_DIRS

ALL_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS


def _load_gitignore(project_root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns if available."""
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


def _should_skip_dir(dirname: str) -> bool:
    """Check if directory should be skipped."""
    if dirname.startswith(".") and dirname not in {".github", ".gitlab"}:
        return True
    for pattern in SKIP_DIRS:
        if fnmatch.fnmatch(dirname, pattern):
            return True
    return False


def scan_files(project_root: Path, extra_extensions: set[str] | None = None) -> list[Path]:
    """
    Scan a project directory and return all code/doc files.

    Respects .gitignore, skips binary/build directories.
    """
    project_root = project_root.resolve()
    gitignore = _load_gitignore(project_root)
    extensions = ALL_EXTENSIONS | (extra_extensions or set())

    files: list[Path] = []

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(project_root)

        # Skip directories in path
        if any(_should_skip_dir(part) for part in rel.parts):
            continue

        # Skip .codechat dir
        if ".codechat" in rel.parts:
            continue

        # Check gitignore
        if gitignore and gitignore.match_file(str(rel)):
            continue

        # Check extension
        if path.suffix.lower() not in extensions:
            continue

        # Check file size
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue

        files.append(path)

    return sorted(files)


def read_file(path: Path) -> str | None:
    """Read a file, returning None if it's binary or unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
        return text
    except (UnicodeDecodeError, OSError):
        return None
