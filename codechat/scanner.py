"""File scanning - discover and read code files from a project."""

import fnmatch
import os
from pathlib import Path

import pathspec

from .config import CODE_EXTENSIONS, DOC_EXTENSIONS, CONFIG_EXTENSIONS, MAX_FILE_SIZE, SKIP_DIRS

ALL_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS


def _load_ignore_patterns(project_root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore and .snowcodeignore patterns if available."""
    lines = []
    
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        lines.extend(gitignore.read_text(encoding="utf-8", errors="ignore").splitlines())
        
    snowcodeignore = project_root / ".snowcodeignore"
    if snowcodeignore.exists():
        lines.extend(snowcodeignore.read_text(encoding="utf-8", errors="ignore").splitlines())
        
    # Clean up lines
    valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
        
    if valid_lines:
        return pathspec.PathSpec.from_lines("gitignore", valid_lines)
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

    Uses os.walk with directory pruning for efficient traversal.
    Respects .gitignore, skips binary/build directories.
    """
    project_root = project_root.resolve()
    ignore_patterns = _load_ignore_patterns(project_root)
    extensions = ALL_EXTENSIONS | (extra_extensions or set())

    files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(str(project_root)):
        # Prune directories in-place (os.walk supports this)
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(d) and d != ".snowcode"
        ]

        rel_dir = Path(dirpath).relative_to(project_root)

        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = rel_dir / fname

            # Check ignore patterns
            if ignore_patterns and ignore_patterns.match_file(str(rel)):
                continue

            # Check extension
            if fpath.suffix.lower() not in extensions:
                continue

            # Check file size
            try:
                size = fpath.stat().st_size
                if size == 0 or size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            files.append(fpath)

    return sorted(files)


def read_file(path: Path) -> str | None:
    """Read a text file, handling potential encoding issues."""
    # First try fast binary check
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return None
    except OSError:
        return None

    # Try common encodings
    # Omitted latin-1/cp1252 as they silently mask encoding errors leading to garbage text
    encodings = ["utf-8", "gbk", "gb2312"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="strict")
        except UnicodeDecodeError:
            continue
            
    # Fallback to utf-8 with replacement if all strict reads fail
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
