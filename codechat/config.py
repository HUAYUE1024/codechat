"""Configuration and constants."""

import json
import os
from pathlib import Path

# Code file extensions to ingest
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".m", ".mm", ".scala", ".clj", ".ex", ".exs", ".hs", ".ml", ".r",
    ".lua", ".pl", ".pm", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".graphql", ".gql", ".proto", ".thrift",
}

# Documentation extensions
DOC_EXTENSIONS = {
    ".md", ".rst", ".txt", ".adoc",
}

# Config file extensions
CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".env", ".properties",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".eggs", "*.egg-info", ".next", ".nuxt", "target", "vendor",
    ".idea", ".vscode", ".codechat",
}

# File size limit (bytes) - skip files larger than this
MAX_FILE_SIZE = 1_000_000  # 1MB

# Default chunk settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Config dir inside project
CODECHAT_DIR = ".codechat"


def get_codechat_dir(project_root: Path) -> Path:
    """Get or create the .codechat directory inside a project."""
    d = project_root / CODECHAT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config(project_root: Path) -> dict:
    """Load codechat config from project, or return defaults."""
    config_path = get_codechat_dir(project_root) / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def save_config(project_root: Path, config: dict) -> None:
    """Save codechat config."""
    config_path = get_codechat_dir(project_root) / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
