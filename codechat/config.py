"""Configuration and constants."""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists in the current directory or its parents
def _init_dotenv():
    # Find project root similar to cli.py but simpler
    markers = [".git", "pyproject.toml", "package.json", ".snowcode", ".env"]
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return
        if any((parent / m).exists() for m in markers):
            # If we found a project root but no .env, stop looking
            return

_init_dotenv()

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
    ".idea", ".vscode", ".snowcode",
}

# File size limit (bytes) - skip files larger than this
MAX_FILE_SIZE = 1_000_000  # 1MB

# Default chunk settings
DEFAULT_CHUNK_SIZE = 1500       # characters per chunk
DEFAULT_CHUNK_OVERLAP = 5       # lines of overlap between chunks

# Embedding model — all-mpnet-base-v2 for better code retrieval quality
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Rerank model — cross-encoder/ms-marco-MiniLM-L-6-v2 for lightweight reranking
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Config dir inside project
CODECHAT_DIR = ".snowcode"


def get_snowcode_dir(project_root: Path) -> Path:
    """Get or create the .snowcode directory inside a project."""
    d = project_root / CODECHAT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_config(project_root: Path) -> dict:
    """Load snowcode config from project, or return defaults."""
    config_path = get_snowcode_dir(project_root) / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def save_config(project_root: Path, config: dict) -> None:
    """Save snowcode config."""
    config_path = get_snowcode_dir(project_root) / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "default_model": "gpt-4o-mini",
    "api_base_url": "https://api.openai.com/v1",
    "thinking_enabled": False,
}


def get_llm_config_from_file(project_root: Path) -> dict:
    """Get LLM config from project config file, falling back to defaults."""
    config = load_config(project_root)
    llm_config = config.get("llm", {})
    result = DEFAULT_LLM_CONFIG.copy()
    result.update(llm_config)
    return result


def save_llm_config(project_root: Path, llm_config: dict) -> None:
    """Save LLM config to project config file."""
    config = load_config(project_root)
    config["llm"] = llm_config
    save_config(project_root, config)
