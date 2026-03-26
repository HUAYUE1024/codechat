"""Vector store - NumPy + JSON based local vector storage (ChromaDB-free)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .config import DEFAULT_EMBEDDING_MODEL, get_codechat_dir


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a (1D) and matrix b (2D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


class VectorStore:
    """Local vector store backed by NumPy + JSON — no external DB, no HNSW issues."""

    def __init__(self, project_root: Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.project_root = project_root.resolve()
        self.codechat_dir = get_codechat_dir(self.project_root)

        self._embeddings_path = self.codechat_dir / "embeddings.npy"
        self._metadata_path = self.codechat_dir / "metadata.json"
        self._hashes_path = self.codechat_dir / "file_hashes.json"
        self._model_name = embedding_model

        self._model = None  # lazy load

        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._ids: list[str] = []

        self._load()

    # ------------------------------------------------------------------ model

    def _get_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return self._model
        import os
        import warnings

        # Save original env vars to restore after loading
        _saved_env = {}
        _ssl_vars = ("HF_HUB_DISABLE_SSL_VERIFICATION", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")
        for k in _ssl_vars:
            _saved_env[k] = os.environ.get(k)

        # HuggingFace mirror for China users
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        # Temporarily disable SSL only for HuggingFace model download
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

        # Filter stderr: keep progress bars, suppress LOAD REPORT noise
        _orig_stderr = sys.stderr
        _filter_keys = ("LOAD REPORT", "UNEXPECTED", "Notes:", "embeddings.position", "--------+")

        class _StderrFilter:
            def write(self, text):
                if any(k in text for k in _filter_keys):
                    return
                _orig_stderr.write(text)
            def flush(self):
                _orig_stderr.flush()

        sys.stderr = _StderrFilter()
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        finally:
            sys.stderr = _orig_stderr
            # Restore original SSL env vars
            for k in _ssl_vars:
                if _saved_env[k] is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = _saved_env[k]

        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        model = self._get_model()
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    # -------------------------------------------------------------- persistence

    def _load(self) -> None:
        """Load existing data from disk."""
        if self._embeddings_path.exists() and self._metadata_path.exists():
            try:
                self._embeddings = np.load(str(self._embeddings_path))
                raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
                self._ids = raw.get("ids", [])
                self._metadata = raw.get("metadata", [])
            except Exception:
                self._embeddings = None
                self._metadata = []
                self._ids = []

    def _save(self) -> None:
        """Persist current data to disk."""
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._embeddings_path), self._embeddings)
            self._metadata_path.write_text(
                json.dumps({"ids": self._ids, "metadata": self._metadata}, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            # Empty — remove files
            self._embeddings_path.unlink(missing_ok=True)
            self._metadata_path.unlink(missing_ok=True)

    # -------------------------------------------------------------- file hashes

    def load_hashes(self) -> dict[str, str]:
        """Load stored file hashes: {rel_path: hash_string}."""
        if self._hashes_path.exists():
            try:
                return json.loads(self._hashes_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_hashes(self, hashes: dict[str, str]) -> None:
        """Save file hashes to disk."""
        self._hashes_path.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )

    @staticmethod
    def file_hash(path: Path) -> str:
        """Compute a fast hash for a file (mtime + size)."""
        try:
            st = path.stat()
            return f"{st.st_mtime_ns}:{st.st_size}"
        except OSError:
            return ""

    def get_indexed_files(self) -> set[str]:
        """Return set of file paths currently in the index."""
        return {m["file_path"] for m in self._metadata}

    def remove_by_file(self, file_path: str) -> int:
        """Remove all chunks for a specific file. Returns count removed."""
        if self._embeddings is None or not self._ids:
            return 0

        keep_indices = []
        removed = 0
        for i, meta in enumerate(self._metadata):
            if meta["file_path"] == file_path:
                removed += 1
            else:
                keep_indices.append(i)

        if removed == 0:
            return 0

        if keep_indices:
            self._embeddings = self._embeddings[keep_indices]
            self._metadata = [self._metadata[i] for i in keep_indices]
            self._ids = [self._ids[i] for i in keep_indices]
        else:
            self._embeddings = None
            self._metadata = []
            self._ids = []

        self._save()
        return removed

    # ------------------------------------------------------------------ public API

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store code chunks. Returns count of new chunks added."""
        if not chunks:
            return 0

        new_ids: list[str] = []
        new_texts: list[str] = []
        new_meta: list[dict] = []

        for chunk in chunks:
            cid = self._make_id(chunk)
            new_ids.append(cid)
            new_texts.append(chunk.content)
            new_meta.append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_index": chunk.chunk_index,
            })

        # Deduplicate against existing
        existing_set = set(self._ids)
        unique_idx = [i for i, cid in enumerate(new_ids) if cid not in existing_set]

        if not unique_idx:
            return 0

        dedup_ids = [new_ids[i] for i in unique_idx]
        dedup_texts = [new_texts[i] for i in unique_idx]
        dedup_meta = [new_meta[i] for i in unique_idx]

        # Embed
        new_vecs = self._embed(dedup_texts)

        # Merge with existing
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.vstack([self._embeddings, new_vecs])
        else:
            self._embeddings = new_vecs

        self._ids.extend(dedup_ids)
        self._metadata.extend(dedup_meta)

        self._save()
        return len(dedup_ids)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        """Search for similar code chunks using cosine similarity."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        q_vec = self._embed([text])[0]
        sims = _cosine_similarity(q_vec, self._embeddings)

        # Boost code files over doc files to avoid README dominating results
        boosts = np.ones(len(sims), dtype=np.float32)
        for i, meta in enumerate(self._metadata):
            ext = Path(meta["file_path"]).suffix.lower()
            if ext in {".md", ".rst", ".txt", ".adoc"}:
                boosts[i] = 0.4
            elif ext in {".json", ".yaml", ".yml", ".toml", ".xml"}:
                boosts[i] = 0.7
        sims = sims * boosts

        # Retrieve more candidates, then diversify by file
        candidate_k = min(n_results * 5, len(sims))
        candidate_idx = np.argsort(sims)[-candidate_k:][::-1]

        # Pick top results ensuring no more than 1 chunk per file first, then allow 2
        seen_files: dict[str, int] = {}
        picked: list[int] = []
        for idx in candidate_idx:
            fp = self._metadata[idx]["file_path"]
            count = seen_files.get(fp, 0)
            if count < 1:
                picked.append(idx)
                seen_files[fp] = count + 1
            if len(picked) >= n_results:
                break
        # If not enough, fill from remaining candidates
        if len(picked) < n_results:
            for idx in candidate_idx:
                if idx not in picked:
                    fp = self._metadata[idx]["file_path"]
                    if seen_files.get(fp, 0) < 2:
                        picked.append(idx)
                        seen_files[fp] = seen_files.get(fp, 0) + 1
                if len(picked) >= n_results:
                    break

        results: list[dict] = []
        for idx in picked:
            results.append({
                "content": "",
                "metadata": self._metadata[idx],
                "distance": float(1.0 - sims[idx]),
            })

        # Re-read source files to get chunk content for top results
        self._attach_content(results)
        return results

    def _attach_content(self, results: list[dict]) -> None:
        """Re-read source files to fill in chunk content."""
        cache: dict[str, str] = {}
        for r in results:
            meta = r["metadata"]
            fp = meta["file_path"]
            if fp not in cache:
                full_path = self.project_root / fp
                try:
                    cache[fp] = full_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    cache[fp] = ""

            lines = cache[fp].splitlines()
            start = meta["start_line"] - 1  # 0-indexed
            end = meta["end_line"]
            r["content"] = "\n".join(lines[start:end])

    def count(self) -> int:
        """Return total number of chunks."""
        if self._embeddings is None:
            return 0
        return len(self._embeddings)

    def reset(self) -> None:
        """Delete all data."""
        self._embeddings = None
        self._metadata = []
        self._ids = []
        self._save()

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        """Create a stable ID for a chunk."""
        raw = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
