"""Vector store - NumPy + JSON based local vector storage (ChromaDB-free)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import numpy as np

from .chunker import Chunk
from .config import DEFAULT_EMBEDDING_MODEL, get_codechat_dir


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a (1D) and matrix b (2D)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    text = text.lower()
    # Extract words, camelCase/snake_case parts
    words = re.findall(r'[a-zA-Z0-9]+', text)
    return [w for w in words if len(w) > 1]


class BM25:
    """Simple BM25 implementation for keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[dict[str, int]] = []
        self.df: dict[str, int] = Counter()
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.corpus_size = 0
        
    def fit(self, corpus: list[str]):
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return
            
        self.doc_freqs = []
        self.doc_len = []
        
        for doc in corpus:
            tokens = _tokenize(doc)
            self.doc_len.append(len(tokens))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for token in freqs:
                self.df[token] += 1
                
        self.avgdl = sum(self.doc_len) / self.corpus_size
        
    def score(self, query: str) -> np.ndarray:
        if self.corpus_size == 0:
            return np.array([])
            
        q_tokens = _tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        
        for token in q_tokens:
            if token not in self.df:
                continue
                
            idf = math.log((self.corpus_size - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1.0)
            
            for i, doc_freq in enumerate(self.doc_freqs):
                if token in doc_freq:
                    tf = doc_freq[token]
                    doc_len = self.doc_len[i]
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                    scores[i] += idf * (numerator / denominator)
                    
        return scores
        
    def to_dict(self) -> dict:
        return {
            "df": dict(self.df),
            "doc_len": self.doc_len,
            "doc_freqs": [dict(df) for df in self.doc_freqs],
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl
        }
        
    def from_dict(self, data: dict):
        self.df = Counter(data.get("df", {}))
        self.doc_len = data.get("doc_len", [])
        self.doc_freqs = [Counter(df) for df in data.get("doc_freqs", [])]
        self.corpus_size = data.get("corpus_size", 0)
        self.avgdl = data.get("avgdl", 0.0)


class VectorStore:
    """Local vector store backed by NumPy + JSON — no external DB, no HNSW issues."""

    def __init__(self, project_root: Path, embedding_model: str | None = None):
        self.project_root = project_root.resolve()
        self.codechat_dir = get_codechat_dir(self.project_root)

        self._embeddings_path = self.codechat_dir / "embeddings.npy"
        self._metadata_path = self.codechat_dir / "metadata.json"
        self._hashes_path = self.codechat_dir / "file_hashes.json"
        self._bm25_path = self.codechat_dir / "bm25.json"
        self._config_path = self.codechat_dir / "config.json"
        
        # Determine the model to use:
        # 1. Passed explicitly (via CLI arg)
        # 2. Loaded from config.json (previous ingest)
        # 3. Default fallback
        if embedding_model:
            self._model_name = embedding_model
        else:
            if self._config_path.exists():
                try:
                    conf = json.loads(self._config_path.read_text(encoding="utf-8"))
                    self._model_name = conf.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                except Exception:
                    self._model_name = DEFAULT_EMBEDDING_MODEL
            else:
                self._model_name = DEFAULT_EMBEDDING_MODEL

        self._model = None  # lazy load

        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict] = []
        self._ids: list[str] = []
        self._bm25 = BM25()
        self._texts: list[str] = []  # needed for BM25 fit

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
        _ssl_vars = ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")
        for k in _ssl_vars:
            _saved_env[k] = os.environ.get(k)

        # HuggingFace mirror for China users
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            
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
                self._texts = raw.get("texts", [])
                
                if self._bm25_path.exists():
                    bm25_data = json.loads(self._bm25_path.read_text(encoding="utf-8"))
                    self._bm25.from_dict(bm25_data)
                elif self._texts:
                    self._bm25.fit(self._texts)
            except Exception:
                self._embeddings = None
                self._metadata = []
                self._ids = []
                self._texts = []
                self._bm25 = BM25()

    def _save(self) -> None:
        """Persist current data to disk."""
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._embeddings_path), self._embeddings)
            self._metadata_path.write_text(
                json.dumps({
                    "ids": self._ids, 
                    "metadata": self._metadata,
                    "texts": self._texts
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            self._bm25_path.write_text(
                json.dumps(self._bm25.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            # Empty — remove files
            self._embeddings_path.unlink(missing_ok=True)
            self._metadata_path.unlink(missing_ok=True)
            self._bm25_path.unlink(missing_ok=True)

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
            if self._texts:
                self._texts = [self._texts[i] for i in keep_indices]
                self._bm25.fit(self._texts)
        else:
            self._embeddings = None
            self._metadata = []
            self._ids = []
            self._texts = []
            self._bm25 = BM25()

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
        self._texts.extend(dedup_texts)
        self._bm25.fit(self._texts)

        self._save()
        return len(dedup_ids)

    def query(self, text: str, n_results: int = 5, hybrid_alpha: float = 0.5) -> list[dict]:
        """Search for similar code chunks using hybrid search (Vector + BM25)."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Vector search
        q_vec = self._embed([text])[0]
        
        # Guard against dimension mismatch
        if self._embeddings.shape[1] != q_vec.shape[0]:
            print(f"\n[Warning] Embedding dimension mismatch (Index: {self._embeddings.shape[1]}, Query: {q_vec.shape[0]}). "
                  f"Please run `codechat ingest --reset` to rebuild the index.", file=sys.stderr)
            return []
            
        vec_sims = _cosine_similarity(q_vec, self._embeddings)
        
        # BM25 keyword search
        bm25_scores = self._bm25.score(text)
        if len(bm25_scores) > 0 and bm25_scores.max() > 0:
            # Normalize BM25 scores to [0, 1]
            bm25_sims = bm25_scores / bm25_scores.max()
        else:
            bm25_sims = np.zeros_like(vec_sims)
            
        # Combine scores
        sims = (hybrid_alpha * vec_sims) + ((1 - hybrid_alpha) * bm25_sims)

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
        self._texts = []
        self._bm25 = BM25()
        self._save()

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        """Create a stable ID for a chunk."""
        raw = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
