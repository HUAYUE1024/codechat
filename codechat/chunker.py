"""Code chunking - split code into semantic chunks for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


@dataclass
class Chunk:
    """A piece of code with metadata."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_index: int


def _split_by_lines(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split text into overlapping line-based chunks. Returns (content, start_line, end_line)."""
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0

    while start < len(lines):
        # Approximate chunk by character count
        end = start
        char_count = 0
        while end < len(lines) and char_count < chunk_size:
            char_count += len(lines[end]) + 1
            end += 1

        if end <= start:
            end = start + 1

        content = "\n".join(lines[start:end])
        if content.strip():
            chunks.append((content, start + 1, end))  # 1-indexed lines

        # Move forward with overlap
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _split_by_functions(text: str) -> list[tuple[str, int, int]]:
    """
    Attempt to split code by function/class definitions.
    Falls back to line-based splitting for individual functions.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return [(text, 1, len(lines))]

    # Patterns for common function/class definitions
    fn_pattern = re.compile(
        r"^(def |class |func |fn |function |pub fn |async def |export (function|class|default) "
        r"|public |private |protected |void |int |string |bool )",
        re.MULTILINE,
    )

    boundaries: list[int] = [0]
    for i, line in enumerate(lines):
        if fn_pattern.match(line.strip()):
            # Only add if there's meaningful content before this boundary
            if i > 0 and any(l.strip() for l in lines[boundaries[-1]:i]):
                boundaries.append(i)

    if len(boundaries) <= 1:
        return [(text, 1, len(lines))]

    chunks: list[tuple[str, int, int]] = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        content = "\n".join(lines[start:end])
        if content.strip():
            chunks.append((content, start + 1, end))

    return chunks


def chunk_file(
    file_path: str,
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a file's content into chunks with metadata."""
    lines = content.splitlines()
    total_lines = len(lines)

    # For small files, keep as single chunk
    if len(content) <= chunk_size:
        if content.strip():
            return [Chunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                chunk_index=0,
            )]
        return []

    # Try function-level splitting first, then fall back to line-based
    fn_chunks = _split_by_functions(content)

    # If function splitting produced large chunks, further split them
    final_chunks: list[Chunk] = []
    for sub_content, start_line, end_line in fn_chunks:
        if len(sub_content) <= chunk_size * 1.5:
            if sub_content.strip():
                final_chunks.append(Chunk(
                    content=sub_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_index=len(final_chunks),
                ))
        else:
            sub_lines = sub_content.splitlines()
            for sub_s, sub_ss, sub_es in _split_by_lines(sub_content, chunk_size, overlap):
                if sub_s.strip():
                    final_chunks.append(Chunk(
                        content=sub_s,
                        file_path=file_path,
                        start_line=start_line + sub_ss - 1,
                        end_line=start_line + sub_es - 1,
                        chunk_index=len(final_chunks),
                    ))

    return final_chunks
