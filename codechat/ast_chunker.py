"""AST-aware code chunking using Tree-sitter.

Parses code into syntax trees and extracts top-level definitions
(functions, classes, methods, type definitions) as semantic chunks.
Falls back gracefully when tree-sitter doesn't support the language.
"""

from __future__ import annotations

from dataclasses import dataclass

# Extension -> tree-sitter language name
_EXT_TO_LANG: dict[str, str] = {
    # Python
    ".py": "python",
    # JavaScript / TypeScript
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    # Go
    ".go": "go",
    # Rust
    ".rs": "rust",
    # Java
    ".java": "java",
    # C / C++
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # C#
    ".cs": "c_sharp",
    # Kotlin
    ".kt": "kotlin",
    # Swift
    ".swift": "swift",
    # Lua
    ".lua": "lua",
    # Bash
    ".sh": "bash", ".bash": "bash",
    # SQL
    ".sql": "sql",
    # R
    ".r": "r", ".R": "r",
    # HTML / CSS (limited)
    ".html": "html", ".htm": "html",
    ".css": "css",
}

# Node types that represent top-level definitions per language
_DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition",
                    "arrow_function", "variable_declarator",
                    "export_statement", "lexical_declaration"},
    "typescript": {"function_declaration", "class_declaration", "method_definition",
                    "arrow_function", "interface_declaration", "type_alias_declaration",
                    "enum_declaration", "variable_declarator",
                    "export_statement", "lexical_declaration"},
    "tsx": {"function_declaration", "class_declaration", "method_definition",
            "arrow_function", "interface_declaration", "type_alias_declaration",
            "enum_declaration", "export_statement"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item",
             "trait_item", "mod_item", "type_item", "use_declaration"},
    "java": {"method_declaration", "constructor_declaration",
             "class_declaration", "interface_declaration", "enum_declaration",
             "field_declaration"},
    "c": {"function_definition", "declaration", "struct_specifier",
          "enum_specifier", "type_definition"},
    "cpp": {"function_definition", "declaration", "class_specifier",
            "struct_specifier", "enum_specifier", "namespace_definition",
            "template_declaration", "type_definition"},
    "ruby": {"method", "class", "module", "singleton_method"},
    "php": {"function_definition", "method_declaration", "class_declaration",
            "interface_declaration", "trait_declaration"},
    "c_sharp": {"method_declaration", "constructor_declaration",
                "class_declaration", "interface_declaration", "struct_declaration",
                "enum_declaration", "property_declaration"},
    "kotlin": {"function_declaration", "class_declaration",
               "object_declaration", "property_declaration"},
    "swift": {"function_declaration", "class_declaration",
              "struct_declaration", "enum_declaration", "protocol_declaration"},
    "lua": {"function_declaration", "local_function"},
    "bash": {"function_definition", "command"},
    "sql": {"create_function", "create_table", "create_view"},
    "r": {"function_definition"},
}

# Cache for loaded languages
_lang_cache: dict[str, object] = {}


def _get_language(lang_name: str):
    """Get a tree-sitter Language object (cached)."""
    if lang_name in _lang_cache:
        return _lang_cache[lang_name]
    try:
        import tree_sitter_languages
        lang = tree_sitter_languages.get_language(lang_name)
        _lang_cache[lang_name] = lang
        return lang
    except Exception:
        _lang_cache[lang_name] = None
        return None


# Pre-import tree_sitter_languages with SSL bypass
def _safe_import_tree_sitter():
    """Attempt to import tree-sitter, avoiding SSL issues."""
    try:
        import tree_sitter
        import tree_sitter_languages
        return tree_sitter, tree_sitter_languages
    except ImportError:
        return None, None

_safe_import_tree_sitter()


def _get_parser(lang_name: str):
    """Create a tree-sitter Parser for the given language."""
    import tree_sitter
    lang = _get_language(lang_name)
    if lang is None:
        return None
    try:
        parser = tree_sitter.Parser(lang)
        return parser
    except Exception:
        return None


def get_language_for_file(file_path: str) -> str | None:
    """Get tree-sitter language name for a file extension."""
    import os
    ext = os.path.splitext(file_path)[1].lower()
    return _EXT_TO_LANG.get(ext)


def ast_split_definitions(
    file_path: str,
    content: str,
    chunk_size: int = 1500,
    min_lines: int = 5,
) -> list[tuple[str, int, int]]:
    """
    Split code by top-level definitions using Tree-sitter AST.

    Returns list of (content, start_line, end_line) tuples.
    Falls back to empty list if parsing fails (caller should use regex fallback).
    """
    lang_name = get_language_for_file(file_path)
    if lang_name is None:
        return []

    parser = _get_parser(lang_name)
    if parser is None:
        return []

    source_bytes = content.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception:
        return []

    def_types = _DEFINITION_TYPES.get(lang_name, set())
    if not def_types:
        return []

    lines = content.splitlines()
    total_lines = len(lines)

    # Walk the tree to find top-level definition nodes
    definitions: list[tuple[int, int]] = []  # (start_line, end_line) 1-indexed
    module_level_chunks: list[tuple[int, int]] = []

    def _collect_definitions(node, depth=0):
        """Recursively collect definition nodes at the top level."""
        if node.type in def_types:
            start = node.start_point[0] + 1  # 1-indexed
            end = node.end_point[0] + 1
            if end - start + 1 >= min_lines:
                definitions.append((start, end))
            return  # Don't recurse into definitions

        # For top-level scope, recurse into children
        if depth <= 2:
            for child in node.children:
                _collect_definitions(child, depth + 1)

    _collect_definitions(tree.root_node)

    if not definitions:
        return []

    # Sort by start line
    definitions.sort()

    # Extract module-level code (imports, constants, etc.) between definitions
    chunks: list[tuple[str, int, int]] = []
    prev_end = 0

    for start, end in definitions:
        # Module-level code before this definition
        if start > prev_end + 1:
            module_start = prev_end + 1
            module_end = start - 1
            module_text = "\n".join(lines[module_start - 1:module_end])
            if module_text.strip():
                chunks.append((module_text, module_start, module_end))

        # The definition itself
        def_text = "\n".join(lines[start - 1:end])
        if def_text.strip():
            chunks.append((def_text, start, end))
        prev_end = end

    # Trailing module-level code
    if prev_end < total_lines:
        tail_text = "\n".join(lines[prev_end:])
        if tail_text.strip():
            chunks.append((tail_text, prev_end + 1, total_lines))

    # Merge tiny adjacent chunks
    merged = _merge_tiny(chunks, min_lines)

    # Split oversized chunks
    final: list[tuple[str, int, int]] = []
    for text, s, e in merged:
        if len(text) <= chunk_size * 1.5:
            final.append((text, s, e))
        else:
            # Split oversized into line-based chunks
            sub_lines = text.splitlines()
            overlap_lines = 3
            offset = 0
            while offset < len(sub_lines):
                char_count = 0
                end_offset = offset
                while end_offset < len(sub_lines) and char_count < chunk_size:
                    char_count += len(sub_lines[end_offset]) + 1
                    end_offset += 1
                if end_offset <= offset:
                    end_offset = offset + 1
                sub_text = "\n".join(sub_lines[offset:end_offset])
                if sub_text.strip():
                    final.append((sub_text, s + offset, min(e, s + end_offset - 1)))
                offset = end_offset - overlap_lines
                if offset <= end_offset - overlap_lines:
                    offset = end_offset

    return final


def _merge_tiny(
    chunks: list[tuple[str, int, int]],
    min_lines: int,
) -> list[tuple[str, int, int]]:
    """Merge tiny chunks (< min_lines) with their neighbors."""
    if not chunks:
        return []

    merged: list[tuple[str, int, int]] = []
    buf_text = ""
    buf_start = 0
    buf_end = 0

    for text, start, end in chunks:
        lines_count = end - start + 1
        if not buf_text:
            buf_text = text
            buf_start = start
            buf_end = end
            continue

        buf_lines = buf_end - buf_start + 1
        if buf_lines < min_lines:
            # Merge with current
            buf_text += "\n" + text
            buf_end = end
        else:
            merged.append((buf_text, buf_start, buf_end))
            buf_text = text
            buf_start = start
            buf_end = end

    if buf_text:
        buf_lines = buf_end - buf_start + 1
        if buf_lines < min_lines and merged:
            prev_text, prev_start, _ = merged[-1]
            merged[-1] = (prev_text + "\n" + buf_text, prev_start, buf_end)
        else:
            merged.append((buf_text, buf_start, buf_end))

    return merged
