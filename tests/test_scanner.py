import pytest
from pathlib import Path
from codechat.scanner import scan_files, read_file


def test_scan_files(tmp_path):
    # Create test directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    (tmp_path / "src" / "utils.js").write_text("console.log('hi')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    files = scan_files(tmp_path)
    file_names = [f.name for f in files]

    assert "main.py" in file_names
    assert "utils.js" in file_names
    assert "README.md" in file_names
    assert "image.png" not in file_names  # binary excluded


def test_scan_skips_gitignore(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("x=1", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n", encoding="utf-8")
    (tmp_path / "debug.log").write_text("log", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("# built", encoding="utf-8")

    files = scan_files(tmp_path)
    file_names = [f.name for f in files]

    assert "keep.py" in file_names
    assert "debug.log" not in file_names
    assert "out.py" not in file_names  # build/ dir skipped


def test_scan_skips_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module.exports={}", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")
    (tmp_path / ".snowcode").mkdir()
    (tmp_path / ".snowcode" / "data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")

    files = scan_files(tmp_path)
    file_names = [f.name for f in files]

    assert "main.py" in file_names
    assert "pkg.js" not in file_names
    assert "mod.cpython-312.pyc" not in file_names
    assert "data.json" not in file_names


def test_read_file(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("hello\nworld", encoding="utf-8")
    assert read_file(f) == "hello\nworld"


def test_read_file_binary(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01\x02\xff\xfe")
    assert read_file(f) is None
