"""CLI - Command-line interface for codechat."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.syntax import Syntax

from . import __version__
from .chunker import chunk_file
from .config import get_codechat_dir, load_config, save_config, DEFAULT_EMBEDDING_MODEL
from .rag import answer_question, answer_question_stream, _get_llm_config
from .scanner import scan_files, read_file
from .agent import CodeAgent
from .skills import run_skill, run_skill_stream, SKILL_QUERIES
from .store import VectorStore

console = Console(legacy_windows=False)


def _find_project_root() -> Path:
    """Find project root by looking for .git, pyproject.toml, package.json, etc."""
    markers = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile"]
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if any((parent / m).exists() for m in markers):
            return parent
    return cwd


@click.group()
@click.version_option(__version__, prog_name="codechat")
def cli():
    """codechat - Chat with your codebase using local RAG."""


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--reset", is_flag=True, help="Full rebuild: clear and re-index everything")
@click.option("--chunk-size", default=None, type=int, help="Chunk size in characters")
@click.option("--chunk-overlap", default=None, type=int, help="Chunk overlap in lines")
@click.option("--model", "-m", default=None, help="Embedding model name")
def ingest(
    path: str | None,
    reset: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    model: str | None,
):
    """Ingest a project — full or incremental (only changed files)."""
    from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

    root = Path(path).resolve() if path else _find_project_root()
    emb_model = model or DEFAULT_EMBEDDING_MODEL
    c_size = chunk_size or DEFAULT_CHUNK_SIZE
    c_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    console.print(f"\n[bold cyan]codechat[/] v{__version__}")
    console.print(f"  Project: [green]{root}[/]")
    console.print(f"  Model:   [dim]{emb_model}[/]\n")

    # Initialize vector store
    with console.status("[bold green]Initializing embedding model...", spinner="dots"):
        store = VectorStore(root, embedding_model=emb_model)

    # Full rebuild
    if reset:
        store.reset()
        old_hashes: dict[str, str] = {}
        console.print("  [yellow]Index reset — full rebuild.[/]\n")
    else:
        old_hashes = store.load_hashes()

    # Scan files and compute hashes
    with console.status("[bold green]Scanning project files...", spinner="dots"):
        files = scan_files(root)

    if not files:
        console.print("[red]No code files found in the project.[/]")
        return

    # Compute current hashes and diff
    new_hashes: dict[str, str] = {}
    files_to_process: list[Path] = []
    unchanged_count = 0

    for f in files:
        rel = str(f.relative_to(root))
        h = store.file_hash(f)
        if not h:
            continue
        new_hashes[rel] = h
        if old_hashes.get(rel) == h:
            unchanged_count += 1
        else:
            files_to_process.append(f)

    # Find deleted files
    current_rels = set(new_hashes.keys())
    deleted_rels = set(old_hashes.keys()) - current_rels

    # Remove chunks for deleted and changed files
    removed_count = 0
    for rel in deleted_rels:
        removed_count += store.remove_by_file(rel)
    for f in files_to_process:
        rel = str(f.relative_to(root))
        removed_count += store.remove_by_file(rel)

    # Report diff
    if old_hashes:
        console.print(f"  Files: [bold]{len(files)}[/] total, "
                       f"[green]{unchanged_count}[/] unchanged, "
                       f"[yellow]{len(files_to_process)}[/] changed/new, "
                       f"[red]{len(deleted_rels)}[/] deleted")
        if not files_to_process and not deleted_rels:
            console.print(f"  [green]Index is up to date.[/]")
            # Still save hashes in case format changed
            store.save_hashes(new_hashes)
            save_config(root, {
                "embedding_model": emb_model,
                "chunk_size": c_size,
                "chunk_overlap": c_overlap,
                "last_ingest": time.time(),
                "files_count": len(files),
                "chunks_count": store.count(),
            })
            return
        console.print()
    else:
        console.print(f"  Found [bold]{len(files)}[/] files (first ingest)\n")

    # Chunk only changed/new files
    new_chunks = []
    if files_to_process:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking...", total=len(files_to_process))
            for f in files_to_process:
                rel = str(f.relative_to(root))
                content = read_file(f)
                if content:
                    chunks = chunk_file(rel, content, c_size, c_overlap)
                    new_chunks.extend(chunks)
                progress.advance(task)

    if new_chunks:
        console.print(f"  Generated [bold]{len(new_chunks)}[/] chunks from {len(files_to_process)} files\n")

        # Embed and add
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Embedding & indexing...", total=1)
            store.add_chunks(new_chunks)
            progress.advance(task)

    # Save hashes and config
    store.save_hashes(new_hashes)
    save_config(root, {
        "embedding_model": emb_model,
        "chunk_size": c_size,
        "chunk_overlap": c_overlap,
        "last_ingest": time.time(),
        "files_count": len(files),
        "chunks_count": store.count(),
    })

    # Summary
    table = Table(title="Ingestion Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Files scanned", str(len(files)))
    table.add_row("Files processed", str(len(files_to_process)))
    table.add_row("Files unchanged (skipped)", str(unchanged_count))
    if deleted_rels:
        table.add_row("Files deleted", str(len(deleted_rels)))
    table.add_row("New chunks", str(len(new_chunks)))
    table.add_row("Total chunks in store", str(store.count()))
    console.print(table)
    console.print("\n[bold green]Done![/] You can now run [cyan]codechat ask \"...\"[/]\n")


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--context", "-k", default=5, type=int, help="Number of context chunks to retrieve")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files in output")
@click.option("--show-thinking", is_flag=True, help="Show LLM thinking/reasoning process")
def ask(
    question: tuple[str, ...],
    path: str | None,
    context: int,
    model: str | None,
    show_sources: bool,
    show_thinking: bool,
):
    """Ask a question about the codebase."""
    root = Path(path).resolve() if path else _find_project_root()
    q = " ".join(question)

    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]codechat ingest[/] first.[/]")
        return

    api_key, base_url, llm_model, thinking = _get_llm_config(model)

    if api_key:
        # Streaming mode with thinking support
        console.print(f"\n  [dim]LLM: {llm_model} @ {base_url}[/]")
        console.print()
        if thinking and show_thinking:
            console.print("[dim]=== Thinking ===[/]")

        think_buf: list[str] = []
        answer_buf: list[str] = []
        in_answer = False

        def on_think(token: str):
            nonlocal in_answer
            if not in_answer and token.strip():
                think_buf.append(token)
                if show_thinking:
                    console.print(f"[dim]{token}[/]", end="")

        def on_answer(token: str):
            nonlocal in_answer
            if not in_answer:
                in_answer = True
                if thinking and show_thinking and think_buf:
                    console.print()
                    console.print("[dim]=== Answer ===[/]")
            answer_buf.append(token)
            console.print(token, end="")

        result = answer_question_stream(
            store, q, n_context=context, model=model,
            on_think=on_think, on_answer=on_answer,
        )
        console.print()
    else:
        # No LLM - use non-streaming fallback
        with console.status("[bold cyan]Searching...", spinner="dots"):
            result = answer_question(store, q, n_context=context, model=model)
        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold]Answer[/]",
            border_style="green",
            padding=(1, 2),
        ))

    # Show sources
    if show_sources or not api_key:
        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
    console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.option("--context", "-k", default=5, type=int, help="Number of context chunks")
@click.option("--model", "-m", default=None, help="LLM model to use")
def chat(path: str | None, context: int, model: str | None):
    """Start an interactive chat session with the codebase."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

    root = Path(path).resolve() if path else _find_project_root()
    store = VectorStore(root)

    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]codechat ingest[/] first.[/]")
        return

    console.print(f"\n[bold cyan]codechat[/] v{__version__} - Interactive Mode")
    console.print(f"  Project: [green]{root}[/]")
    console.print(f"  Chunks:  [dim]{store.count()}[/]")
    console.print(f"  Type your questions, or [dim]/quit[/] to exit.\n")

    style = Style.from_dict({
        "prompt": "ansicyan bold",
    })
    history = InMemoryHistory()
    session: PromptSession = PromptSession(history=history, style=style, auto_suggest=AutoSuggestFromHistory())

    while True:
        try:
            q = session.prompt([("class:prompt", "codechat> ")])
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/]")
            break

        q = q.strip()
        if not q:
            continue

        if q in {"/quit", "/exit", "/q"}:
            console.print("[dim]Bye![/]")
            break

        if q == "/reset":
            store.reset()
            console.print("[yellow]Index reset.[/]")
            continue

        if q == "/stats":
            console.print(f"  Chunks: [bold]{store.count()}[/]")
            continue

        if q == "/help":
            console.print("  [cyan]/quit[/]    Exit")
            console.print("  [cyan]/reset[/]   Clear the index")
            console.print("  [cyan]/stats[/]   Show index stats")
            console.print("  [cyan]/help[/]    Show this help")
            continue

        with console.status("[bold cyan]Thinking...", spinner="dots"):
            api_key, _, _, thinking = _get_llm_config(model)

        if api_key:
            # Streaming mode
            if thinking:
                console.print("[dim]=== Thinking ===[/]")
            is_answer = False

            def _on_think(t: str):
                nonlocal is_answer
                if not is_answer and t.strip():
                    console.print(f"[dim]{t}[/]", end="")

            def _on_answer(t: str):
                nonlocal is_answer
                if not is_answer:
                    is_answer = True
                    console.print()
                    console.print("[dim]=== Answer ===[/]")
                console.print(t, end="")

            result = answer_question_stream(
                store, q, n_context=context, model=model,
                on_think=_on_think, on_answer=_on_answer,
            )
            console.print()
        else:
            result = answer_question(store, q, n_context=context, model=model)
            console.print()
            console.print(Markdown(result["answer"]))

        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
        console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
def status(path: str | None):
    """Show the current index status."""
    root = Path(path).resolve() if path else _find_project_root()
    config = load_config(root)

    if not config:
        console.print("[dim]No index found. Run [cyan]codechat ingest[/] to get started.[/]")
        return

    store = VectorStore(root)

    table = Table(title="codechat Status", show_header=False)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project", str(root))
    table.add_row("Embedding model", config.get("embedding_model", "N/A"))
    table.add_row("Chunk size", str(config.get("chunk_size", "N/A")))
    table.add_row("Chunk overlap", str(config.get("chunk_overlap", "N/A")))
    table.add_row("Files indexed", str(config.get("files_count", "N/A")))
    table.add_row("Chunks in store", str(store.count()))
    if config.get("last_ingest"):
        import datetime
        ts = datetime.datetime.fromtimestamp(config["last_ingest"])
        table.add_row("Last ingested", ts.strftime("%Y-%m-%d %H:%M:%S"))

    console.print()
    console.print(table)
    console.print()


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path (default: auto-detect)")
@click.confirmation_option(prompt="Are you sure you want to delete the index?")
def clean(path: str | None):
    """Delete the vector index for the project."""
    root = Path(path).resolve() if path else _find_project_root()
    codechat_dir = get_codechat_dir(root)

    import shutil
    if codechat_dir.exists():
        shutil.rmtree(codechat_dir)
        console.print("[green]Index deleted.[/]")
    else:
        console.print("[dim]No index found.[/]")


# ================================================================== Skills

def _run_skill_common(skill_name: str, target: str, path: str | None, model: str | None, show_sources: bool):
    """Shared logic for all skill commands."""
    root = Path(path).resolve() if path else _find_project_root()
    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]codechat ingest[/] first.[/]")
        return

    api_key, _, _, thinking = _get_llm_config(model)

    if api_key:
        console.print()
        is_answer = False

        def on_think(t: str):
            nonlocal is_answer
            if not is_answer and t.strip():
                console.print(f"[dim]{t}[/]", end="")

        def on_answer(t: str):
            nonlocal is_answer
            if not is_answer:
                is_answer = True
                if thinking:
                    console.print()
            console.print(t, end="")

        result = run_skill_stream(store, skill_name, target, model=model, on_think=on_think, on_answer=on_answer)
        console.print()
    else:
        with console.status("[bold cyan]Analyzing...", spinner="dots"):
            result = run_skill(store, skill_name, target, model=model)
        console.print()
        console.print(Panel(Markdown(result["answer"]), border_style="green", padding=(1, 2)))

    if show_sources or not api_key:
        if result["sources"]:
            console.print("\n[bold dim]Sources:[/]")
            seen: set[str] = set()
            for s in result["sources"]:
                fp = s["metadata"]["file_path"]
                if fp not in seen:
                    seen.add(fp)
                    sl = s["metadata"]["start_line"]
                    el = s["metadata"]["end_line"]
                    console.print(f"  [dim]*[/] {fp}:[cyan]{sl}[/]-[cyan]{el}[/]")
    console.print()


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def explain(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Explain a function, class, or file in detail."""
    _run_skill_common("explain", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=False)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def review(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Code review: find bugs, security issues, and quality problems."""
    t = " ".join(target) if target else "审查整个项目的核心代码"
    _run_skill_common("review", t, path, model, show_sources)


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def find(query: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Find code patterns: TODOs, FIXMEs, security risks, specific logic."""
    _run_skill_common("find", " ".join(query), path, model, show_sources)


@cli.command()
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def summary(path: str | None, model: str | None, show_sources: bool):
    """Generate project architecture summary."""
    _run_skill_common("summary", "生成项目架构概览", path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def trace(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Trace function call chain."""
    _run_skill_common("trace", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("file_a", required=True)
@click.argument("file_b", required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def compare(file_a: str, file_b: str, path: str | None, model: str | None, show_sources: bool):
    """Compare two files or modules."""
    target = f"对比 {file_a} 和 {file_b} 的异同"
    _run_skill_common("compare", target, path, model, show_sources)


@cli.command()
@click.argument("target", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--show-sources", is_flag=True, help="Show source files")
def test_suggest(target: tuple[str, ...], path: str | None, model: str | None, show_sources: bool):
    """Suggest test cases for a function or module."""
    _run_skill_common("test", " ".join(target), path, model, show_sources)


@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--path", "-p", default=None, help="Project root path")
@click.option("--model", "-m", default=None, help="LLM model to use")
@click.option("--steps", "-s", default=5, type=int, help="Max agent steps (default 5)")
@click.option("--no-plan", is_flag=True, help="Skip planning phase")
def agent(question: tuple[str, ...], path: str | None, model: str | None, steps: int, no_plan: bool):
    """Multi-step agent: Planning + Tools + Memory + Action + LLM."""
    root = Path(path).resolve() if path else _find_project_root()
    q = " ".join(question)

    store = VectorStore(root)
    if store.count() == 0:
        console.print("[red]No data indexed. Run [cyan]codechat ingest[/] first.[/]")
        return

    a = CodeAgent(store, root, model=model, max_steps=steps, use_planning=not no_plan)

    api_key, _, llm_model, _ = _get_llm_config(model)

    if not api_key:
        # No LLM, just search
        with console.status("[bold cyan]Searching...", spinner="dots"):
            results = store.query(q, n_results=5)
        if results:
            from .rag import _format_context
            console.print(Panel(Markdown("相关代码：\n\n" + _format_context(results)), border_style="green", padding=(1, 2)))
        else:
            console.print("[red]未找到相关代码。[/]")
        return

    console.print(f"\n  [dim]LLM: {llm_model} | Steps: {steps} | Planning: {'on' if not no_plan else 'off'}[/]\n")

    def on_step(num, tool_name, preview):
        console.print(f"  [cyan]Step {num}[/] [yellow]→ {tool_name}[/]")
        console.print(f"  [dim]{preview}[/]\n")

    def on_think(t):
        console.print(f"  [dim]Think: {t}[/]")

    def on_answer(t):
        console.print()
        console.print(Panel(Markdown(t), title="[bold green]Answer[/]", border_style="green", padding=(1, 2)))

    result = a.run(q, on_step=on_step, on_think=on_think, on_answer=on_answer)

    # Show execution summary
    if result.actions:
        console.print(f"  [dim]Actions: {result.steps_taken} steps | Memory: {result.memory_entries} entries[/]")
    console.print()


def main():
    cli()


if __name__ == "__main__":
    main()
