"""Workspace-bound local coding tools.

Read-only tools are safe to run in parallel. Mutating or shell tools are marked
sequential so a model-generated batch cannot accidentally reorder side effects.
Failures are represented structurally with ``ToolResult(ok=False)`` instead of
ordinary strings, allowing the Agent/CLI to distinguish observations from errors.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired

from my_agent_core.tools import Tool, ToolResult

from my_coding_agent.mutation_queue import FileMutationQueue

_TIMEOUT_SECONDS = 120
_DANGEROUS = [
    "rm -rf /",
    "sudo",
    "shutdown",
    "reboot",
    "> /dev/",
    "format c:",
    "del /f /s /q c:\\",
]
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".my_agent_core",
}


def _error(message: str) -> ToolResult:
    return ToolResult(ok=False, error=message)


def _safe_path(root: Path, p: str) -> Path:
    """Resolve a file-tool path and reject workspace traversal."""
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def make_read_tool(root: str | Path) -> Tool:
    root = Path(root).resolve()

    def read(path: str, limit: int | None = None, offset: int | None = None) -> str | ToolResult:
        """Read a UTF-8 text file inside the workspace."""
        try:
            fp = _safe_path(root, path)
            if not fp.exists():
                return _error(f"File '{path}' does not exist.")
            if fp.is_dir():
                return _error(f"'{path}' is a directory, not a regular file.")
            if offset is not None and offset < 1:
                return _error("offset must be >= 1.")
            if limit is not None and limit < 1:
                return _error("limit must be >= 1.")

            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            if offset is not None and offset > total and total > 0:
                return _error(
                    f"Offset {offset} is beyond end of file "
                    f"('{path}' has only {total} lines total)."
                )

            start = 0 if offset is None else offset - 1
            selected = lines[start:]
            if limit is not None and limit < len(selected):
                remaining = len(selected) - limit
                selected = selected[:limit] + [
                    f"... ({remaining} more lines, {total} lines total)"
                ]
            return "\n".join(selected)[:50000]
        except Exception as exc:
            return _error(str(exc))

    return Tool(func=read, name="read", is_parallel_safe=True)


def make_list_files_tool(root: str | Path) -> Tool:
    root = Path(root).resolve()

    def list_files(path: str = ".", pattern: str = "*", limit: int = 300) -> str | ToolResult:
        """List files recursively under a workspace directory."""
        try:
            if limit < 1:
                return _error("limit must be >= 1.")
            base = _safe_path(root, path)
            if not base.exists():
                return _error(f"Path '{path}' does not exist.")
            if not base.is_dir():
                return _error(f"'{path}' is not a directory.")

            results: list[str] = []
            for current, dirs, files in os.walk(base):
                dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                for name in sorted(files):
                    fp = Path(current) / name
                    rel = fp.relative_to(root).as_posix()
                    if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                        results.append(rel)
                        if len(results) >= limit:
                            return "\n".join(results) + f"\n... (stopped at {limit} files)"
            return "\n".join(results) if results else "(no matching files)"
        except Exception as exc:
            return _error(str(exc))

    return Tool(func=list_files, name="list_files", is_parallel_safe=True)


def make_search_tool(root: str | Path) -> Tool:
    root = Path(root).resolve()

    def search(
        query: str,
        path: str = ".",
        pattern: str = "*",
        limit: int = 100,
    ) -> str | ToolResult:
        """Search workspace text files for a literal string."""
        try:
            if limit < 1:
                return _error("limit must be >= 1.")
            base = _safe_path(root, path)
            if not base.exists():
                return _error(f"Path '{path}' does not exist.")

            candidates = [base] if base.is_file() else []
            if base.is_dir():
                for current, dirs, files in os.walk(base):
                    dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
                    for name in sorted(files):
                        fp = Path(current) / name
                        rel = fp.relative_to(root).as_posix()
                        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel, pattern):
                            candidates.append(fp)

            matches: list[str] = []
            for fp in candidates:
                try:
                    if fp.stat().st_size > 2_000_000:
                        continue
                    text = fp.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                rel = fp.relative_to(root).as_posix()
                for line_no, line in enumerate(text.splitlines(), 1):
                    if query in line:
                        matches.append(f"{rel}:{line_no}:{line.strip()}")
                        if len(matches) >= limit:
                            return "\n".join(matches) + f"\n... (stopped at {limit} matches)"
            return "\n".join(matches) if matches else "(no matches)"
        except Exception as exc:
            return _error(str(exc))

    return Tool(func=search, name="search", is_parallel_safe=True)


def make_write_tool(
    root: str | Path,
    mutation_queue: FileMutationQueue | None = None,
) -> Tool:
    root = Path(root).resolve()
    queue = mutation_queue or FileMutationQueue()

    async def write(path: str, content: str) -> str | ToolResult:
        """Create or overwrite a UTF-8 text file inside the workspace."""
        try:
            fp = _safe_path(root, path)
            lock = await queue.get_lock(fp)
            async with lock:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(content, encoding="utf-8")
                return (
                    f"Wrote {len(content.encode('utf-8'))} bytes "
                    f"({len(content.splitlines())} lines) to {path}"
                )
        except Exception as exc:
            return _error(f"Error writing to '{path}': {exc}")

    return Tool(func=write, name="write", is_parallel_safe=False)


def make_edit_tool(
    root: str | Path,
    mutation_queue: FileMutationQueue | None = None,
) -> Tool:
    root = Path(root).resolve()
    queue = mutation_queue or FileMutationQueue()

    async def edit(path: str, old_text: str, new_text: str) -> str | ToolResult:
        """Replace one unique exact text occurrence in a workspace file."""
        try:
            fp = _safe_path(root, path)
            if not fp.exists():
                return _error(f"File '{path}' does not exist.")
            if fp.is_dir():
                return _error(f"'{path}' is a directory, not a regular file.")

            lock = await queue.get_lock(fp)
            async with lock:
                content = fp.read_text(encoding="utf-8", errors="replace")
                count = content.count(old_text)
                if count == 0:
                    return _error(
                        f"Text not found in {path} (file has {len(content.splitlines())} lines total). "
                        f"Tip: Check exact whitespace, indentation, and newlines, or call read('{path}') first."
                    )
                if count > 1:
                    return _error(
                        f"Could not edit '{path}': old_text matched {count} locations. "
                        "Please provide more surrounding context lines to ensure a unique match."
                    )
                fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
                return f"Edited {path} successfully (1 replacement made)"
        except Exception as exc:
            return _error(f"Error editing '{path}': {exc}")

    return Tool(func=edit, name="edit", is_parallel_safe=False)


def make_bash_tool(root: str | Path) -> Tool:
    root = Path(root).resolve()

    def bash(command: str) -> str | ToolResult:
        """Run a non-interactive shell command with the workspace as cwd."""
        if any(d.lower() in command.lower() for d in _DANGEROUS):
            return _error("Dangerous command blocked")

        try:
            cmd = ["cmd.exe", "/c", command] if os.name == "nt" else ["/bin/sh", "-c", command]
            proc = Popen(cmd, cwd=root, stdout=PIPE, stderr=PIPE, text=True)
            try:
                out, err = proc.communicate(timeout=_TIMEOUT_SECONDS)
                text = (out + err).strip()
                if proc.returncode != 0:
                    detail = text[:50000] if text else "(no output)"
                    return _error(f"{detail}\n(exit code {proc.returncode})")
                return text[:50000] if text else "(no output)"
            except TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
                captured = ((out or "") + (err or "")).strip()
                partial = captured[-2000:] if captured else "(no output captured before timeout)"
                return _error(
                    f"Timeout ({_TIMEOUT_SECONDS}s) for command '{command}'.\n"
                    f"=== Output before timeout ===\n{partial}"
                )
        except OSError as exc:
            return _error(str(exc))

    return Tool(func=bash, name="bash", is_parallel_safe=False)
