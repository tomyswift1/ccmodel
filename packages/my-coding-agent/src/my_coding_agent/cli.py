"""Interactive terminal product for the local coding agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from my_agent_core import SessionStore
from my_agent_core.events import (
    AgentEnd,
    ContextCompacted,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnStart,
)
from my_agent_llm import Config, LLM

from my_coding_agent.agent import CodingAgent
from my_coding_agent.permissions import PermissionController, PermissionMode
from my_coding_agent.prompts import build_system_prompt


class A:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"


def paint(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
        return text
    return f"{code}{text}{A.RESET}"


def shorten(text: str, limit: int = 700) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"... ({len(text) - limit} chars omitted)"


def _quoted(value: object) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def describe_tool(name: str, args: dict) -> str:
    """Render a stable one-line action summary for humans and video recording."""
    if name == "read":
        suffix = []
        if args.get("offset") is not None:
            suffix.append(f"offset={args['offset']}")
        if args.get("limit") is not None:
            suffix.append(f"limit={args['limit']}")
        extra = f" ({', '.join(suffix)})" if suffix else ""
        return f"read {args.get('path', '?')}{extra}"
    if name == "list_files":
        path = args.get("path", ".")
        pattern = args.get("pattern", "*")
        return f"list_files {path} pattern={_quoted(pattern)}"
    if name == "search":
        return f"search {_quoted(args.get('query', ''))} in {args.get('path', '.')}"
    if name == "write":
        content = args.get("content", "")
        lines = len(content.splitlines()) if isinstance(content, str) else "?"
        return f"write {args.get('path', '?')} ({lines} lines)"
    if name == "edit":
        return f"edit {args.get('path', '?')}"
    if name == "bash":
        return f"bash {_quoted(args.get('command', ''))}"
    if name == "memory":
        return f"memory {args.get('target', '?')}:{args.get('action', '?')}"
    if name == "task":
        prompt = shorten(str(args.get("prompt", "")), 90)
        return f"delegate {args.get('agent_type', 'default')}: {prompt}"
    return f"{name} {shorten(str(args), 180)}"


def visual_rule(label: str, width: int = 78) -> str:
    label = f" {label} "
    fill = max(2, width - len(label) - 2)
    return "├" + label + "─" * fill + "┤"


def visual_panel(title: str, rows: list[str], width: int = 78) -> str:
    width = max(48, width)
    inner = width - 2
    top_label = f" {title} "
    top = "╭" + top_label + "─" * max(0, inner - len(top_label)) + "╮"
    body = []
    for row in rows:
        text = shorten(str(row), inner - 3)
        body.append("│ " + text.ljust(inner - 1) + "│")
    bottom = "╰" + "─" * inner + "╯"
    return "\n".join([top, *body, bottom])


@dataclass
class VisualRenderer:
    """Event-driven visual terminal renderer; it observes the Agent without owning the loop."""

    streamed: bool = False
    tool_count: int = 0
    failures: int = 0
    compactions: int = 0

    def turn(self, event: TurnStart) -> None:
        self.streamed = False
        print(paint(visual_rule(f"STEP {event.iteration}"), A.DIM))

    def msg(self, event: MessageUpdate) -> None:
        delta = getattr(getattr(event, "chunk", None), "content", "")
        if not delta:
            return
        if not self.streamed:
            sys.stdout.write(paint("│ ASSISTANT  ", A.GREEN))
            self.streamed = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    def tool_start(self, event: ToolExecutionStart) -> None:
        self.tool_count += 1
        if self.streamed:
            print()
        print(paint(f"│ ACTION     {describe_tool(event.tool_name, event.args)}", A.CYAN))

    def tool_end(self, event: ToolExecutionEnd) -> None:
        if event.is_error:
            self.failures += 1
        label = "ERROR" if event.is_error else "OBSERVE"
        color = A.RED if event.is_error else A.DIM
        result = shorten(event.result, 620)
        if "\n" in result:
            result = "\n│            " + result.replace("\n", "\n│            ")
        else:
            result = " " + result
        print(paint(f"│ {label:<10}{result}", color))

    def compact(self, event: ContextCompacted) -> None:
        self.compactions += 1
        print(
            paint(
                f"│ CONTEXT    compacted {event.tokens_before} -> {event.tokens_after} tokens "
                f"({event.summarized_count} messages summarized)",
                A.YELLOW,
            )
        )

    def end(self, event: AgentEnd) -> None:
        if self.streamed:
            print()
        label = "COMPLETED" if event.stop_reason in {"completed", "stop", "end_turn"} else event.stop_reason.upper()
        summary = (
            f"{label} · {event.iterations} step(s) · {self.tool_count} tool call(s) · "
            f"{self.failures} tool error(s) · {self.compactions} compaction(s)"
        )
        print(paint(visual_rule(summary), A.DIM))


@dataclass
class Renderer:
    streamed: bool = False

    def turn(self, event: TurnStart) -> None:
        self.streamed = False
        print(paint(f"\n[step {event.iteration}]", A.DIM))

    def msg(self, event: MessageUpdate) -> None:
        delta = getattr(getattr(event, "chunk", None), "content", "")
        if not delta:
            return
        if not self.streamed:
            sys.stdout.write(paint("assistant> ", A.GREEN))
            self.streamed = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    def tool_start(self, event: ToolExecutionStart) -> None:
        if self.streamed:
            print()
        print(paint(f"tool> {describe_tool(event.tool_name, event.args)}", A.CYAN))

    def tool_end(self, event: ToolExecutionEnd) -> None:
        label = "error" if event.is_error else "obs"
        color = A.RED if event.is_error else A.DIM
        result = shorten(event.result)
        if "\n" in result:
            result = "\n  " + result.replace("\n", "\n  ")
        else:
            result = " " + result
        print(paint(f"{label}>{result}", color))

    def end(self, event: AgentEnd) -> None:
        if self.streamed:
            print()
        label = "done" if event.stop_reason in {"completed", "stop", "end_turn"} else event.stop_reason
        print(paint(f"[{label} | {event.iterations} step(s)]", A.DIM))


def _read_int_setting(explicit: int | None, env_name: str, default: int) -> int:
    if explicit is not None:
        return explicit
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw!r}") from exc


def build_llm_from_env(
    model: str | None = None,
    base_url: str | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> LLM:
    """Build the OpenAI-compatible client without accepting API keys on the command line."""
    key = os.getenv("AGENT_API_KEY")
    model = model or os.getenv("AGENT_MODEL")
    base_url = base_url or os.getenv("AGENT_BASE_URL")

    if not key:
        raise RuntimeError("Missing API key: set AGENT_API_KEY.")
    if not model:
        raise RuntimeError("Missing model: set AGENT_MODEL.")

    timeout = _read_int_setting(timeout, "AGENT_TIMEOUT", 30)
    max_retries = _read_int_setting(max_retries, "AGENT_MAX_RETRIES", 3)

    try:
        config = Config(
            model=model,
            api_key=key,
            base_url=base_url,
            temperature=0.2,
            timeout=timeout,
            max_retries=max_retries,
        )
        return LLM(config=config)
    except Exception as exc:
        raise RuntimeError(f"Invalid LLM configuration: {exc}") from exc


class CodingShell:
    def __init__(
        self,
        workspace: Path,
        llm: LLM,
        mode: PermissionMode = "ask",
        max_iterations: int = 30,
        visual: bool = True,
    ) -> None:
        self.workspace = workspace.resolve()
        self.llm = llm
        self.store = SessionStore(workspace=self.workspace)
        self.permissions = PermissionController(mode)
        self.max_iterations = max_iterations
        self.visual = visual
        self.session = self.store.create()
        self.agent = self._build()

    def _build(self) -> CodingAgent:
        renderer = VisualRenderer() if self.visual else Renderer()
        return CodingAgent(
            workspace=self.workspace,
            llm=self.llm,
            session=self.session,
            system_prompt=build_system_prompt(self.workspace),
            max_iterations=self.max_iterations,
            hooks=[
                (TurnStart, renderer.turn),
                (MessageUpdate, renderer.msg),
                (ToolExecutionStart, renderer.tool_start),
                (ToolExecutionStart, self.permissions),
                (ToolExecutionEnd, renderer.tool_end),
                *([(ContextCompacted, renderer.compact)] if hasattr(renderer, "compact") else []),
                (AgentEnd, renderer.end),
            ],
        )

    def _banner(self) -> None:
        tools = ", ".join(t.name for t in self.agent.agent.registry.list())
        memory = "on" if self.agent.agent.memory_store else "off"
        agents = ", ".join(a.name for a in self.agent.agent.subagent_manager.list()) or "none"
        rows = [
            f"workspace : {self.workspace}",
            f"model     : openai/{self.llm.model}",
            f"mode      : {self.permissions.mode}    memory: {memory}    subagents: {agents}",
            f"tools     : {tools}",
            "commands  : /dashboard /memory /agents /help",
        ]
        if self.visual:
            print(paint(visual_panel("MY CODING AGENT", rows), A.BOLD))
        else:
            print(paint("my-coding-agent", A.BOLD))
            for row in rows:
                print(row)
        print()

    def _show_memory(self) -> None:
        store = self.agent.agent.memory_store
        if store is None:
            print("long-term memory is disabled")
            return
        rows: list[str] = []
        for target in ("user", "memory"):
            used, limit = store.usage(target)
            entries = store.list_entries(target)
            rows.append(f"{target.upper()}  {used}/{limit} chars  {len(entries)} entr{'y' if len(entries)==1 else 'ies'}")
            rows.extend(f"  - {entry}" for entry in entries)
            if not entries:
                rows.append("  (empty)")
        print(visual_panel("LONG-TERM MEMORY", rows) if self.visual else "\n".join(rows))

    def _show_agents(self) -> None:
        agents = self.agent.agent.subagent_manager.list()
        rows = []
        for sub in agents:
            tools = ", ".join(sub.tools or ()) or "inherited"
            rows.append(f"{sub.name}: {sub.description}")
            rows.append(f"  tools={tools}; maxTurns={sub.max_turns or self.max_iterations}")
        if not rows:
            rows = ["(no subagents)"]
        print(visual_panel("SUBAGENTS", rows) if self.visual else "\n".join(rows))

    def _show_dashboard(self) -> None:
        core = self.agent.agent
        mem = core.memory_store
        if mem:
            u_used, u_limit = mem.usage("user")
            m_used, m_limit = mem.usage("memory")
            memory_line = f"on · USER {u_used}/{u_limit} · MEMORY {m_used}/{m_limit}"
        else:
            memory_line = "off"
        rows = [
            f"workspace : {self.workspace}",
            f"session   : {self.session.id}",
            f"model     : openai/{self.llm.model}",
            f"mode      : {self.permissions.mode}",
            f"messages  : {len(core.messages)}",
            f"context   : budget={core._ctx.budget} tokens; keep_recent={core._ctx.keep_recent_tokens}",
            f"memory    : {memory_line}",
            f"subagents : {', '.join(a.name for a in core.subagent_manager.list()) or 'none'}",
            f"tools     : {', '.join(t.name for t in core.registry.list())}",
        ]
        print(visual_panel("AGENT DASHBOARD", rows) if self.visual else "\n".join(rows))

    async def repl(self) -> None:
        self._banner()

        while True:
            try:
                text = (await asyncio.to_thread(input, paint("you> ", A.YELLOW))).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return

            if not text:
                continue
            if text.startswith("/"):
                if await self.command(text):
                    return
                continue

            try:
                await self.agent.run(text)
            except Exception as exc:
                print(paint(f"agent error: {exc}", A.RED))

    async def command(self, raw: str) -> bool:
        cmd, _, arg = raw[1:].partition(" ")
        cmd = cmd.lower().strip()
        arg = arg.strip()

        if cmd in {"exit", "quit", "q"}:
            print("bye")
            return True
        if cmd == "help":
            print(
                "/help  /dashboard  /memory  /agents  /status  /mode ask|auto|plan  "
                "/sessions  /new  /resume <id>  /compact  /exit"
            )
            return False
        if cmd == "dashboard":
            self._show_dashboard()
            return False
        if cmd == "memory":
            self._show_memory()
            return False
        if cmd == "agents":
            self._show_agents()
            return False
        if cmd == "status":
            print(
                f"workspace: {self.workspace}\n"
                f"session: {self.session.id}\n"
                f"mode: {self.permissions.mode}\n"
                f"timeout: {self.llm.config.timeout}s\n"
                f"retries: {self.llm.config.max_retries}\n"
                f"memory: {'on' if self.agent.agent.memory_store else 'off'}\n"
                f"subagents: {', '.join(a.name for a in self.agent.agent.subagent_manager.list()) or 'none'}"
            )
            return False
        if cmd == "mode":
            if arg not in {"ask", "auto", "plan"}:
                print("usage: /mode ask|auto|plan")
            else:
                self.permissions.set_mode(arg)  # type: ignore[arg-type]
                print(f"mode -> {arg}")
            return False
        if cmd == "sessions":
            sessions = self.store.list()[:20]
            if not sessions:
                print("no sessions")
            for meta in sessions:
                print(f"{meta.id}  entries={meta.entries}  created={meta.created_at}")
            return False
        if cmd == "new":
            self.session = self.store.create()
            self.agent = self._build()
            print(f"new session: {self.session.id}")
            return False
        if cmd == "resume":
            if not arg:
                print("usage: /resume <id-or-prefix>")
                return False
            try:
                self.session = self.store.open(arg)
                self.agent = self._build()
                print(f"resumed: {self.session.id}")
            except ValueError as exc:
                print(paint(str(exc), A.RED))
            return False
        if cmd == "compact":
            await self.agent.agent.compact()
            print("context compacted")
            return False

        print(f"unknown command: /{cmd}")
        return False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local coding agent")
    p.add_argument("workspace", nargs="?", default=".")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--timeout", type=int, help="API request timeout in seconds")
    p.add_argument("--max-retries", type=int, help="API transport retry count")
    p.add_argument("--yes", action="store_true", help="auto-approve mutating tools for this run")
    p.add_argument("--plan", action="store_true", help="start in read-only plan mode")
    p.add_argument("--max-iterations", type=int, default=30, help="agent loop safety cap")
    p.add_argument("--plain", action="store_true", help="disable the visual terminal dashboard renderer")
    return p


async def amain(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        print(f"invalid workspace: {workspace}", file=sys.stderr)
        return 2

    load_dotenv()
    try:
        llm = build_llm_from_env(
            args.model,
            args.base_url,
            args.timeout,
            args.max_retries,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    mode: PermissionMode = "auto" if args.yes else ("plan" if args.plan else "ask")
    await CodingShell(workspace, llm, mode, args.max_iterations, visual=not args.plain).repl()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
