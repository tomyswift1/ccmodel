"""Core tool-calling agent loop.

The model decides what to do; the local harness executes tools and feeds the
observations back into the next model turn. Session persistence, context
compaction, hooks, long-term memory and read-only subagent delegation are kept
as first-class parts of this loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from my_agent_llm import LLM, Message  # pyright: ignore[reportMissingImports]

from my_agent_core.context import ContextManager, ContextSessionBridge
from my_agent_core.events import (
    AgentEnd,
    AgentStart,
    BeforeModelCall,
    ContextCompacted,
    Event,
    HookRegistry,
    HookResult,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    ToolExecutionEnd,
    ToolExecutionStart,
    TurnEnd,
    TurnStart,
    UserInput,
)
from my_agent_core.memory import MemoryStore, make_memory_tool
from my_agent_core.registry import ToolRegistry
from my_agent_core.session import Session
from my_agent_core.subagents import SubagentManager
from my_agent_core.tools import Tool, ToolResult
from my_agent_core.tools.builtin import make_task_tool


class Agent:
    """Local agent harness around an LLM and a registry of executable tools."""

    def __init__(
        self,
        *,
        llm: LLM,
        tools: list[Tool],
        session: Session,
        system_prompt: str | None = None,
        max_iterations: int | None = None,
        context_budget: int | None = None,
        keep_recent_tokens: int | None = None,
        model: str | None = None,
        subagent_dirs: Sequence[str | Path] | None = None,
        memory_dir: str | Path | None | Literal[False] = None,
        hooks: list[tuple[type[Event], Callable]] | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.max_iterations = max_iterations
        self.session = session
        self._system_prompt = system_prompt
        self._aborted = False

        self.hooks = HookRegistry()
        self.registry = ToolRegistry()
        self.subagent_manager = SubagentManager(subagent_dirs)
        self.memory_store = self._init_memory_store(memory_dir)

        self._register_tools(tools)
        self.messages = self._init_messages(session, system_prompt)
        self._init_context(session, context_budget, keep_recent_tokens)
        self._register_hooks(hooks)

    def _init_memory_store(
        self, memory_dir: str | Path | None | Literal[False]
    ) -> MemoryStore | None:
        if isinstance(memory_dir, bool) and not memory_dir:
            return None
        if memory_dir is not None:
            store = MemoryStore(memory_dir)
            store.load_from_disk()
            return store
        default_dir = Path.cwd() / ".my_agent_core" / "memory"
        if default_dir.exists() and default_dir.is_dir():
            store = MemoryStore(default_dir)
            store.load_from_disk()
            return store
        return None

    def _register_tools(self, tools: list[Tool]) -> None:
        for item in tools:
            self.registry.register(item)

        if self.subagent_manager:
            if self.registry.get("task") is not None:
                raise ValueError("Tool name 'task' conflicts with the built-in subagent tool")
            self.registry.register(make_task_tool(self.subagent_manager, self))

        if self.memory_store:
            if self.registry.get("memory") is not None:
                raise ValueError("Tool name 'memory' conflicts with the built-in memory tool")
            self.registry.register(make_memory_tool(self.memory_store))

    def _init_messages(self, session: Session, system_prompt: str | None) -> list[Message]:
        memory_prompt = (
            self.memory_store.format_all_for_system_prompt() if self.memory_store else None
        )
        parts = [
            part
            for part in (
                system_prompt or "",
                self.subagent_manager.format_prompt(),
                memory_prompt,
            )
            if part
        ]
        messages = session.get_full_history_messages()
        if parts:
            messages.insert(0, Message(role="system", content="\n\n".join(parts)))
        return messages

    def _init_context(
        self,
        session: Session,
        context_budget: int | None,
        keep_recent_tokens: int | None,
    ) -> None:
        self._ctx_bridge = ContextSessionBridge(session)
        self._ctx = ContextManager(
            llm=self.llm,
            keep_recent_tokens=keep_recent_tokens,
            results_dir=self._ctx_bridge.results_dir(),
            **({} if context_budget is None else {"budget": context_budget}),
        )
        self._ctx_bridge.restore_cache(self._ctx)

    def _register_hooks(self, hooks) -> None:
        for event_cls, callback in hooks or []:
            self.hooks.register(event_cls, callback)

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    def abort(self) -> None:
        """Stop the current run at the next safe point."""
        self._aborted = True

    async def run(self, user_input: str) -> str | None:
        """Run the native tool-calling loop until the model returns no tool calls."""
        self._aborted = False

        input_hook = await self._emit(UserInput(input_text=user_input))
        if isinstance(input_hook, HookResult):
            if input_hook.block:
                reason = f": {input_hook.reason}" if input_hook.reason else ""
                return f"(blocked{reason})"
            if input_hook.updated_input is not None:
                user_input = input_hook.updated_input

        # Rebuild in-memory transcript from the session's current branch before each run.
        system = [m for m in self.messages if m.role == "system"]
        self.messages = system + self.session.get_current_path_messages()

        start_hook = await self._emit(
            AgentStart(system_prompt=self.system_prompt or "", user_input=user_input)
        )
        if isinstance(start_hook, HookResult):
            if start_hook.block:
                reason = f": {start_hook.reason}" if start_hook.reason else ""
                return f"(blocked{reason})"
            if start_hook.updated_system_prompt is not None:
                replacement = Message(
                    role="system", content=start_hook.updated_system_prompt
                )
                if self.messages and self.messages[0].role == "system":
                    self.messages[0] = replacement
                else:
                    self.messages.insert(0, replacement)

        user_msg = Message(role="user", content=user_input)
        self.messages.append(user_msg)
        self.session.add_message("user", user_input)
        await self._emit(MessageStart(user_msg))
        await self._emit(MessageEnd(user_msg))

        iteration = 0
        final_text: str | None = None
        final_stop_reason = "end_turn"
        has_more_tool_calls = True

        while has_more_tool_calls:
            iteration += 1

            if self._aborted:
                await self._finish(None, iteration, "cancelled")
                return "(cancelled)"

            if self.max_iterations is not None and iteration > self.max_iterations:
                await self._finish(final_text, iteration, "max_iterations")
                return final_text

            await self._emit(TurnStart(iteration))

            tools = self.registry.get_schemas()
            view = await self._ctx.prepare(self.messages)
            ctx_hook = await self._emit(
                BeforeModelCall(messages=list(view), iteration=iteration)
            )
            if isinstance(ctx_hook, HookResult):
                if ctx_hook.block:
                    reason = ctx_hook.reason or "blocked"
                    await self._finish(None, iteration, "blocked")
                    return f"(blocked: {reason})"
                if ctx_hook.updated_messages is not None:
                    view = ctx_hook.updated_messages

            content_acc = ""
            final_tool_calls = None
            last_usage = None
            cancelled = False

            try:
                if hasattr(self.llm, "achat_stream"):
                    async for chunk in self.llm.achat_stream(
                        messages=view, tools=tools, model=self.model
                    ):
                        if self._aborted:
                            cancelled = True
                            break
                        if chunk.content:
                            content_acc += chunk.content
                        if getattr(chunk, "tool_calls", None):
                            final_tool_calls = chunk.tool_calls
                        if getattr(chunk, "usage", None):
                            last_usage = chunk.usage

                        hook = await self._emit(
                            MessageUpdate(
                                message=Message(role="assistant", content=content_acc),
                                chunk=chunk,
                            )
                        )
                        if isinstance(hook, HookResult) and hook.block:
                            self._aborted = True
                            cancelled = True
                            break
                elif hasattr(self.llm, "achat"):
                    response = await self.llm.achat(
                        messages=view, tools=tools, model=self.model
                    )
                    content_acc = response.content or ""
                    final_tool_calls = response.tool_calls
                    last_usage = response.usage
                else:
                    response = self._llm_chat(view, tools)
                    content_acc = response.content or ""
                    final_tool_calls = response.tool_calls
                    last_usage = response.usage
            except Exception:
                await self._finish(None, iteration, "model_error")
                raise

            if cancelled or self._aborted:
                await self._finish(None, iteration, "cancelled")
                return "(cancelled)"

            if last_usage:
                self._ctx.record_usage(last_usage)
            await self._handle_compaction()

            assistant = Message(
                role="assistant",
                content=content_acc,
                metadata={"tool_calls": final_tool_calls} if final_tool_calls else None,
            )
            self.messages.append(assistant)
            self.session.add_message(
                "assistant", assistant.content, **(assistant.metadata or {})
            )
            await self._emit(MessageStart(assistant))
            await self._emit(MessageEnd(assistant))

            if final_tool_calls:
                tool_results = await self._execute_tool_calls(final_tool_calls)
                has_more_tool_calls = True
            else:
                tool_results = []
                has_more_tool_calls = False
                if content_acc.strip():
                    final_text = content_acc
                else:
                    final_text = "(model returned an empty response; please retry)"
                    final_stop_reason = "empty_response"

            await self._emit(TurnEnd(message=assistant, tool_results=tool_results))

        await self._finish(final_text, iteration, final_stop_reason)
        return final_text

    async def _execute_tool_calls(self, tool_call_dicts: list[dict]) -> list[Message]:
        prepared_calls: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        direct_observations: dict[int, ToolResult] = {}

        for idx, tool_call in enumerate(tool_call_dicts):
            _name, args, error, _hook = await self._prepare_tool(tool_call)
            if error is not None:
                direct_observations[idx] = ToolResult(ok=False, error=error)
            else:
                prepared_calls.append((idx, tool_call, args))

        if prepared_calls:
            effective_calls = [
                (
                    idx,
                    {
                        **tool_call,
                        "function": {
                            **tool_call["function"],
                            "arguments": json.dumps(args),
                        },
                    },
                )
                for idx, tool_call, args in prepared_calls
            ]
            batch_results = await self.registry.execute_batch(
                [item[1] for item in effective_calls]
            )
            for (idx, tool_call), result in zip(
                effective_calls, batch_results, strict=False
            ):
                observation, is_error = await self._post_execute_hook(tool_call, result)
                direct_observations[idx] = (
                    ToolResult(ok=True, data=observation)
                    if not is_error
                    else ToolResult(ok=False, error=observation)
                )

        messages: list[Message] = []
        for idx, tool_call in enumerate(tool_call_dicts):
            result = direct_observations[idx]
            observation = result.serialize()
            tool_msg = Message(
                role="tool",
                content=observation,
                metadata={"tool_call_id": tool_call["id"]},
            )
            self.messages.append(tool_msg)
            self.session.add_message("tool", observation, tool_call_id=tool_call["id"])
            await self._emit(MessageStart(tool_msg))
            await self._emit(MessageEnd(tool_msg))
            messages.append(tool_msg)
        return messages

    def reset(self) -> None:
        """Clear the current session while preserving configuration and long-term memory."""
        self.session.reset()
        if self.memory_store:
            self.memory_store.load_from_disk()
        self.messages = self._init_messages(self.session, self._system_prompt)
        self._ctx.reset()

    async def compact(self, custom_instructions: str = "") -> None:
        """Force the L4 context summarization pass."""
        await self._ctx.force_compact(self.messages)
        await self._handle_compaction()

    def _llm_chat(self, messages, tools):
        return self.llm.chat(messages=messages, tools=tools, model=self.model)

    async def _prepare_tool(
        self, tool_call: dict
    ) -> tuple[str, dict, str | None, HookResult | None]:
        function = tool_call.get("function") or {}
        name = function.get("name", "")
        try:
            args = json.loads(function.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError) as exc:
            return name, {}, f"Invalid JSON arguments for tool '{name}': {exc}", None
        if not isinstance(args, dict):
            return name, {}, f"Invalid JSON arguments for tool '{name}': expected object dict", None

        try:
            hook = await self._emit(ToolExecutionStart(tool_call.get("id", ""), name, args))
        except Exception as exc:
            return name, args, f"Error in ToolExecutionStart hook for '{name}': {exc}", None

        if isinstance(hook, HookResult) and hook.block:
            return name, args, f"Tool '{name}' blocked: {hook.reason}", hook
        if isinstance(hook, HookResult) and hook.updated_args is not None:
            args = hook.updated_args
        return name, args, None, hook

    async def _post_execute_hook(
        self, tool_call: dict, result: ToolResult
    ) -> tuple[str, bool]:
        name = (tool_call.get("function") or {}).get("name", "")
        try:
            hook = await self._emit(
                ToolExecutionEnd(
                    tool_call.get("id", ""), name, result.serialize(), not result.ok
                )
            )
        except Exception as exc:
            return f"Error in ToolExecutionEnd hook for '{name}': {exc}", True

        if isinstance(hook, HookResult) and hook.updated_result is not None:
            return hook.updated_result, False
        return result.serialize(), not result.ok

    async def _handle_compaction(self) -> None:
        self._ctx_bridge.write_compaction(self._ctx)
        info = self._ctx.pending_compaction
        if info is not None:
            await self._emit(
                ContextCompacted(
                    tokens_before=info.tokens_before,
                    tokens_after=info.tokens_after,
                    summarized_count=info.summarized_count,
                )
            )

    async def _finish(
        self, final_text: str | None, iterations: int, stop_reason: str
    ) -> None:
        await self._emit(
            AgentEnd(
                messages=list(self.messages),
                final_text=final_text,
                iterations=iterations,
                stop_reason=stop_reason,
            )
        )

    async def _emit(self, event: Event) -> HookResult | None:
        return await self.hooks.emit(event)
