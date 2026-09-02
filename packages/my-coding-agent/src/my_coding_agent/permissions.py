from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from my_agent_core.events import HookResult, ToolExecutionStart

PermissionMode = Literal["ask", "auto", "plan"]


@dataclass
class PermissionController:
    mode: PermissionMode = "ask"
    approve_rest: bool = False
    READ_ONLY = frozenset({"list_files", "search", "read", "task"})

    def set_mode(self, mode: PermissionMode) -> None:
        if mode not in {"ask", "auto", "plan"}:
            raise ValueError("mode must be ask, auto, or plan")
        self.mode = mode
        self.approve_rest = False

    async def __call__(self, event: ToolExecutionStart) -> HookResult | None:
        if event.tool_name in self.READ_ONLY:
            return None
        if self.mode == "auto" or self.approve_rest:
            return None
        if self.mode == "plan":
            return HookResult(block=True, reason=f"plan mode blocks '{event.tool_name}'")
        try:
            answer = (
                await asyncio.to_thread(
                    input,
                    f"permission> allow {self._describe(event)}? [y]es / [n]o / [a]ll: ",
                )
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in {"y", "yes"}:
            return None
        if answer in {"a", "all"}:
            self.approve_rest = True
            return None
        return HookResult(block=True, reason="permission denied by user")

    @staticmethod
    def _describe(event: ToolExecutionStart) -> str:
        if event.tool_name == "bash":
            return f"bash {event.args.get('command', '')!r}"
        if "path" in event.args:
            return f"{event.tool_name} {event.args.get('path')!r}"
        return event.tool_name
