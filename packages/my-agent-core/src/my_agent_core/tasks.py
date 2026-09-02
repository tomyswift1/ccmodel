"""Subagent task lifecycle and isolated delegation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from my_agent_core.session import Session
from my_agent_core.subagents import Subagent, SubagentManager

if TYPE_CHECKING:
    from my_agent_core.agent import Agent


class TaskStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Task:
    id: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None

    def set_result(self, result: str) -> None:
        self.result = result
        self.error = None
        self.status = TaskStatus.COMPLETED

    def set_error(self, error: str) -> None:
        self.error = error
        self.result = None
        self.status = TaskStatus.ERROR


def _filter_tools(parent: Agent, subagent: Subagent) -> list:
    """Apply the subagent whitelist/blacklist and always isolate task/memory."""
    tools = [tool for tool in parent.registry.list() if tool.name not in {"task", "memory"}]
    if subagent.tools is not None:
        allowed = set(subagent.tools)
        tools = [tool for tool in tools if tool.name in allowed]
    return tools


class TaskManager:
    """Spawn a child Agent with its own Session and restricted tool set."""

    def __init__(self, manager: SubagentManager, parent: Agent):
        self._manager = manager
        self._parent = parent
        self._counter = 0

    async def start_task(self, prompt: str, subagent_type: str = "default") -> Task:
        task = self._create_task()
        try:
            task.set_result(await self._run(prompt, subagent_type, task.id))
        except Exception as exc:
            task.set_error(str(exc))
        return task

    def _create_task(self) -> Task:
        self._counter += 1
        return Task(id=f"task_{self._counter:08x}", status=TaskStatus.RUNNING)

    async def _run(self, prompt: str, subagent_type: str, task_id: str) -> str:
        from my_agent_core.agent import Agent

        subagent = self._manager.get(subagent_type)
        if subagent is None:
            available = ", ".join(sorted(self._manager.subagents)) or "(none)"
            raise ValueError(
                f"Unknown subagent '{subagent_type}'. Available: {available}"
            )

        child_session = Session(
            path=self._parent.session.path.parent
            / "subagents"
            / f"agent-{task_id}.jsonl",
            cwd=self._parent.session.cwd,
            metadata={
                "agent_type": subagent_type,
                "parent_session_id": self._parent.session.id,
            },
        )
        child_session.save()

        child = Agent(
            llm=self._parent.llm,
            tools=_filter_tools(self._parent, subagent),
            session=child_session,
            system_prompt=subagent.content,
            max_iterations=(
                subagent.max_turns
                if subagent.max_turns is not None
                else self._parent.max_iterations
            ),
            subagent_dirs=[],
            memory_dir=False,
        )
        try:
            return (await child.run(prompt)) or "(no summary)"
        except Exception as exc:
            raise RuntimeError(f"Subagent '{subagent_type}' failed: {exc}") from exc
