"""Coding product assembly: small core runtime plus opt-in-safe advanced features."""
from __future__ import annotations
from pathlib import Path
from my_agent_core import Agent
from my_agent_core.tools import Tool
from my_coding_agent.mutation_queue import FileMutationQueue
from my_coding_agent.tools import (
    make_bash_tool,
    make_edit_tool,
    make_list_files_tool,
    make_read_tool,
    make_search_tool,
    make_write_tool,
)


def build_coding_tools(workspace: str | Path) -> list[Tool]:
    workspace = Path(workspace).resolve()
    mutations = FileMutationQueue()
    return [
        make_list_files_tool(workspace),
        make_search_tool(workspace),
        make_read_tool(workspace),
        make_write_tool(workspace, mutations),
        make_edit_tool(workspace, mutations),
        make_bash_tool(workspace),
    ]


def builtin_subagent_dir() -> Path:
    """Directory containing the product's built-in read-only subagent definitions."""
    return Path(__file__).resolve().parent / "agents"


class CodingAgent:
    """Focused coding product with memory and read-only subagent delegation."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        llm,
        session,
        system_prompt: str | None = None,
        extra_tools: list | tuple = (),
        **kw,
    ):
        self.workspace = Path(workspace).resolve()
        builtins = builtin_subagent_dir()
        agent_options = {
            "subagent_dirs": [builtins],
            "memory_dir": self.workspace / ".my_agent_core" / "memory",
        }
        agent_options.update(kw)
        self.agent = Agent(
            llm=llm,
            tools=build_coding_tools(self.workspace) + list(extra_tools),
            session=session,
            system_prompt=system_prompt,
            **agent_options,
        )

    async def run(self, user_input: str):
        return await self.agent.run(user_input)
