"""Focused tests for explicit, read-only subagent discovery and delegation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from my_agent_llm import Response, StreamChunk

from my_agent_core.agent import Agent
from my_agent_core.session import Session
from my_agent_core.subagents import SubagentManager
from my_agent_core.tools import tool
from my_agent_core.tools.builtin import make_task_tool


def _write_agent(
    root: Path,
    name: str,
    description: str = "desc",
    content: str = "body",
    extra: str = "",
) -> Path:
    p = root / f"{name}.md"
    p.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{content}",
        encoding="utf-8",
    )
    return p


def test_load_basic(tmp_path: Path):
    _write_agent(tmp_path, "reviewer", description="review code", content="checklist")
    subagents = SubagentManager([tmp_path]).list()
    assert len(subagents) == 1
    assert subagents[0].name == "reviewer"
    assert subagents[0].description == "review code"
    assert subagents[0].content == "checklist"


def test_name_falls_back_to_stem_and_description_is_required(tmp_path: Path):
    (tmp_path / "reviewer.md").write_text(
        "---\ndescription: review\n---\nbody", encoding="utf-8"
    )
    (tmp_path / "invalid.md").write_text("---\nname: invalid\n---\nbody", encoding="utf-8")
    assert [s.name for s in SubagentManager([tmp_path]).list()] == ["reviewer"]


def test_empty_configuration_loads_no_subagents():
    assert SubagentManager().list() == []


def test_frontmatter_keeps_only_tools_and_max_turns(tmp_path: Path):
    _write_agent(
        tmp_path,
        "tester",
        extra="tools: list_files, search, read\nmaxTurns: 8\n",
    )
    sub = SubagentManager([tmp_path]).get("tester")
    assert sub is not None
    assert sub.tools == ("list_files", "search", "read")
    assert sub.max_turns == 8


def test_ignores_non_markdown_readme_nested_and_bad_yaml(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("---\ndescription: d\n---\nbody", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_agent(nested, "inner")
    (tmp_path / "bad.md").write_text(
        "---\ndescription: [unclosed\n---\nbody", encoding="utf-8"
    )
    assert SubagentManager([tmp_path]).list() == []


class FakeLLM:
    def __init__(self, responses: list[Response]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def achat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self.responses.pop(0)

    async def achat_stream(self, *, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        resp = self.responses.pop(0)
        yield StreamChunk(content=resp.content, tool_calls=resp.tool_calls, usage=resp.usage)

    def chat(self, *, messages, tools=None, **kwargs) -> Response:
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        return self.responses.pop(0)


def _response(content: str = "", tool_calls=None) -> Response:
    return Response(content=content, model="fake", tool_calls=tool_calls)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_time() -> str:
    """Return a fixed test time."""
    return "12:00"


def _task_call(prompt: str, agent_type: str) -> dict:
    return {
        "id": "1",
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps({"prompt": prompt, "agent_type": agent_type}),
        },
    }


def _parent(manager: SubagentManager, llm: FakeLLM) -> Agent:
    agent = Agent(
        llm=llm,
        tools=[multiply, get_time],
        session=Session(path=Path(tempfile.mkdtemp()) / "s.jsonl"),
    )
    agent.registry.register(make_task_tool(manager, agent))
    return agent


@pytest.mark.anyio
async def test_task_delegates_with_fresh_context(tmp_path: Path):
    _write_agent(tmp_path, "reviewer", content="You are a reviewer.", extra="tools: multiply\n")
    manager = SubagentManager([tmp_path])
    llm = FakeLLM([
        _response(tool_calls=[_task_call("review this", "reviewer")]),
        _response(content="found issues"),
        _response(content="done"),
    ])
    answer = await _parent(manager, llm).run("delegate")
    assert answer == "done"
    sub_msgs = llm.calls[1]["messages"]
    assert "You are a reviewer." in sub_msgs[0].content
    assert sub_msgs[-1].content == "review this"
    assert not any(m.content == "delegate" for m in sub_msgs)


@pytest.mark.anyio
async def test_subagent_tool_whitelist_and_no_recursive_task_or_memory(tmp_path: Path):
    _write_agent(
        tmp_path,
        "reviewer",
        extra="tools: multiply, task, memory\n",
    )
    manager = SubagentManager([tmp_path])
    llm = FakeLLM([
        _response(tool_calls=[_task_call("go", "reviewer")]),
        _response(content="sub done"),
        _response(content="parent done"),
    ])
    agent = _parent(manager, llm)
    await agent.run("delegate")
    names = [t["function"]["name"] for t in llm.calls[1]["tools"]]
    assert names == ["multiply"]


@pytest.mark.anyio
async def test_unknown_agent_is_structured_tool_error_and_parent_continues(tmp_path: Path):
    manager = SubagentManager([tmp_path])
    llm = FakeLLM([
        _response(tool_calls=[_task_call("go", "missing")]),
        _response(content="parent done"),
    ])
    answer = await _parent(manager, llm).run("delegate")
    assert answer == "parent done"
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert "Unknown subagent" in tool_msgs[0].content


@pytest.mark.anyio
async def test_agent_auto_registers_task_only_when_explicit_agents_exist(tmp_path: Path):
    _write_agent(tmp_path, "reviewer", content="review")
    llm = FakeLLM([_response(content="ok")])
    agent = Agent(
        llm=llm,
        tools=[multiply],
        session=Session(path=tmp_path / "s.jsonl"),
        subagent_dirs=[tmp_path],
    )
    await agent.run("hi")
    names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert "task" in names
    assert "<available_agents>" in llm.calls[0]["messages"][0].content

    llm2 = FakeLLM([_response(content="ok")])
    agent2 = Agent(
        llm=llm2,
        tools=[multiply],
        session=Session(path=tmp_path / "s2.jsonl"),
        subagent_dirs=[],
    )
    await agent2.run("hi")
    names2 = [t["function"]["name"] for t in llm2.calls[0]["tools"]]
    assert "task" not in names2


def test_task_name_conflict_is_rejected(tmp_path: Path):
    _write_agent(tmp_path, "reviewer")

    @tool(name="task")
    def custom_task(prompt: str) -> str:
        """Conflicting task tool."""
        return prompt

    with pytest.raises(ValueError, match="task"):
        Agent(
            llm=FakeLLM([]),
            tools=[custom_task],
            session=Session(path=tmp_path / "s.jsonl"),
            subagent_dirs=[tmp_path],
        )
