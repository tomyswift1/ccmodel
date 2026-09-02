from pathlib import Path

import pytest
from my_agent_llm import Response, StreamChunk

from my_agent_core.session import Session
from my_coding_agent.agent import CodingAgent


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def achat_stream(self, *, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        resp = self.responses.pop(0)
        yield StreamChunk(
            content=resp.content,
            tool_calls=resp.tool_calls,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
        )


@pytest.mark.anyio
async def test_reviewer_subagent_runs_in_independent_session_with_read_only_tools(tmp_path: Path):
    (tmp_path / "app.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    llm = FakeLLM([Response(content="No correctness issue found.", model="fake")])
    parent_session = Session(path=tmp_path / "parent.jsonl")
    agent = CodingAgent(workspace=tmp_path, llm=llm, session=parent_session)

    task = agent.agent.registry.get("task")
    assert task is not None
    result = await task.execute({"prompt": "Review app.py", "agent_type": "reviewer"})

    assert result.ok is True
    assert result.data == "No correctness issue found."
    child_files = list((tmp_path / "subagents").glob("agent-*.jsonl"))
    assert len(child_files) == 1
    child_tool_names = {
        schema["function"]["name"] for schema in llm.calls[0]["tools"]
    }
    assert child_tool_names == {"list_files", "search", "read"}
