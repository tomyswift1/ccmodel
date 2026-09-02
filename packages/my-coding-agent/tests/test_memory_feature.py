import pytest

from my_agent_core.session import Session
from my_coding_agent.agent import CodingAgent


class FakeLLM:
    async def achat_stream(self, **kwargs):
        if False:
            yield None


@pytest.mark.anyio
async def test_memory_persists_across_new_agent_instance(tmp_path):
    first = CodingAgent(
        workspace=tmp_path,
        llm=FakeLLM(),
        session=Session(path=tmp_path / "s1.jsonl"),
    )
    memory = first.agent.registry.get("memory")
    assert memory is not None
    result = await memory.execute(
        {
            "target": "user",
            "action": "add",
            "content": "Prefer unittest for Python regression tests.",
        }
    )
    assert result.ok is True
    assert (tmp_path / ".my_agent_core" / "memory" / "USER.md").exists()

    second = CodingAgent(
        workspace=tmp_path,
        llm=FakeLLM(),
        session=Session(path=tmp_path / "s2.jsonl"),
    )
    assert second.agent.memory_store is not None
    assert second.agent.memory_store.list_entries("user") == [
        "Prefer unittest for Python regression tests."
    ]
    assert "<MEMORY_CONTEXT>" in second.agent.messages[0].content
    assert "Prefer unittest" in second.agent.messages[0].content
