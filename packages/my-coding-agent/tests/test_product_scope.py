from my_agent_core.session import Session
from my_coding_agent.agent import CodingAgent


class FakeLLM:
    async def achat_stream(self, **kwargs):
        if False:
            yield None


def test_product_enables_memory_and_read_only_subagents_by_default(tmp_path):
    agent = CodingAgent(
        workspace=tmp_path,
        llm=FakeLLM(),
        session=Session(path=tmp_path / "s.jsonl"),
    )
    names = {t.name for t in agent.agent.registry.list()}
    assert {"list_files", "search", "read", "write", "edit", "bash", "task", "memory"} <= names
    assert agent.agent.memory_store is not None
    assert agent.agent.memory_store.mem_dir == tmp_path / ".my_agent_core" / "memory"
    assert {a.name for a in agent.agent.subagent_manager.list()} == {"default", "reviewer", "tester"}
    for sub in agent.agent.subagent_manager.list():
        assert set(sub.tools or ()) <= {"list_files", "search", "read"}
