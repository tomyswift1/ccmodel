import json
import pytest
from my_agent_llm import StreamChunk
from my_agent_core.session import Session
from my_coding_agent.agent import CodingAgent

class FakeLLM:
    def __init__(self,chunks): self.chunks=list(chunks)
    async def achat_stream(self, *, messages, tools=None, **kwargs): yield self.chunks.pop(0)

def tc(i,name,args): return {"id":i,"type":"function","function":{"name":name,"arguments":json.dumps(args)}}

@pytest.mark.anyio
async def test_read_edit_bash_final_answer(tmp_path):
    (tmp_path/"app.py").write_text("def answer():\n    return 1\n\nprint(answer())\n",encoding="utf-8")
    llm=FakeLLM([
        StreamChunk(content="",tool_calls=[tc("1","read",{"path":"app.py"})]),
        StreamChunk(content="",tool_calls=[tc("2","edit",{"path":"app.py","old_text":"return 1","new_text":"return 2"})]),
        StreamChunk(content="",tool_calls=[tc("3","bash",{"command":"python app.py"})]),
        StreamChunk(content="Task complete."),
    ])
    agent=CodingAgent(workspace=tmp_path,llm=llm,session=Session(path=tmp_path/"s.jsonl"),max_iterations=10)
    result=await agent.run("Change answer() to return 2 and verify it.")
    assert result=="Task complete."
    assert "return 2" in (tmp_path/"app.py").read_text(encoding="utf-8")
    observations=[m.content for m in agent.agent.messages if m.role=="tool"]
    assert any("Edited app.py successfully" in x for x in observations)
    assert any(x.startswith("2") for x in observations)

@pytest.mark.anyio
async def test_tool_failure_is_visible_as_error_event(tmp_path):
    from my_agent_core.events import ToolExecutionEnd

    events = []
    llm = FakeLLM([
        StreamChunk(content="", tool_calls=[tc("1", "read", {"path": "missing.py"})]),
        StreamChunk(content="Recovered after observing the tool error."),
    ])
    agent = CodingAgent(
        workspace=tmp_path,
        llm=llm,
        session=Session(path=tmp_path / "s2.jsonl"),
        max_iterations=5,
        hooks=[(ToolExecutionEnd, events.append)],
    )

    result = await agent.run("Inspect missing.py, then recover if it is absent.")
    assert result == "Recovered after observing the tool error."
    assert len(events) == 1
    assert events[0].is_error is True
    assert "does not exist" in events[0].result
