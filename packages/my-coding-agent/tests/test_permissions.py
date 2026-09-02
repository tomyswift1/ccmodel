import pytest
from my_agent_core.events import ToolExecutionStart
from my_coding_agent.permissions import PermissionController

@pytest.mark.anyio
async def test_plan_mode_blocks_mutation_but_allows_read():
    gate=PermissionController("plan")
    assert await gate(ToolExecutionStart("1","read",{"path":"a.py"})) is None
    blocked=await gate(ToolExecutionStart("2","write",{"path":"a.py","content":"x"}))
    assert blocked is not None and blocked.block is True

@pytest.mark.anyio
async def test_auto_mode_allows_guarded_tool():
    assert await PermissionController("auto")(ToolExecutionStart("1","bash",{"command":"pytest -q"})) is None

@pytest.mark.anyio
async def test_plan_mode_allows_read_only_subagent_delegation():
    gate = PermissionController("plan")
    event = ToolExecutionStart("3", "task", {"prompt": "review", "agent_type": "reviewer"})
    assert await gate(event) is None


@pytest.mark.anyio
async def test_plan_mode_blocks_long_term_memory_mutation():
    gate = PermissionController("plan")
    event = ToolExecutionStart("4", "memory", {"target": "user", "action": "add", "content": "x"})
    blocked = await gate(event)
    assert blocked is not None and blocked.block is True
