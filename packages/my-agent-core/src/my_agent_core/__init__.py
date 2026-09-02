"""Public API for the focused coding-agent core."""

from my_agent_core.agent import Agent
from my_agent_core.context import ContextManager
from my_agent_core.events import (
    AgentEnd,
    AgentStart,
    BeforeModelCall,
    ContextCompacted,
    Event,
    HookResult,
    Interceptable,
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
from my_agent_core.session import Session, SessionTree
from my_agent_core.session_store import SessionStore
from my_agent_core.subagents import Subagent, SubagentManager
from my_agent_core.tasks import Task, TaskManager, TaskStatus
from my_agent_core.tools import Tool, ToolResult, tool

__all__ = [
    "Agent",
    "tool",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "HookResult",
    "Interceptable",
    "Event",
    "UserInput",
    "AgentStart",
    "AgentEnd",
    "TurnStart",
    "BeforeModelCall",
    "TurnEnd",
    "MessageStart",
    "MessageUpdate",
    "MessageEnd",
    "ToolExecutionStart",
    "ToolExecutionEnd",
    "ContextCompacted",
    "Session",
    "SessionTree",
    "SessionStore",
    "ContextManager",
    "MemoryStore",
    "make_memory_tool",
    "Subagent",
    "SubagentManager",
    "Task",
    "TaskStatus",
    "TaskManager",
]
