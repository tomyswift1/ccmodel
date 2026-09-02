# my-agent-core

Focused runtime for the final coding-agent project.

Retained modules:

- `agent.py` — native tool-calling loop
- `tools/` and `registry.py` — local tool abstraction and execution
- `events.py` — lifecycle Hook system
- `session.py` / `session_store.py` — persistent session history
- `context.py` — four-layer context compaction
- `memory.py` — cross-session memory
- `subagents.py` / `tasks.py` — isolated subagent delegation

