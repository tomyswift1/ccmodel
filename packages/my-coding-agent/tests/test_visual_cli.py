from my_coding_agent.cli import describe_tool, visual_panel, visual_rule


def test_visual_panel_and_rule_render_terminal_ui():
    panel = visual_panel("AGENT DASHBOARD", ["memory: on", "subagents: reviewer, tester"])
    assert "AGENT DASHBOARD" in panel
    assert "memory: on" in panel
    assert panel.startswith("╭") and panel.endswith("╯")
    assert "STEP 2" in visual_rule("STEP 2")


def test_tool_descriptions_cover_memory_and_subagent():
    assert describe_tool("memory", {"target": "user", "action": "add"}) == "memory user:add"
    text = describe_tool("task", {"agent_type": "reviewer", "prompt": "review app.py"})
    assert text.startswith("delegate reviewer:")
