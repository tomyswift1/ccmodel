from my_coding_agent.cli import describe_tool


def test_edit_summary_hides_large_replacement_payload():
    args = {
        "path": "calculator.py",
        "old_text": "old secret-looking code" * 20,
        "new_text": "new replacement code" * 20,
    }
    rendered = describe_tool("edit", args)
    assert rendered == "edit calculator.py"
    assert "old secret-looking code" not in rendered
    assert "new replacement code" not in rendered


def test_write_summary_reports_path_and_line_count_not_body():
    rendered = describe_tool(
        "write",
        {"path": "test_calculator.py", "content": "line1\nline2\nline3"},
    )
    assert rendered == "write test_calculator.py (3 lines)"
    assert "line1" not in rendered


def test_bash_summary_keeps_command_visible():
    rendered = describe_tool("bash", {"command": "python -m unittest -v"})
    assert "python -m unittest -v" in rendered
