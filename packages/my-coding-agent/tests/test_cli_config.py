import os

import pytest

from my_coding_agent.cli import build_llm_from_env


class DummyLLM:
    pass


def test_network_settings_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("AGENT_TIMEOUT", "17")
    monkeypatch.setenv("AGENT_MAX_RETRIES", "2")

    # LLM construction uses installed OpenAI-compatible SDK but does not make a network request.
    llm = build_llm_from_env()
    assert llm.config.timeout == 17
    assert llm.config.max_retries == 2


def test_invalid_network_setting_is_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("AGENT_TIMEOUT", "not-an-int")
    with pytest.raises(RuntimeError, match="AGENT_TIMEOUT must be an integer"):
        build_llm_from_env()
