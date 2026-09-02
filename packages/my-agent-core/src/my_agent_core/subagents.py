"""Read-only subagent definitions loaded from configured Markdown files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Subagent:
    name: str
    description: str
    content: str
    file_path: Path
    max_turns: int | None = None
    tools: tuple[str, ...] | None = None


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    header = "---\n"
    if not text.startswith(header):
        return {}, text
    end = text.find("\n---\n", len(header))
    if end == -1:
        return {}, text
    block = text[len(header) : end]
    body = text[end + len("\n---\n") :].strip()
    try:
        fields = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}, text
    return (fields if isinstance(fields, dict) else {}), body


def _split_csv(value: object) -> tuple[str, ...] | None:
    if isinstance(value, list):
        parts = tuple(str(item).strip() for item in value if str(item).strip())
        return parts or None
    if not isinstance(value, str):
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or None


def _parse_max_turns(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


class SubagentManager:
    """Discover the subagents explicitly configured by the product layer."""

    def __init__(self, dirs: Sequence[str | Path] | None = None):
        self.subagents: dict[str, Subagent] = {}
        for root in dirs or []:
            self._discover_dir(Path(root))

    def _discover_dir(self, root: Path) -> None:
        if not root.is_dir():
            return
        for child in sorted(root.iterdir()):
            if child.is_dir() or child.suffix.lower() != ".md":
                continue
            if child.name.startswith(".") or child.name.lower() == "readme.md":
                continue
            subagent = self._load_one(child)
            if subagent is not None:
                self.subagents[subagent.name] = subagent

    def _load_one(self, path: Path) -> Subagent | None:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            return None
        metadata, body = _parse_frontmatter(text)
        description = str(metadata.get("description") or "").strip()
        if not description:
            return None
        name = str(metadata.get("name") or "").strip() or path.stem
        return Subagent(
            name=name,
            description=description,
            content=body,
            file_path=path,
            max_turns=_parse_max_turns(metadata.get("maxTurns")),
            tools=_split_csv(metadata.get("tools")),
        )

    def get(self, name: str) -> Subagent | None:
        return self.subagents.get(name)

    def list(self) -> list[Subagent]:
        return list(self.subagents.values())

    def __len__(self) -> int:
        return len(self.subagents)

    def __bool__(self) -> bool:
        return bool(self.subagents)

    def format_prompt(self) -> str:
        if not self.subagents:
            return ""
        parts = ["<available_agents>"]
        for subagent in self.subagents.values():
            parts.extend(
                [
                    "  <agent>",
                    f"    <name>{subagent.name}</name>",
                    f"    <description>{subagent.description}</description>",
                    "  </agent>",
                ]
            )
        parts.append("</available_agents>")
        return "\n".join(parts)
