from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str | None = None  # ISO date (YYYY-MM-DD) if known


@dataclass
class Conversation:
    id: str
    title: str
    messages: list[Message] = field(default_factory=list)
    source: str = "unknown"  # chatgpt | json | markdown | text
    date: str | None = None

    def user_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "user")

    def all_text(self) -> str:
        return "\n".join(m.content for m in self.messages)
