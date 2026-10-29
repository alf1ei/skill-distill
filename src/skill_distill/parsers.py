"""Parsers for chat exports.

- ChatGPT 官方导出 ``conversations.json``（mapping 树结构）
- 通用 JSON：``[{"role":..., "content":...}]``、``{"messages": [...]}`` 或会话列表
- Markdown / TXT：豆包、Claude 等复制出来的文本，按角色前缀切分
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Conversation, Message

USER_ALIASES = {"user", "human", "me", "我", "用户", "你"}
ASSISTANT_ALIASES = {
    "assistant", "ai", "bot", "gpt", "chatgpt", "claude", "豆包", "doubao",
    "kimi", "deepseek", "gemini", "grok", "助手", "model",
}
SYSTEM_ALIASES = {"system", "系统"}


def normalize_role(raw: str | None) -> str | None:
    if raw is None:
        return None
    r = str(raw).strip().strip("*#>：: ").lower()
    if r in USER_ALIASES:
        return "user"
    if r in ASSISTANT_ALIASES:
        return "assistant"
    if r in SYSTEM_ALIASES:
        return "system"
    return None


def _ts_to_date(ts: Any) -> str | None:
    if ts in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        m = re.match(r"\d{4}-\d{2}-\d{2}", str(ts))
        return m.group(0) if m else None


# ---------------------------------------------------------------- ChatGPT

def _chatgpt_parts_to_text(content: dict[str, Any] | None) -> str:
    if not content:
        return ""
    parts = content.get("parts")
    if parts is None and "text" in content:
        parts = [content["text"]]
    out = []
    for p in parts or []:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict) and isinstance(p.get("text"), str):
            out.append(p["text"])
    return "\n".join(s for s in out if s).strip()


def parse_chatgpt_conversation(conv: dict[str, Any], index: int = 0) -> Conversation:
    mapping: dict[str, Any] = conv.get("mapping") or {}
    # 沿 current_node 回溯得到当前分支；没有就按 create_time 排序全部节点
    chain: list[dict[str, Any]] = []
    node_id = conv.get("current_node")
    seen = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        chain.append(mapping[node_id])
        node_id = mapping[node_id].get("parent")
    chain.reverse()
    if not chain:
        chain = sorted(
            mapping.values(),
            key=lambda n: ((n.get("message") or {}).get("create_time") or 0),
        )
    messages = []
    for node in chain:
        msg = node.get("message")
        if not msg:
            continue
        role = normalize_role((msg.get("author") or {}).get("role"))
        if role not in ("user", "assistant"):
            continue
        text = _chatgpt_parts_to_text(msg.get("content"))
        if not text:
            continue
        messages.append(Message(role, text, _ts_to_date(msg.get("create_time"))))
    return Conversation(
        id=str(conv.get("id") or conv.get("conversation_id") or f"chatgpt-{index}"),
        title=conv.get("title") or "untitled",
        messages=messages,
        source="chatgpt",
        date=_ts_to_date(conv.get("create_time")),
    )


def parse_chatgpt_export(data: Any) -> list[Conversation]:
    if isinstance(data, dict):
        data = [data]
    return [parse_chatgpt_conversation(c, i) for i, c in enumerate(data) if isinstance(c, dict)]


# ---------------------------------------------------------------- generic JSON

def _msgs_from_list(items: Iterable[Any]) -> list[Message]:
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        role = normalize_role(it.get("role") or it.get("sender") or it.get("author") or it.get("from"))
        text = it.get("content") if it.get("content") is not None else it.get("text")
        if isinstance(text, list):  # Claude/OpenAI 风格 content blocks
            text = "\n".join(b.get("text", "") for b in text if isinstance(b, dict))
        if role and isinstance(text, str) and text.strip():
            out.append(Message(role, text.strip(), _ts_to_date(it.get("timestamp") or it.get("created_at"))))
    return out


def parse_generic_json(data: Any, name: str = "json") -> list[Conversation]:
    if isinstance(data, list) and data and isinstance(data[0], dict) and "mapping" in data[0]:
        return parse_chatgpt_export(data)
    if isinstance(data, dict) and "mapping" in data:
        return parse_chatgpt_export(data)
    if isinstance(data, dict):
        msgs = data.get("messages") or data.get("chat_messages") or []
        return [Conversation(str(data.get("id") or name), data.get("title") or data.get("name") or name,
                             _msgs_from_list(msgs), "json", _ts_to_date(data.get("created_at")))]
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and ("messages" in data[0] or "chat_messages" in data[0]):
            convs: list[Conversation] = []
            for i, d in enumerate(data):
                convs.extend(parse_generic_json(d, f"{name}-{i}"))
            return convs
        return [Conversation(name, name, _msgs_from_list(data), "json")]
    return []


# ---------------------------------------------------------------- Markdown / TXT

_ROLE_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*|>\s*)?(?P<role>[A-Za-z\u4e00-\u9fff]{1,12})(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(?P<rest>.*)$"
)


def parse_text(text: str, name: str = "text", source: str = "text") -> list[Conversation]:
    """按 ``我：`` / ``豆包：`` / ``**User:**`` / ``## Assistant`` 等前缀切分。

    没有任何角色前缀时，整段视为一条 user 消息（例如用户自己写的偏好总结）。
    """
    messages: list[Message] = []
    role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if role and "".join(buf).strip():
            messages.append(Message(role, "\n".join(buf).strip()))

    for line in text.splitlines():
        m = _ROLE_LINE.match(line)
        header_only = re.match(r"^\s*#{1,6}\s*(?P<role>[A-Za-z\u4e00-\u9fff]{1,12})\s*$", line)
        new_role = None
        rest = ""
        if m and normalize_role(m.group("role")):
            new_role, rest = normalize_role(m.group("role")), m.group("rest")
        elif header_only and normalize_role(header_only.group("role")):
            new_role = normalize_role(header_only.group("role"))
        if new_role:
            flush()
            role, buf = new_role, ([rest] if rest else [])
        else:
            buf.append(line)
    flush()
    if not messages and text.strip():
        messages = [Message("user", text.strip())]
    return [Conversation(name, name, messages, source)]


def load_path(path: str | Path) -> list[Conversation]:
    p = Path(path)
    if p.is_dir():
        convs: list[Conversation] = []
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in {".json", ".md", ".markdown", ".txt"}:
                convs.extend(load_path(f))
        return convs
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return parse_generic_json(json.loads(raw), p.stem)
    src = "markdown" if p.suffix.lower() in {".md", ".markdown"} else "text"
    return parse_text(raw, p.stem, src)
