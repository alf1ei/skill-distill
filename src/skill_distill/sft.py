"""把清洗后的会话转成 SFT JSONL（chat 格式：{"messages": [...]}）。"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .models import Conversation

DEFAULT_SYSTEM = "你是用户的个人助手，按用户的偏好用简体中文回答：先结论，再依据、风险和下一步；区分事实与推测。"


def to_examples(convs: list[Conversation], system: str | None = DEFAULT_SYSTEM,
                max_turns: int = 12, min_assistant_chars: int = 10) -> list[dict]:
    """每个会话切成若干样本：以 assistant 回复结尾，保留前面最多 max_turns 条上下文。"""
    out: list[dict] = []
    for c in convs:
        msgs = [m for m in c.messages if m.role in ("user", "assistant")]
        # 合并连续同角色消息
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m.role:
                merged[-1]["content"] += "\n" + m.content
            else:
                merged.append({"role": m.role, "content": m.content})
        while merged and merged[0]["role"] != "user":
            merged.pop(0)
        for i, m in enumerate(merged):
            if m["role"] == "assistant" and len(m["content"]) >= min_assistant_chars:
                ctx = merged[max(0, i + 1 - max_turns): i + 1]
                while ctx and ctx[0]["role"] != "user":
                    ctx = ctx[1:]
                if not ctx:
                    continue
                ex = ([{"role": "system", "content": system}] if system else []) + [dict(x) for x in ctx]
                out.append({"messages": ex, "meta": {"conversation": c.id, "source": c.source}})
    return out


def split(examples: list[dict], eval_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict], list[dict]]:
    ex = list(examples)
    random.Random(seed).shuffle(ex)
    n_eval = max(1, int(len(ex) * eval_ratio)) if len(ex) > 1 else 0
    return ex[n_eval:], ex[:n_eval]


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_row(row: dict) -> list[str]:
    errs = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["messages missing"]
    roles = [m.get("role") for m in msgs]
    if roles[-1] != "assistant":
        errs.append("last message must be assistant")
    body = [r for r in roles if r != "system"]
    if not body or body[0] != "user":
        errs.append("first non-system message must be user")
    for a, b in zip(body, body[1:]):
        if a == b:
            errs.append("roles must alternate")
            break
    return errs
