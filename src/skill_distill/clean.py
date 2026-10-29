"""清洗、隐私脱敏和去重。"""
from __future__ import annotations

import hashlib
import re

from .models import Conversation, Message

# 顺序重要：先长后短，先 token 后数字
REDACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
    ("TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{16,}|xox[abpr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})\b")),
    ("TOKEN", re.compile(r"(?i)\b(?:bearer)\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("TOKEN", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ID_NUMBER", re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")),
    ("BANK_CARD", re.compile(r"(?<!\d)(?:62|4\d|5[1-5])\d{14,17}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d[- ]?\d{4}[- ]?\d{4}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)0\d{2,3}-\d{7,8}(?!\d)")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pat in REDACTION_PATTERNS:
        text, n = pat.subn(f"[{label}]", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


_WS = re.compile(r"[ \t\u3000]+")
_MANY_NL = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u200b", "")
    text = "\n".join(_WS.sub(" ", ln).rstrip() for ln in text.split("\n"))
    return _MANY_NL.sub("\n\n", text).strip()


def _fingerprint(text: str) -> str:
    key = re.sub(r"\W+", "", text.lower())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def clean_conversations(convs: list[Conversation], min_chars: int = 2) -> tuple[list[Conversation], dict[str, int]]:
    """脱敏 + 规范化 + 去重（消息级去重和整段会话去重）。返回 (结果, 统计)。"""
    stats: dict[str, int] = {"conversations_in": len(convs), "duplicate_messages": 0, "duplicate_conversations": 0}
    seen_convs: set[str] = set()
    out: list[Conversation] = []
    for c in convs:
        seen_msgs: set[str] = set()
        msgs: list[Message] = []
        for m in c.messages:
            text = normalize(m.content)
            text, counts = redact(text)
            for k, v in counts.items():
                stats[f"redacted_{k}"] = stats.get(f"redacted_{k}", 0) + v
            if len(text) < min_chars:
                continue
            fp = m.role + _fingerprint(text)
            if fp in seen_msgs:
                stats["duplicate_messages"] += 1
                continue
            seen_msgs.add(fp)
            msgs.append(Message(m.role, text, m.timestamp))
        if not msgs:
            continue
        cfp = _fingerprint("".join(x.content for x in msgs))
        if cfp in seen_convs:
            stats["duplicate_conversations"] += 1
            continue
        seen_convs.add(cfp)
        title, _ = redact(c.title)
        out.append(Conversation(c.id, title, msgs, c.source, c.date))
    stats["conversations_out"] = len(out)
    return out, stats
