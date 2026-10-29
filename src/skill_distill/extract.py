"""从一个话题簇里抽取偏好、坑点、步骤（确定性规则）。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Conversation

PREF = re.compile(r"(喜欢|希望|偏好|倾向|想要|最好|重视|关注|prefer|i like|i want)", re.I)
NEG = re.compile(r"(不要|别|不喜欢|不希望|不想|避免|讨厌|don't|do not|never)", re.I)
FAIL = re.compile(r"(失败|没用|不行|报错|错误|又|还是不|没成功|failed|error|didn't work)", re.I)
STEP = re.compile(r"^\s*(?:\d+[.)、]|[-*•]|第[一二三四五六七八九十]+步|先|然后|再|最后)\s*")
_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def sentences(text: str) -> list[str]:
    return [s.strip(" -*•\n") for s in _SPLIT.split(text) if len(s.strip()) >= 4]


@dataclass
class Item:
    text: str
    origin: str  # "user-confirmed"（用户原话）| "ai-inference"（来自 AI 回复或规则推断）
    date: str | None = None
    source: str = ""


@dataclass
class Extracted:
    preferences: list[Item] = field(default_factory=list)
    pitfalls: list[Item] = field(default_factory=list)
    steps: list[Item] = field(default_factory=list)
    sources: list[tuple[str, str, str | None]] = field(default_factory=list)  # (title, source, date)


def _add(bucket: list[Item], item: Item, seen: set[str], limit: int) -> None:
    key = re.sub(r"\W+", "", item.text)
    if key and key not in seen and len(bucket) < limit:
        seen.add(key)
        bucket.append(item)


def extract(convs: list[Conversation], limit: int = 12) -> Extracted:
    ex = Extracted()
    seen: set[str] = set()
    for c in convs:
        ex.sources.append((c.title, c.source, c.date))
        for m in c.messages:
            date = m.timestamp or c.date
            if m.role == "user":
                for s in sentences(m.content):
                    s1 = s[:200]
                    if re.search(r"\[(PHONE|EMAIL|ID_NUMBER|BANK_CARD|TOKEN|PRIVATE_KEY)\]", s1):
                        continue  # 含脱敏占位符的句子没有复用价值
                    if NEG.search(s) or FAIL.search(s):
                        _add(ex.pitfalls, Item(s1, "user-confirmed", date, c.title), seen, limit)
                    elif PREF.search(s):
                        _add(ex.preferences, Item(s1, "user-confirmed", date, c.title), seen, limit)
            elif m.role == "assistant":
                for line in m.content.splitlines():
                    if STEP.match(line) and len(line.strip()) > 6:
                        _add(ex.steps, Item(STEP.sub("", line).strip()[:200], "ai-inference", date, c.title), seen, limit)
    return ex
