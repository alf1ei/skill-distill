"""SKILL.md 模板渲染（确定性回退）。"""
from __future__ import annotations

import re
from datetime import date

from .extract import Extracted, Item

LABEL = {"user-confirmed": "本人确认", "ai-inference": "推断", "third-party-summary": "第三方总结"}


def slugify(words: list[str], fallback: str) -> str:
    ascii_words = [re.sub(r"[^a-z0-9]+", "-", w.lower()).strip("-") for w in words if re.match(r"^[A-Za-z0-9]", w)]
    slug = "-".join(w for w in ascii_words if w)[:40].strip("-")
    return slug or fallback


def _fmt(items: list[Item], empty: str) -> str:
    if not items:
        return f"- {empty}\n"
    return "".join(f"- {it.text}（{LABEL.get(it.origin, it.origin)}{'，' + it.date if it.date else ''}）\n" for it in items)


def render_skill(name: str, description: str, ex: Extracted, keywords: list[str], today: str | None = None) -> str:
    today = today or date.today().isoformat()
    desc = description.replace('"', "'")
    steps = "".join(f"{i}. {it.text}（{LABEL[it.origin]}）\n" for i, it in enumerate(ex.steps, 1)) or "1. 待补充（聊天中未找到明确步骤）\n"
    src = "".join(f"- {t}（{s}{'，' + d if d else ''}）\n" for t, s, d in ex.sources)
    return (
        f"---\nname: {name}\ndescription: \"{desc}\"\n---\n\n"
        f"# {name}\n\n"
        f"> 自动蒸馏于 {today}。关键词：{'、'.join(keywords) or '无'}。\n"
        f"> 标注说明：**本人确认** = 用户原话；**推断** = 来自 AI 回复或规则归纳，使用前请核实。\n\n"
        f"## 步骤\n\n{steps}\n"
        f"## 偏好\n\n{_fmt(ex.preferences, '待补充')}\n"
        f"## 坑点\n\n{_fmt(ex.pitfalls, '暂无记录')}\n"
        f"## 来源\n\n{src}"
    )


def parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip().strip('"')
    return out
