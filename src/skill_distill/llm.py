"""可选：调用任意 OpenAI 兼容接口起草 SKILL.md。默认关闭。

环境变量：
- ``SKILL_DISTILL_LLM=1``           打开
- ``OPENAI_BASE_URL``               例如 https://api.deepseek.com/v1 、http://localhost:11434/v1
- ``OPENAI_API_KEY``                本地模型可留空
- ``SKILL_DISTILL_MODEL``           模型名，默认 gpt-4o-mini
"""
from __future__ import annotations

import json
import os
import urllib.request

PROMPT = """你是个人技能蒸馏助手。根据下面已经脱敏的聊天片段，写一份 SKILL.md（简体中文）。
要求：
1. 开头 YAML frontmatter，只含 name（小写短横线）和 description（一句话说明"什么时候用"）。
2. 小节：## 步骤、## 偏好、## 坑点、## 来源。
3. 用户原话里的内容标注（本人确认），你的归纳标注（推断）。不得编造聊天里没有的事实。
片段：
{material}
"""


def enabled() -> bool:
    return os.environ.get("SKILL_DISTILL_LLM", "").lower() in {"1", "true", "yes"}


def draft_skill(material: str, timeout: int = 60) -> str | None:
    if not enabled():
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({
        "model": os.environ.get("SKILL_DISTILL_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": PROMPT.format(material=material[:12000])}],
        "temperature": 0.2,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # 网络/鉴权失败时回退到模板
        print(f"[llm] 调用失败，回退到模板：{exc}")
        return None
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    return text if text.lstrip().startswith("---") else None
