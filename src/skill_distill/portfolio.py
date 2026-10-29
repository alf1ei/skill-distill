"""技能清单 + GitHub 公开仓库 → 能力矩阵 docs/portfolio.md 和主页 README 草稿。"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import yaml

LEVELS = {
    0: "了解（听过/看过）",
    1: "入门（跟着教程做过）",
    2: "能用（独立完成过小项目）",
    3: "熟练（在工作或多个项目中稳定使用）",
    4: "精通（能指导别人、处理疑难）",
}


def load_skills(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for s in data.get("skills", []):
        lvl = s.get("level", 0)
        if lvl not in LEVELS:
            raise ValueError(f"skill {s.get('name')}: level must be 0-4, got {lvl}")
    return data


def fetch_repos(user: str, token: str | None = None, timeout: int = 20) -> list[dict]:
    req = urllib.request.Request(
        f"https://api.github.com/users/{user}/repos?type=owner&per_page=100&sort=updated",
        headers={"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        repos = json.loads(r.read().decode("utf-8"))
    return [
        {k: x.get(k) for k in ("name", "html_url", "description", "language", "topics", "stargazers_count", "pushed_at", "fork")}
        for x in repos if not x.get("private")
    ]


def evidence_for(skill: dict, repos: list[dict]) -> list[dict]:
    names = {n.lower() for n in skill.get("repos", [])}
    tags = {t.lower() for t in skill.get("match", [])}
    out = []
    for r in repos:
        if r.get("fork"):
            continue
        hay = {(r.get("language") or "").lower(), *[t.lower() for t in (r.get("topics") or [])]}
        if r["name"].lower() in names or (tags & hay):
            out.append(r)
    return out


def visible_repos(data: dict, repos: list[dict]) -> list[dict]:
    excl = {n.lower() for n in data.get("exclude_repos", [])}
    return [r for r in repos if not r.get("fork") and r["name"].lower() not in excl]


def render_portfolio(data: dict, repos: list[dict], today: str) -> str:
    repos = visible_repos(data, repos)
    lines = [
        f"# 能力矩阵（{data.get('owner', '')}）",
        "",
        f"> 生成日期：{today}。等级为本人自评（0–4），\"证据\"列只引用公开仓库；没有证据的等级请按自评理解。",
        "",
        "| 领域 | 技能 | 等级 | 证据 | 备注 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in data.get("skills", []):
        ev = evidence_for(s, repos)
        ev_s = "<br>".join(f"[{r['name']}]({r['html_url']})" for r in ev) or "暂无公开证据"
        lines.append(f"| {s.get('area', '')} | {s['name']} | {s['level']} · {LEVELS[s['level']]} | {ev_s} | {s.get('note', '')} |")
    lines += ["", "## 等级说明", ""] + [f"- **{k}** {v}" for k, v in LEVELS.items()]
    lines += ["", "## 公开仓库", ""]
    for r in repos:
        if not r.get("fork"):
            lines.append(f"- [{r['name']}]({r['html_url']})：{r.get('description') or '（无描述）'}（{r.get('language') or '—'}）")
    if not repos:
        lines.append("- 暂无")
    return "\n".join(lines) + "\n"


def render_profile_readme(data: dict, repos: list[dict]) -> str:
    p = data.get("profile", {})
    repos = visible_repos(data, repos)
    top = [s for s in data.get("skills", []) if s.get("level", 0) >= 2]
    learning = [s for s in data.get("skills", []) if s.get("level", 0) < 2]
    lines = [f"# 你好，我是 {data.get('owner', '')} 👋", ""]
    if p.get("intro"):
        lines += [p["intro"], ""]
    if p.get("intro_en"):
        lines += [f"*{p['intro_en']}*", ""]
    if top:
        lines += ["## 能用 / 熟练", ""] + [f"- {s['name']}" for s in top] + [""]
    if learning:
        lines += ["## 正在学", ""] + [f"- {s['name']}" for s in learning] + [""]
    lines += ["## 项目", ""]
    for r in repos:
        if not r.get("fork") and r["name"] != data.get("owner"):
            lines.append(f"- **[{r['name']}]({r['html_url']})**：{r.get('description') or ''}")
    if p.get("contact"):
        lines += ["", "## 联系", "", p["contact"]]
    return "\n".join(lines).rstrip() + "\n"
