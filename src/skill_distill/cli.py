"""命令行入口：skill-distill {distill,sft,portfolio}"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from . import llm
from .clean import clean_conversations
from .cluster import cluster_conversations
from .extract import extract
from .parsers import load_path
from .render import render_skill, slugify
from .sft import DEFAULT_SYSTEM, split, to_examples, validate_row, write_jsonl


def _load(inputs: list[str]):
    convs = []
    for p in inputs:
        convs.extend(load_path(p))
    return clean_conversations(convs)


def cmd_distill(a: argparse.Namespace) -> int:
    convs, stats = _load(a.inputs)
    clusters = cluster_conversations(convs, threshold=a.threshold)
    out = Path(a.out)
    index = ["# Skills 索引", "", "| skill | 关键词 | 会话数 |", "| --- | --- | --- |"]
    used: set[str] = set()
    for cl in clusters:
        members = [convs[i] for i in cl.members]
        if len(members) < a.min_size:
            continue
        slug = f"topic-{cl.id + 1:02d}" + (("-" + slugify(cl.keywords[:2], "")) if slugify(cl.keywords[:2], "") else "")
        base, n = slug, 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        ex = extract(members)
        desc = f"当话题涉及 {'、'.join(cl.keywords[:4])} 时使用（自动聚类，含 {len(members)} 段会话）"
        text = None
        if llm.enabled():
            material = "\n\n".join(f"# {c.title}\n{c.all_text()}" for c in members)
            text = llm.draft_skill(material)
        text = text or render_skill(slug, desc, ex, cl.keywords, a.today)
        (out / slug).mkdir(parents=True, exist_ok=True)
        (out / slug / "SKILL.md").write_text(text, encoding="utf-8")
        index.append(f"| [{slug}]({slug}/SKILL.md) | {'、'.join(cl.keywords[:5])} | {len(members)} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "clusters": len(clusters), "skills_written": len(used), "out": str(out)}, ensure_ascii=False))
    return 0


def cmd_sft(a: argparse.Namespace) -> int:
    convs, stats = _load(a.inputs)
    rows = to_examples(convs, system=None if a.no_system else (a.system or DEFAULT_SYSTEM))
    bad = [e for r in rows for e in validate_row(r)]
    if bad:
        print("invalid rows:", bad[:5])
        return 1
    train, ev = split(rows, a.eval_ratio)
    out = Path(a.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(ev, out / "eval.jsonl")
    print(json.dumps({"stats": stats, "train": len(train), "eval": len(ev)}, ensure_ascii=False))
    return 0


def cmd_portfolio(a: argparse.Namespace) -> int:
    from .portfolio import fetch_repos, load_skills, render_portfolio, render_profile_readme
    data = load_skills(a.skills)
    if a.repos_json:
        repos = json.loads(Path(a.repos_json).read_text(encoding="utf-8"))
    else:
        repos = fetch_repos(a.user or data.get("owner"), os.environ.get("GITHUB_TOKEN"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "portfolio.md").write_text(render_portfolio(data, repos, a.today), encoding="utf-8")
    (out / "profile-README.md").write_text(render_profile_readme(data, repos), encoding="utf-8")
    print(f"wrote {out/'portfolio.md'} and {out/'profile-README.md'} ({len(repos)} public repos)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="skill-distill", description="聊天记录 → SKILL.md / SFT 数据 / 能力矩阵")
    ap.add_argument("--today", default=date.today().isoformat())
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distill", help="聊天记录 → 按话题聚类 → SKILL.md")
    d.add_argument("inputs", nargs="+")
    d.add_argument("-o", "--out", default="out/skills")
    d.add_argument("--threshold", type=float, default=0.12)
    d.add_argument("--min-size", type=int, default=1)
    d.set_defaults(func=cmd_distill)

    s = sub.add_parser("sft", help="聊天记录 → SFT JSONL（train/eval）")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--out", default="out/sft")
    s.add_argument("--eval-ratio", type=float, default=0.1)
    s.add_argument("--system", default=None)
    s.add_argument("--no-system", action="store_true")
    s.set_defaults(func=cmd_sft)

    p = sub.add_parser("portfolio", help="skills.yaml + GitHub 公开仓库 → docs/portfolio.md + profile-README.md")
    p.add_argument("--skills", default="skills.yaml")
    p.add_argument("--user", default=None)
    p.add_argument("--repos-json", default=None, help="离线：用保存的仓库 JSON 代替 GitHub API")
    p.add_argument("-o", "--out", default="docs")
    p.set_defaults(func=cmd_portfolio)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
