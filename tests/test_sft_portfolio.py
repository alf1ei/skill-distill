import json
from pathlib import Path

from skill_distill.cli import main
from skill_distill.models import Conversation, Message
from skill_distill.portfolio import evidence_for, load_skills, render_portfolio, render_profile_readme
from skill_distill.sft import split, to_examples, validate_row

ROOT = Path(__file__).resolve().parents[1]


def test_to_examples_merges_and_ends_with_assistant():
    c = Conversation("1", "t", [Message("assistant", "开场白会被丢掉"), Message("user", "问1"), Message("user", "补充"),
                                Message("assistant", "回答一，足够长的内容"), Message("user", "问2"), Message("assistant", "回答二，足够长的内容")])
    rows = to_examples([c])
    assert len(rows) == 2
    for r in rows:
        assert validate_row(r) == []
        assert r["messages"][0]["role"] == "system"
    assert rows[0]["messages"][1]["content"] == "问1\n补充"


def test_validate_row_detects_errors():
    assert validate_row({"messages": [{"role": "assistant", "content": "x"}]})
    assert validate_row({"messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}, {"role": "assistant", "content": "z"}]})


def test_split_deterministic():
    rows = [{"i": i} for i in range(20)]
    assert split(rows) == split(rows)
    tr, ev = split(rows)
    assert len(ev) == 2 and len(tr) == 18


def test_example_sft_files_are_valid():
    for name in ("train.example.jsonl", "eval.example.jsonl"):
        for line in (ROOT / "finetune" / "data" / name).read_text("utf-8").splitlines():
            assert validate_row(json.loads(line)) == []


def test_sft_cli(tmp_path):
    assert main(["sft", str(ROOT / "examples" / "chats"), "-o", str(tmp_path)]) == 0
    n = sum(1 for _ in open(tmp_path / "train.jsonl", encoding="utf-8")) + sum(1 for _ in open(tmp_path / "eval.jsonl", encoding="utf-8"))
    assert n >= 3


REPOS = [{"name": "lottery-backtest", "html_url": "https://github.com/alf1ei/lottery-backtest", "description": "d", "language": "Python", "topics": ["lottery"], "fork": False},
         {"name": "forked", "html_url": "u", "description": "", "language": "Python", "topics": [], "fork": True}]


def test_portfolio_render_honest():
    data = load_skills(ROOT / "skills.yaml")
    py = next(s for s in data["skills"] if s["name"] == "Python")
    assert [r["name"] for r in evidence_for(py, REPOS)] == ["lottery-backtest"]
    md = render_portfolio(data, REPOS, "2026-10-30")
    assert "暂无公开证据" in md and "forked" not in md
    readme = render_profile_readme(data, REPOS)
    assert "lottery-backtest" in readme and "正在学" in readme


def test_bad_level_rejected(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("skills: [{name: x, level: 9}]", encoding="utf-8")
    try:
        load_skills(p)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_exclude_repos():
    data = {"owner": "x", "exclude_repos": ["lottery-backtest"], "skills": []}
    assert "lottery-backtest" not in render_portfolio(data, REPOS, "2026-10-30")
