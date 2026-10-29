from pathlib import Path

from skill_distill.cli import main
from skill_distill.cluster import cluster_conversations, tokenize
from skill_distill.models import Conversation, Message
from skill_distill.render import parse_frontmatter

EX = Path(__file__).resolve().parents[1] / "examples" / "chats"


def conv(i, text):
    return Conversation(str(i), f"c{i}", [Message("user", text), Message("assistant", "1. 先这样做一下")])


def test_tokenize_mixed():
    t = tokenize("CS2 鼠标选购")
    assert "cs2" in t and "鼠标" in t


def test_cluster_groups_similar_topics_deterministically():
    convs = [conv(0, "双色球 历史数据 回测 随机基准"), conv(1, "鼠标 趴握 手长 CS2"),
             conv(2, "大乐透 双色球 回测 样本外"), conv(3, "CS2 鼠标 延迟 人体工学")]
    a = cluster_conversations(convs)
    b = cluster_conversations(convs)
    groups = sorted(sorted(c.members) for c in a)
    assert groups == [[0, 2], [1, 3]]
    assert groups == sorted(sorted(c.members) for c in b)


def test_distill_cli_writes_skill_files(tmp_path, monkeypatch):
    monkeypatch.delenv("SKILL_DISTILL_LLM", raising=False)
    assert main(["--today", "2026-10-30", "distill", str(EX), "-o", str(tmp_path)]) == 0
    files = list(tmp_path.glob("*/SKILL.md"))
    assert files and (tmp_path / "INDEX.md").exists()
    all_text = ""
    for f in files:
        text = f.read_text("utf-8")
        fm = parse_frontmatter(text)
        assert fm["name"] and fm["description"].startswith("当话题涉及")
        for sec in ("## 步骤", "## 偏好", "## 坑点", "## 来源"):
            assert sec in text
        all_text += text
    assert "本人确认" in all_text and "推断" in all_text
    assert "13812345678" not in all_text and "demo@example.com" not in all_text


def test_llm_failure_falls_back(monkeypatch):
    from skill_distill import llm
    monkeypatch.setenv("SKILL_DISTILL_LLM", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    assert llm.draft_skill("x", timeout=1) is None
