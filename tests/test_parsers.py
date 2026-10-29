import json
from pathlib import Path

from skill_distill.parsers import load_path, normalize_role, parse_chatgpt_export, parse_generic_json, parse_text

EX = Path(__file__).resolve().parents[1] / "examples" / "chats"


def test_chatgpt_follows_current_node_and_skips_system():
    convs = parse_chatgpt_export(json.loads((EX / "chatgpt_conversations.example.json").read_text("utf-8")))
    assert len(convs) == 2
    c = convs[0]
    assert [m.role for m in c.messages] == ["user", "assistant", "user"]
    assert c.date == "2025-10-30"
    assert c.source == "chatgpt"


def test_chatgpt_without_current_node_sorted_by_time():
    convs = parse_chatgpt_export(json.loads((EX / "chatgpt_conversations.example.json").read_text("utf-8")))
    assert [m.role for m in convs[1].messages] == ["user", "assistant"]


def test_generic_json_variants():
    a = parse_generic_json([{"role": "human", "content": "你好呀"}, {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}])
    assert [m.role for m in a[0].messages] == ["user", "assistant"]
    b = parse_generic_json({"title": "t", "messages": [{"sender": "我", "text": "测试一下"}]})
    assert b[0].title == "t" and b[0].messages[0].role == "user"
    c = parse_generic_json([{"messages": [{"role": "user", "content": "x1"}]}, {"chat_messages": [{"sender": "human", "text": "x2"}]}])
    assert len(c) == 2


def test_text_role_prefixes():
    text = "我：问题一\n继续说\n豆包：回答\n**User:** 第二问\n## Assistant\n第二答"
    c = parse_text(text)[0]
    assert [m.role for m in c.messages] == ["user", "assistant", "user", "assistant"]
    assert c.messages[0].content == "问题一\n继续说"
    assert c.messages[3].content == "第二答"


def test_text_without_roles_is_single_user_message():
    c = parse_text("纯笔记，没有角色前缀")[0]
    assert len(c.messages) == 1 and c.messages[0].role == "user"


def test_normalize_role():
    assert normalize_role("ChatGPT") == "assistant"
    assert normalize_role("用户") == "user"
    assert normalize_role("随便") is None


def test_load_dir():
    convs = load_path(EX)
    assert {c.source for c in convs} == {"chatgpt", "markdown", "text"}
