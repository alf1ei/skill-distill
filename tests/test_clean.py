from skill_distill.clean import clean_conversations, redact
from skill_distill.models import Conversation, Message


def test_redact_all_kinds():
    s = ("电话 13812345678 或 +86 139-1234-5678，座机 0934-1234567，邮箱 a.b@x.com，"
         "身份证 62100119900101123X，卡号 6222021234567890123，token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123 "
         "sk-abcdefghijklmnop1234 Bearer abcdefghijklmnopqrstuvwxyz")
    out, counts = redact(s)
    for raw in ["13812345678", "139-1234-5678", "0934-1234567", "a.b@x.com", "62100119900101123X", "6222021234567890123", "ghp_", "sk-abc", "abcdefghijklmnopqrstuvwxyz"]:
        assert raw not in out, raw
    assert counts["PHONE"] == 3 and counts["EMAIL"] == 1 and counts["ID_NUMBER"] == 1
    assert counts["TOKEN"] == 3 and counts["BANK_CARD"] == 1


def test_redact_keeps_normal_numbers():
    out, counts = redact("手长 16.5cm，预算 3000 元，2026-10-30，双色球 3521 期")
    assert counts == {} and "3521" in out


def test_dedupe_messages_and_conversations():
    m = [Message("user", "同一句话"), Message("user", "同一句话 "), Message("assistant", "回答")]
    c1 = Conversation("1", "t", m)
    c2 = Conversation("2", "t", list(m))
    out, stats = clean_conversations([c1, c2])
    assert len(out) == 1 and len(out[0].messages) == 2
    assert stats["duplicate_conversations"] == 1 and stats["duplicate_messages"] >= 1


def test_clean_redacts_title():
    out, _ = clean_conversations([Conversation("1", "找 13812345678", [Message("user", "hello world")])])
    assert "13812345678" not in out[0].title
