#!/usr/bin/env python3
"""评估：对 eval.jsonl 生成回答，对比基座与 LoRA 的
1) 字符 bigram F1（与参考回答的相似度）
2) 偏好规则命中率（是否"先结论"、是否区分事实/推测、是否给下一步）
3) 隐私泄漏检查（输出里是否出现手机号/邮箱等）

用法：python finetune/eval.py --config finetune/lora_config.yaml [--adapter checkpoints/persona-lora] [--offline]
--offline 不加载模型，把参考答案当作预测，只用于验证评估代码本身。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill_distill.clean import redact  # noqa: E402

RULES = {
    "先给结论": re.compile(r"^(【示例】)?\s*(结论|建议|先)"),
    "区分事实与推测": re.compile(r"(依据|证据|推测|推断|不确定|统计事实)"),
    "给出下一步": re.compile(r"(下一步|请先|告诉我|第一步)"),
}


def bigram_f1(pred: str, ref: str) -> float:
    bg = lambda s: Counter(s[i:i + 2] for i in range(len(s) - 1))
    p, r = bg(pred), bg(ref)
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def score(preds: list[str], refs: list[str]) -> dict:
    n = max(len(preds), 1)
    out = {"n": len(preds), "bigram_f1": round(sum(bigram_f1(p, r) for p, r in zip(preds, refs)) / n, 4)}
    for name, pat in RULES.items():
        out[f"rule:{name}"] = round(sum(bool(pat.search(p)) for p in preds) / n, 4)
    out["privacy_leaks"] = sum(bool(redact(p)[1]) for p in preds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/lora_config.yaml")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in Path(cfg["eval_file"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = [r["messages"][-1]["content"] for r in rows]
    if a.offline:
        print(json.dumps(score(refs, refs), ensure_ascii=False, indent=2))
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    preds = []
    for r in rows:
        ids = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True, return_tensors="pt").to(model.device)
        gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False)
        preds.append(tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))
    print(json.dumps(score(preds, refs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
