#!/usr/bin/env python3
"""LoRA SFT 训练脚本（transformers + peft）。本仓库没有实际运行过训练，请在有 GPU 的机器上使用。

用法：python finetune/train_lora.py --config finetune/lora_config.yaml [--dry-run]
--dry-run 只检查数据和配置，不加载模型。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from skill_distill.sft import validate_row  # noqa: E402


def load_rows(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, r in enumerate(rows):
        errs = validate_row(r)
        if errs:
            raise SystemExit(f"{path}:{i + 1}: {errs}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune/lora_config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_rows, eval_rows = load_rows(cfg["train_file"]), load_rows(cfg["eval_file"])
    if any(r.get("meta", {}).get("synthetic") for r in train_rows):
        print("⚠️ 正在使用示例（合成）数据，只能用来验证流程，训练结果没有意义。")
    print(f"train={len(train_rows)} eval={len(eval_rows)} base={cfg['base_model']}")
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(row: dict) -> dict:
        msgs = row["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"][: cfg["max_seq_len"]]
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]  # 只对最后一条 assistant 计算 loss
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels[: len(ids)]}

    ds_train = Dataset.from_list([{"messages": r["messages"]} for r in train_rows]).map(encode, remove_columns=["messages"])
    ds_eval = Dataset.from_list([{"messages": r["messages"]} for r in eval_rows]).map(encode, remove_columns=["messages"])

    kw = {}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16, **kw)
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=cfg["epochs"], learning_rate=float(cfg["lr"]),
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["grad_accum"],
        warmup_ratio=cfg["warmup_ratio"], bf16=cfg.get("bf16", False), fp16=cfg.get("fp16", False),
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        seed=cfg["seed"], report_to=[])
    Trainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_eval,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True)).train()
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
