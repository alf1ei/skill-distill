# 人格微调脚手架（persona fine-tune）

> ⚠️ 本目录只提供脚手架，**作者没有实际运行过训练**。`data/*.example.jsonl` 是手写的**合成示例数据**，只用来验证流程。

## 流程

```bash
pip install -e ".[train]"
skill-distill sft path/to/your_chats/ -o out/sft          # 1. 清洗+脱敏 → train.jsonl / eval.jsonl
# 2. 修改 lora_config.yaml 里的 train_file / eval_file 指向 out/sft/
python finetune/train_lora.py --config finetune/lora_config.yaml --dry-run   # 只检查数据
python finetune/train_lora.py --config finetune/lora_config.yaml             # 3. 训练（需要 GPU）
python finetune/eval.py --config finetune/lora_config.yaml                    # 4a. 基座模型得分
python finetune/eval.py --config finetune/lora_config.yaml --adapter checkpoints/persona-lora  # 4b. LoRA 得分
```

## 硬件与数据量（经验估计，不是实测）

| 基座 | LoRA bf16 显存 | QLoRA 4bit 显存 | 说明 |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | ~6 GB | ~4 GB | 消费级显卡或免费 Colab T4 可试（T4 不支持 bf16，改 fp16） |
| Qwen2.5-1.5B-Instruct | ~12 GB | ~6 GB | 3060 12G / 4070 级别 |

- 数据：建议至少 **几百条**高质量对话样本；少于 100 条通常只能学到语气，学不到判断。
- 时间：0.5B、500 条、3 epoch，单张中端显卡大约几十分钟（估计）。
- 没有 GPU 也可以租用按小时计费的云 GPU；先用 `--dry-run` 和示例数据确认流程再花钱。

## 微调 vs SKILL.md：选哪个？

- **SKILL.md（推荐先做）**：可读、可改、可以给任何 AI 用，成本几乎为零。
- **微调**：学"说话方式"，但小模型知识和推理能力有限，事实性内容容易幻觉；更新偏好要重新训练。

## 隐私

- `skill-distill sft` 会自动脱敏手机号、邮箱、身份证号、银行卡、token、私钥，但正则不可能 100% 覆盖，**训练前请人工抽查**。
- 真实数据放在 `data/private/`（已在 `.gitignore`），**不要**提交到公开仓库；训练出的 adapter 也可能记住原文，同样不要公开。
- 用云 GPU 时，训练结束后删除上传的数据和 checkpoint。
