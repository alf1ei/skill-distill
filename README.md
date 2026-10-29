# skill-distill

把你和各种 AI 的聊天记录，蒸馏成**可复用的 SKILL.md**、**人格微调数据集**和**技能作品集**。纯 Python，核心只依赖 PyYAML，不需要付费 API。

> **English summary:** A small Python CLI that turns personal chat exports (ChatGPT `conversations.json`, or Markdown/TXT/JSON copies from Doubao, Claude, etc.) into (1) topic-clustered `SKILL.md` files with PII redaction and clear "user-confirmed vs. inferred" labels, (2) chat-format SFT JSONL plus a LoRA (transformers + peft) scaffold for small Qwen models, and (3) an honest capability matrix and GitHub profile README draft from a skills YAML and the GitHub API. Optional LLM drafting via any OpenAI-compatible endpoint; deterministic template fallback by default.

## 三个部分

| 部分 | 命令 | 产出 |
| --- | --- | --- |
| 1. 聊天 → SKILL.md | `skill-distill distill <文件或目录>` | `out/skills/<slug>/SKILL.md` + `INDEX.md` |
| 2. 人格微调脚手架 | `skill-distill sft <文件或目录>` + `finetune/` | `train.jsonl` / `eval.jsonl`，LoRA 训练与评估脚本 |
| 3. 技能作品集 | `skill-distill portfolio --skills skills.yaml` | `docs/portfolio.md`、`docs/profile-README.md` |

## 安装

```bash
git clone https://github.com/alf1ei/skill-distill && cd skill-distill
pip install -e ".[dev]"      # 微调再加 .[train]
pytest -q
```

## 1. 聊天记录 → SKILL.md

```bash
skill-distill distill examples/chats -o out/skills
```

流程：**解析 → 清洗 → 脱敏 → 去重 → 聚类 → 抽取 → 渲染**

- **解析**（`parsers.py`）
  - ChatGPT 官方导出 `conversations.json`：沿 `current_node` 取当前分支，跳过 system/tool 消息
  - 通用 JSON：`[{role, content}]`、`{messages: [...]}`、Claude 风格 `chat_messages` / content blocks
  - Markdown / TXT：按 `我：`、`豆包：`、`User:`、`**Assistant:**`、`## Claude` 等前缀切分；没有前缀就整段当作用户原话
- **脱敏**（`clean.py`）：手机号、座机、邮箱、身份证号、银行卡号、GitHub/OpenAI/Slack/AWS/Google token、Bearer、JWT、私钥 → `[PHONE]` 等占位符。正则做不到 100%，公开前请人工检查。
- **去重**：会话内相同消息去重，内容完全相同的会话只保留一份。
- **聚类**（`cluster.py`）：中文字符 bigram + 英文单词的 TF-IDF，余弦相似度单遍聚类，再重分配一轮；结果是确定性的。出现在过半文档里的词当作噪声丢掉。
- **抽取**（`extract.py`）：
  - 用户原话里的"喜欢 / 希望 / 偏好…" → **偏好（本人确认）**
  - 用户原话里的"不要 / 别 / 失败 / 报错…" → **坑点（本人确认）**
  - AI 回复里的编号步骤 → **步骤（推断）**，因为那是 AI 说的，不是你确认过的
- **渲染**（`render.py`）：frontmatter（`name`、`description` 写"什么时候用"）+ 步骤 / 偏好 / 坑点 / 来源（含日期）。

### 可选：用大模型起草（默认关闭）

```bash
export SKILL_DISTILL_LLM=1
export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 任意 OpenAI 兼容接口，也可以是本地 Ollama: http://localhost:11434/v1
export OPENAI_API_KEY=...                            # 本地模型可不填
export SKILL_DISTILL_MODEL=deepseek-chat
skill-distill distill my_chats/ -o out/skills
```

只把**已脱敏**的片段发出去；调用失败会自动回退到模板。注意：打开这个开关就意味着聊天内容会发到第三方服务。

## 2. 人格微调（只是脚手架）

```bash
skill-distill sft my_chats/ -o out/sft
python finetune/train_lora.py --dry-run
```

默认基座 `Qwen/Qwen2.5-0.5B-Instruct`，LoRA r=16，只对最后一条 assistant 回复计算 loss。显存、数据量、隐私注意事项见 [finetune/README.md](finetune/README.md)。**仓库里没有跑过训练**，`finetune/data/*.example.jsonl` 是合成示例。

## 3. 技能清单与作品集

编辑 `skills.yaml`（等级 0–4，自评），然后：

```bash
skill-distill portfolio --skills skills.yaml -o docs          # 调 GitHub API 读取公开仓库
skill-distill portfolio --skills skills.yaml --repos-json repos.json -o docs   # 离线
```

"证据"列只引用公开仓库（按 `repos` 名称或 `match` 语言/topic 匹配，fork 不算）。没有证据就写"暂无公开证据"，不会替你吹。

## 目录

```
src/skill_distill/   parsers clean cluster extract render llm sft portfolio cli
finetune/            lora_config.yaml train_lora.py eval.py data/*.example.jsonl
examples/chats/      合成的示例聊天（ChatGPT JSON / 豆包 Markdown / Claude TXT）
docs/                生成的 portfolio.md 与 profile-README.md
tests/               pytest
```

## 隐私

- 真实聊天放在 `data/private/`（已 `.gitignore`），不要提交。
- 示例数据全部是手写的合成内容，标有"示例 / example"。

## License

MIT
