# 数据配方（三套数据集）

所有数据均为「messages 对话格式」（MLX 直接读 `data/<x>/{train,valid,test}.jsonl`），每条含
`messages`（role/content）与 `source`（来源标签）。切分一律**按 source 分层 90/5/5**，保证各 split 配比一致。

## 1. `full/` — 100% DocVQA（最初那版）
- 来源：DocVQA 文档问答语料（原始 16,445 条）
- 清洗：去完全重复 QA（−575）；同问题（不分大小写）上限 50 次（−147）→ 15,723 条
- 划分（seed=42）：train 15,000 / valid 360 / test 360
- 特点：**纯单一领域**，用于检验「纯 DocVQA 专精」上限

## 2. `mixed_v1/` — v1 混合
- 四块拼接（seed=42）：`docvqa` 10,500 + `code` 2,400 + `docs` 600 + `dialogue` 1,500 = 15,000
- 代码源：`CodeExercise-Python-27k`（抽 2,400，human/bot → user/assistant）
- 对话源：`gen_dialogue.py` 手写 + 事实表 × 模板生成 1,500 条
- 分层 90/5/5 → train 13,500
- 配比 ≈ docvqa 77% / code 16% / docs 4% / dialogue 10%

## 3. `mixed_v2/` — v2 混合 50/25/25（推荐配方）
- 三块拼接（seed=2025）：`docvqa` 7,500（从 10,500 抽）+ `code` 3,750（从 27k 抽）+ `dialogue` 3,750 = 15,000
- 对话源：`gen_dialogue_v2.py`（手写高价值种子 + 紧凑事实表×问法模板，内部去重并和 v1 的 1,500 去重，3,750 = 新 unique 优先 + v1 1,500）
- 分层 90/5/5 → train 13,498
- 配比 **50% DocVQA / 25% 代码 / 25% 对话**（v2 核心改进：对话/代码占比翻倍，更均衡）

## 生成 / 装配脚本（见 `scripts/`）
| 脚本 | 作用 |
|---|---|
| `scripts/data_gen/sample_sources.py` | 抽样 + 格式归一 + 打 source 标签 → `parts/` |
| `scripts/data_gen/gen_dialogue.py` | 生成 v1 对话 1,500 条 |
| `scripts/data_gen/gen_dialogue_v2.py` | 生成 v2 对话 3,750 条 |
| `scripts/data_gen/assemble_mixed.py` | 装配 v1 → `mixed_v1/` |
| `scripts/data_gen/assemble_v2.py` | 装配 v2 → `mixed_v2/`（50/25/25） |

> 注：`full/` 的 DocVQA 清洗结果已内嵌在本仓 `data/full/`；原始 59M 代码语料与 16k DocVQA 源不在仓内（体积原因，已在 `.gitignore` 排除）。
