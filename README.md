# Gemma-4 26B-A4B · MLX LoRA 微调设计项目（完整记录）

> **定位**：在 Apple Silicon（**M1 Max / 64GB 统一内存**）上运行 **MLX LoRA**，对
> `gemma-4-26B-A4B-it-QAT-MLX-4bit` 基座、在 **DocVQA + 代码 + 对话** 数据集上，按**三种不同数据配比**
> （**100% 纯 DocVQA** / **70-20-10 旧混合** / **50-25-25 新混合**）做一套**微调设计**（fine-tune design，
> 不是价格，是「配比 → 效果」的设计研究），并得出对应的**学习效果**。
>
> 本仓是完整存档：三套数据 + 六份配置 + 全部脚本（数据生成/训练/评测/图表）+ **可扩展多领域测试框架** +
> 训练日志/loss 曲线 + 多模型对比结果 + 模型获取方法。14G 完整模型体积超 GitHub 限制，不进仓，
> 见 `models/README.md`（外置盘 + LM Studio + 重新 fuse 命令）。

## 一句话结论
纯 DocVQA（100%）模型在文档问答上最强，但泛化差（Test ppl 5.14）；**50-25-25 混合（v2）** 在保住 DocVQA 的同时
把代码、对话、数学、指令遵循都拉起来了（Test ppl 1.535），且**消除了基座的 `<channel>thought` 思考标签泄漏**。
推荐日常用 **v2 混合模型**，纯 DocVQA 模型留作领域专用对照。

## 硬件 / 工具链
| 项 | 值 |
|---|---|
| 机器 | Mac M1 Max，64GB 统一内存 |
| 框架 | `mlx-lm` v0.32.0（venv: `~/venvs/mlx`） |
| 推理/运行 | LM Studio（本地 OpenAI 兼容服务，端口 1234） |
| 基座模型 | `gemma-4-26B-A4B-it-QAT-MLX-4bit`（MoE，约 14G） |

## 训练总览（三次 LoRA）
| 运行 | 数据 | 条数 | 配比 | 关键超参 | Test loss / ppl |
|---|---|---|---|---|---|
| **full**（最初 100%） | 纯 DocVQA | 15,000 | 100% DocVQA | r8 a4 / lr5e-5 / 15000步 / seq1024 | 1.638 / 5.144 |
| **mixed v1** | DocVQA+代码+对话 | 13,500 | 旧混合 | 同上（恢复自中断） | best_val 0.376 |
| **mixed v2** | DocVQA+代码+对话 | 13,498 | **50/25/25** | 同上（resume 自 7500） | 0.428 / 1.535 |

LoRA 统一：`rank=8, alpha(scale)=4.0, 30 层, dropout 0, 掩码 prompt`。

## 目录结构
```
gemma4-lora-finetune/
├── README.md                # 本文件（总览）
├── configs/                 # 全部 6 个 LoRA 配置（full / mixed / v1·v2 + resume）
├── data/                    # 三套数据 + 数据配方说明
│   ├── full/                #   100% DocVQA（train 15000 / valid / test）
│   ├── mixed_v1/            #   v1 混合
│   ├── mixed_v2/            #   v2 混合 50/25/25
│   └── README.md            #   数据来源、筛选规则、配比
├── scripts/
│   ├── data_gen/            # 对话生成 + 数据装配（gen_dialogue_v2 / assemble_v2 / sample_sources…）
│   ├── train/               # checkpoint 清理器 + fuse.sh
│   ├── eval/                # 测试框架：run_suite / compare_tables / convert_legacy / test_domains / docvqa_eval / ab_test
│   └── charts/              # loss 曲线图生成（make_mixed_charts / make_v2_charts）
├── testing/
│   ├── test_suite.json      # 可扩展多领域问题库（8 组）
│   └── README.md            # 怎么跑、怎么加组
├── results/
│   ├── training/            # 各次训练日志 + loss 数据
│   ├── eval/                # 多模型测试 JSON（v1/v2/base + docvqa_eval + ab_test）
│   └── charts/              # loss 曲线 HTML/PNG
├── docs/
│   ├── training_log.md      # 完整训练时间线
│   ├── incidents.md         # 事故与修复（crash / 泄漏 / 工具坑）
│   └── decisions.md         # 关键决策记录
└── models/
    └── README.md            # 模型在哪、怎么重新 fuse、怎么用
```

## 快速复现
```bash
# 1) 装环境（Python 3.12 venv）
python -m venv ~/venvs/mlx && ~/venvs/mlx/bin/pip install -U mlx-lm
# 2) 数据（本仓 data/ 已含三套）
# 3) 训练（示例：v2 混合）
~/venvs/mlx/bin/mlx_lm.lora -c configs/lora_config_mixed_v2.yaml --model <BASE> --train
# 4) 融合回完整模型
~/venvs/mlx/bin/mlx_lm.fuse --model <BASE> --adapter-path <ADAPTER> --save-path <OUT>
# 5) 跑多领域测试 + 生成对比表
~/venvs/mlx/bin/python scripts/eval/run_suite.py <MODEL> out.json testing/test_suite.json
~/venvs/mlx/bin/python scripts/eval/compare_tables.py out.json:标签 ... -o 对比表.md
```

## 模型本体（14G，不进 Git）
完整 fuse 模型体积 14G，超出 GitHub 普通仓库限制，故**不放进本仓**（`.gitignore` 已排除）。
获取方式见 `models/README.md`：外置盘 `/Volumes/资料项目整理/` 与 LM Studio 本地目录 `~/.lmstudio/models/local/`。
本仓保留 1.2G 的 LoRA adapter 路径说明 + 重新 fuse 命令，可随时重建完整模型。
