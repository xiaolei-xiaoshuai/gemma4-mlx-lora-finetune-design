# 训练时间线（完整记录）

所有运行均在 Mac M1 Max / 64GB 统一内存 + `mlx-lm v0.32.0` 上完成，基座 `gemma-4-26B-A4B-it-QAT-MLX-4bit`。
LoRA 统一超参：`rank=8, alpha(scale)=4.0, 30 层, dropout=0, lr=5e-5, adamw, mask_prompt=true, grad_checkpoint=true`。

## Run 1 · full（100% DocVQA，最初那版）
- **日期**：2026-10-15 ~ 10-16
- **数据**：`data/full`（纯 DocVQA，train 15,000）
- **步数**：15,000（1 epoch）
- **结果**：Train loss 收敛到 ~1.85；Val loss ~1.64；**Test loss 1.638 / ppl 5.144**
- **产物**：`adapters_full/`（1.2G）→ fuse 成 `gemma-4-26B-A4B-docvqa-lora-full`（14G）
- **日志**：`results/training/full_train.log`、`results/charts/loss_chart_full.html`
- **评测**：`results/eval/docvqa_eval_results.json`、`ab_test_results.json`

## Run 2 · mixed v1（旧混合配方）
- **日期**：2026-10-18
- **数据**：`data/mixed_v1`（docvqa77/code16/docs4/dialogue10，train 13,500）
- **结果**：**best_val 0.376 @ step 4600**
- **事故**：step 7670 时 macOS GPU 抢占崩溃（见 incidents.md）→ 选用 7500 ckpt fuse
- **产物**：`adapters_mixed/` → fuse 成 `gemma-4-26B-A4B-mixed-lora`（v1）
- **日志**：`results/training/mixed_train.log`、`results/charts/mixed_loss_chart.{html,png}`

## Run 3 · mixed v2（50/25/25，推荐）
- **日期**：2026-10-22 ~ 10-23
- **数据**：`data/mixed_v2`（docvqa50/code25/dialogue25，train 13,498）
- **结果**：Test loss 0.428 / ppl 1.535（v2 完整跑完 13,498 步）
- **过程**：首轮在 7500 步处中断 → 用 `lora_config_mixed_v2_resume.yaml` 从 7500 ckpt 续训到 13,498
- **产物**：`adapters_mixed_v2/` → fuse 成 `gemma-4-26B-A4B-mixed-v2-lora`（v2，15G 含数据）
- **日志**：`results/training/mixed_v2_train.log` + `mixed_v2_resume.log`、`results/charts/mixed_v2_loss_chart.html`
- **多领域测试**：`results/eval/test_v2.json`（对比 `test_v1.json` / `test_base.json`）

## 评测结论（多领域对比）
| 模型 | DocVQA | Code | 对话 | thought 泄漏 | Test ppl |
|---|---|---|---|---|---|
| 基座（未微调） | 会泄漏思考标签 | 会泄漏 | 会泄漏 | **是**（`<\|channel>thought`） | — |
| full（100% DocVQA） | **最强** | 弱 | 弱 | 否 | 5.144 |
| v1 混合 | 强 | 中 | 中 | 否 | ~1.50 |
| **v2 混合** | 强（略逊 full） | **好** | **好** | 否 | 1.535 |

**推荐 v2 混合模型作为通用主力**；full 100% DocVQA 作为领域专精对照。
