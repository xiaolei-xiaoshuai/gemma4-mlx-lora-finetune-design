# 事故与修复记录

## A1 · macOS GPU 抢占崩溃（mixed v1，step 7670）
- **现象**：训练到 7670 步时，Metal GPU 被系统抢占/掉线，进程中断。
- **根因**：M 系列统一内存上长时间高负载 GPU 训练 + 系统后台任务争抢 GPU，偶发抢占。
- **处置**：选用 7500 步 checkpoint（`adapters_mixed/0007500_adapters.safetensors`）直接 fuse，放弃剩余 1,500 步（7500 已是接近 best_val 的稳定区）。
- **教训**：长训练用 `save_every` 每 100 步存 ckpt + 配套清理器只留最新，随时可从断点恢复。

## A2 · mixed v2 首轮 7500 步中断
- **现象**：v2 跑到 7500 步中断。
- **处置**：写 `lora_config_mixed_v2_resume.yaml`，用 `--resume-adapter-file .../0007500_adapters.safetensors` 从 7500 续训到 13,498。
- **结果**：续训 loss 曲线平滑衔接（拼接 offset 7500），最终 Test ppl 1.535，无过拟合。

## A3 · `mlx_lm.fuse` 参数名坑
- **现象**：用 `--save-fused ... --hf-updated` 报 `unrecognized arguments`（见 `fuse_docvqa_full.log`）。
- **根因**：本版本 `mlx_lm.fuse` 只认 `--save-path`，`--save-fused/--hf-updated` 已不存在。
- **处置**：统一改用 `--save-path <输出目录>`。

## A4 · 浏览器渲染缺失 → 改用 qlmanage
- **现象**：校验 loss 曲线 HTML 时浏览器工具（无 Chromium）不可用。
- **处置**：用 `qlmanage -t` 把 HTML 渲染成 PNG，再用视觉分析确认坐标轴/图例/数据点完整。

## A5 · 对比表生成器无限循环
- **现象**：`compare_tables.py` 的 `parse_args` 里，匹配到带 `:` 的参数分支漏了 `i += 1`，导致死循环。
- **处置**：改为每个分支末尾统一 `i += 1`，并用合成数据回归验证。

## A6 · 基座思考标签泄漏（非 bug，属特性）
- **现象**：未微调基座输出含原始 `<|channel>thought` / `<channel>` 思考标签。
- **结论**：MoE 基座默认暴露思考通道；三组 LoRA（full/v1/v2）均成功抑制该泄漏，是「微调有效」的直接证据。
