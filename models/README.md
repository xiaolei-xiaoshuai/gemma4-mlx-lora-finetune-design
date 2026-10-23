# 模型获取（14G fuse 模型不进 Git）

完整模型每个 14G，超出 GitHub 普通仓库 100MB/文件 限制，故**不放进本仓**。
两个可靠存放处（均已做 SHA256 校验）：

| 模型 | 外置盘 | LM Studio 本地 |
|---|---|---|
| 100% DocVQA（full） | `/Volumes/资料项目整理/gemma4-docvqa-full/gemma-4-26B-A4B-docvqa-lora-full/` | `~/.lmstudio/models/local/gemma-4-26B-A4B-docvqa-lora-full/` |
| v1 混合 | `/Volumes/资料项目整理/gemma4-mixed-lora/` | —（已从本地清理） |
| v2 混合 | `/Volumes/资料项目整理/gemma4-mixed-v2/` | —（已从本地清理） |

## 重新 fuse（从 adapter 重建完整模型）
只需 1.2G 的 LoRA adapter + 基座即可重建 14G 完整模型（约 2 分钟）：
```bash
# full（100% DocVQA）
~/venvs/mlx/bin/mlx_lm.fuse \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit \
  --adapter-path <path>/adapters_full/adapters.safetensors \
  --save-path <输出目录>
# v2（混合）
~/venvs/mlx/bin/mlx_lm.fuse \
  --model ~/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit \
  --adapter-path <path>/adapters_final/adapters.safetensors \
  --save-path <输出目录>
```
> 注意：`mlx_lm.fuse` 参数是 `--save-path`（旧命令里的 `--save-fused/--hf-updated` 在本版本不存在，会报错）。

## 在 LM Studio / OpenAI 兼容 API 里用
fuse 出的目录丢进 `~/.lmstudio/models/local/`（LM Studio 会自动识别），或在 LM Studio 本地服务器（端口 1234）加载后用 OpenAI API 调用。

## 用回 14G 模型
把对应目录从外置盘拷回 `~/.lmstudio/models/local/` 或直接挂载使用；拷完可删本地副本省空间。
