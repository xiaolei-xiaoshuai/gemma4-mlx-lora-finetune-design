#!/usr/bin/env bash
# 从 LoRA adapter 重建 14G 完整模型。
# 用法: ./fuse.sh <adapter_dir> <输出目录> [基座路径]
set -euo pipefail
ADAPTER_DIR="${1:?用法: fuse.sh <adapter_dir> <输出目录> [基座]}"
OUT="${2:?缺少输出目录}"
BASE="${3:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-MLX-4bit}"
MLX="${MLX_BIN:-$HOME/venvs/mlx/bin}"

if [ ! -f "$ADAPTER_DIR/adapters.safetensors" ] && [ ! -f "$ADAPTER_DIR/adapters_final/adapters.safetensors" ]; then
  echo "在 $ADAPTER_DIR 找不到 adapters.safetensors"; exit 1
fi
AP="adapters.safetensors"; [ -f "$ADAPTER_DIR/adapters_final/adapters.safetensors" ] && AP="adapters_final/adapters.safetensors"
echo "fuse: 基座 $BASE"
echo "     adapter $ADAPTER_DIR/$AP -> $OUT"
"$MLX/mlx_lm.fuse" --model "$BASE" --adapter-path "$ADAPTER_DIR/$AP" --save-path "$OUT"
echo "完成。把 $OUT 拷进 ~/.lmstudio/models/local/ 即可在 LM Studio 使用。"
