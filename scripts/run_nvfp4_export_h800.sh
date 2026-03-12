#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/entquant}"
ENV_NAME="${ENV_NAME:-modelopt}"
FULL_PRECISION_MODEL_DIR="${FULL_PRECISION_MODEL_DIR:-$HOME/TensorRT-Model-Optimizer/Qwen3-4B}"
TEMPLATE_NVFP4_DIR="${TEMPLATE_NVFP4_DIR:-$HOME/TensorRT-Model-Optimizer/Qwen3-4B-NVFP4}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Qwen3-4B-NVFP4-entquant}"
VARIANT="${VARIANT:-entquant_exact}"
REG_PARAM="${REG_PARAM:-0.05}"
SOFT_PARAM="${SOFT_PARAM:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.20}"
LR="${LR:-1.0}"
MAX_ITERS="${MAX_ITERS:-80}"
BLOCK_CHUNK_SIZE="${BLOCK_CHUNK_SIZE:-8192}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
MAX_LAYERS="${MAX_LAYERS:-0}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/nvfp4_export_${TIMESTAMP}.log}"

echo "REPO_ROOT=$REPO_ROOT"
echo "FULL_PRECISION_MODEL_DIR=$FULL_PRECISION_MODEL_DIR"
echo "TEMPLATE_NVFP4_DIR=$TEMPLATE_NVFP4_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_FILE=$LOG_FILE"
echo "VARIANT=$VARIANT REG_PARAM=$REG_PARAM SOFT_PARAM=$SOFT_PARAM"
echo "DEVICE=$DEVICE MAX_ITERS=$MAX_ITERS BLOCK_CHUNK_SIZE=$BLOCK_CHUNK_SIZE MAX_SHARD_SIZE=$MAX_SHARD_SIZE"

PYTHONPATH="$REPO_ROOT" \
conda run --no-capture-output -n "$ENV_NAME" \
python -u "$REPO_ROOT/scripts/export_nvfp4_checkpoint.py" \
  --full-precision-model-dir "$FULL_PRECISION_MODEL_DIR" \
  --template-nvfp4-dir "$TEMPLATE_NVFP4_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --variant "$VARIANT" \
  --reg-param "$REG_PARAM" \
  --soft-param "$SOFT_PARAM" \
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --max-iters "$MAX_ITERS" \
  --block-chunk-size "$BLOCK_CHUNK_SIZE" \
  --max-shard-size "$MAX_SHARD_SIZE" \
  --device "$DEVICE" \
  --max-layers "$MAX_LAYERS" \
  2>&1 | tee "$LOG_FILE"
