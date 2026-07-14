#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/benchmarks/qwen3_32b_tp2_fp16/baseline.env"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
CAMPAIGN=${1:?usage: collect_env.sh CAMPAIGN_ID}
OUT="$ROOT/benchmark-results/$CAMPAIGN/environment"
mkdir -p "$OUT"

{
  date --iso-8601=seconds
  uname -a
  python --version
  evalscope --version
  python -m pip show torch torch-npu triton-ascend evalscope
  cat /usr/local/Ascend/version.info 2>/dev/null || true
  cat /usr/local/Ascend/driver/version.info 2>/dev/null || true
} > "$OUT/runtime.txt" 2>&1
{
  git branch --show-current
  git rev-parse HEAD
  git status --short --branch
} > "$OUT/git.txt"
cp "$ROOT/benchmarks/qwen3_32b_tp2_fp16/baseline.env" "$OUT/baseline.env"
cp "$ROOT/VERSION" "$OUT/version.txt"
cp "$MODEL_DIR/config.json" "$OUT/model-config.json"
sha256sum "$MODEL_DIR/config.json" > "$OUT/model-config.sha256"
env | grep -E '^(ASCEND|HCCL|NPU|PYTORCH|TORCH)' | sort > "$OUT/ascend-env.txt" || true
npu-smi info > "$OUT/npu-smi.txt" 2>&1 || true
