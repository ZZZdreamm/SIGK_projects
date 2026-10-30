#!/usr/bin/env bash
# Additional experiments for the documentation (run after run_all.sh):
#   1. perceptual loss (L1 + 0.1 * LPIPS-VGG) for both SR configurations
#   2. architecture ablation on sr:64: plain EDSR-like ResNet, ResUNet without channel attention
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS=${EPOCHS:-20}

run() {  # run <task> <run_name> <task_name> [train args...]
  local task=$1 run_name=$2 task_name=$3; shift 3
  echo "=== $task  $run_name ==="
  uv run python scripts/train.py --task "$task" --epochs "$EPOCHS" --out "runs/${run_name}_${task_name}" "$@"
  uv run python scripts/evaluate.py --task "$task" --skip-baseline --checkpoint "runs/${run_name}_${task_name}/best.pt"
}

run sr:32 resunet-lpips sr_32x32 --lpips-weight 0.1
run sr:64 resunet-lpips sr_64x64 --lpips-weight 0.1
run sr:64 plain         sr_64x64 --model plain
run sr:64 resunet-noca  sr_64x64 --no-ca
uv run python scripts/make_figures.py
uv run python scripts/doc_figures.py
uv run python scripts/report.py
