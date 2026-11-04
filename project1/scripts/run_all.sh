#!/usr/bin/env bash
# Full pipeline: train + evaluate every task configuration, then build the report.
#
#   bash scripts/run_all.sh                      # default: the two tasks below
#   TASKS="denoise:0.01 denoise:0.03" bash scripts/run_all.sh
#   EPOCHS=5 TASKS="sr:32" bash scripts/run_all.sh    # quick run
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS=${TASKS:-"sr:32 sr:64 inpaint:3 inpaint:32"}
EPOCHS=${EPOCHS:-20}
MODEL=${MODEL:-resunet}
EXTRA=${EXTRA:-""}          # extra train.py args, e.g. EXTRA="--lpips-weight 0.1"

for t in $TASKS; do
  name=$(uv run python -c "import sys; sys.path.insert(0,'.'); from sigk.tasks import parse_task; print(parse_task('$t').name)")
  echo "=== $t ($name) ==="
  uv run python scripts/train.py --task "$t" --model "$MODEL" --epochs "$EPOCHS" $EXTRA
  uv run python scripts/evaluate.py --task "$t" --checkpoint "runs/${MODEL}_${name}/best.pt"
done
uv run python scripts/make_figures.py
uv run python scripts/doc_figures.py
uv run python scripts/report.py
