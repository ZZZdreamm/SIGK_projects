#!/usr/bin/env bash
# Builds the submission folder oddanie/ (ignored by git):
#   dokumentacja/  - documentation, figures, LaTeX tables
#   kod/           - source code needed to reproduce everything
#   wyniki/        - metrics (json/csv), all figures, example images
#   modele/        - trained checkpoints (runs/<name>/best.pt + config + history)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=oddanie
rm -rf "$OUT"
mkdir -p "$OUT"/{dokumentacja,kod,wyniki,modele}

cp -r docs/. "$OUT/dokumentacja/"
cp -r sigk scripts configs pyproject.toml uv.lock README.md .gitignore "$OUT/kod/"
find "$OUT/kod" -name __pycache__ -type d -prune -exec rm -rf {} +

cp -r results/*.json results/summary.csv results/tables.tex results/report.md results/figures results/per_image "$OUT/wyniki/"

for run in runs/*/; do
  [ -f "$run/best.pt" ] || continue
  name=$(basename "$run")
  mkdir -p "$OUT/modele/$name"
  cp "$run"/{config.json,history.csv} "$OUT/modele/$name/"
done
# keep only the weights + config (drop optimizer/scheduler state) -> ~4x smaller
uv run python - "$OUT" <<'PY'
import sys, torch
from pathlib import Path
for run in sorted(Path("runs").glob("*/best.pt")):
    ck = torch.load(run, map_location="cpu", weights_only=False)
    slim = {"model": ck["model"], "config": ck["config"], "epoch": ck["epoch"], "best_psnr": ck["best_psnr"]}
    torch.save(slim, Path(sys.argv[1]) / "modele" / run.parent.name / "best.pt")
PY

cat > "$OUT/CO_JEST_GDZIE.md" <<'MD'
# Oddanie projektu SIGK: zwiększanie rozdzielczości i odszumianie (DIV2K)

| Folder | Zawartość |
|---|---|
| `dokumentacja/` | `DOKUMENTACJA.md`: opis eksperymentów, tabele wyników, wizualizacje i wnioski; `tabele.tex`: tabele w LaTeX |
| `kod/` | kod źródłowy (PyTorch); instrukcja uruchomienia w `kod/README.md` |
| `wyniki/` | metryki dla każdej metody (`*.json`, `summary.csv`), wyniki dla każdego obrazu testowego (`per_image/`), wszystkie wykresy (`figures/`) |
| `modele/` | wytrenowane modele (`best.pt`) z konfiguracją i historią uczenia |

Szybka demonstracja na wytrenowanym modelu, uruchamiana z folderu `kod/` po `uv sync` i przygotowaniu danych:

    uv run python scripts/evaluate.py --task sr:64 --checkpoint ../modele/resunet_sr_64x64/best.pt --limit 50 --out demo
MD
du -sh "$OUT"/* | sort -h
