# SIGK – projekt 3: Rendering neuralny (model oświetlenia Phonga)

**Autorzy:** Kacper Multan, Mariusz Pakulski

Warunkowy GAN (PyTorch), który dla parametrów sceny (położenie kuli, położenie
światła, kolor rozproszenia, połyskliwość) generuje obraz kuli w modelu Phonga.
Sieć jest połączona z dostarczonym rendererem OpenGL (`SIGK___Projekt_3`), który
generuje zbiór danych i referencję.

**Dokumentacja eksperymentów, wyniki i wnioski: [docs/DOKUMENTACJA.md](docs/DOKUMENTACJA.md) (wersja PDF: [docs/DOKUMENTACJA.pdf](docs/DOKUMENTACJA.pdf)).**

| Metoda (600 obrazów testowych) | FLIP ↓ | LPIPS ↓ | SSIM ↑ | Hausdorff [px] ↓ |
|---|---|---|---|---|
| neural renderer (cGAN, kodowanie geometryczne) | 0,0014 | 0,0022 | 0,9989 | 2,05 |

GAN vs sama regresja L1 (rozdz. 5.1 dokumentacji): przy kodowaniu względnym
poprawiony cGAN (v2) wygrywa na 78–87% obrazów testowych (FLIP −21%, LPIPS −13%,
kula o 40% ostrzejsza). Przy kodowaniu geometrycznym warianty są równoważne.

## Uruchomienie (uv + just)

```bash
just sync            # uv sync – Python 3.10+, torch z CUDA 12.8
just dataset         # 3000 losowych scen z renderera (headless) -> data/dataset
just train-all       # 4 warianty sieci (~30 min każdy)
just train-v2        # GAN v2 + warianty bez GAN (porównanie GAN vs L1)
just evaluate        # results/summary.csv, results/table.md
just compare-gan     # test Wilcoxona GAN vs L1 + rysunki
just metric-study    # reakcja metryk na kontrolowane zniekształcenia
just figures         # docs/figures/*.png
just pdf             # docs/DOKUMENTACJA.pdf
just demo            # okno: renderer OpenGL | sieć | mapa FLIP
just demo-headless   # to samo bez okna -> results/demo/
```

Bez `just` te same polecenia są w [justfile](justfile), np. demo:

```bash
cd SIGK___Projekt_3/src
uv run python main.py neural --checkpoint ../../runs/geometric_gan/best.pt      # okno
uv run python main.py phong                                                     # podgląd losowych scen
uv run python main.py phong --window headless --num_images 3000 --output_path ../../data/dataset/
```

## Struktura

| Plik | Zawartość |
|---|---|
| `SIGK___Projekt_3/src/phong_window.py` | renderer z instrukcji + losowanie scen w `on_render`, zapis obrazów i `params.jsonl` |
| `SIGK___Projekt_3/src/neural_window.py` | `NeuralPhongWindow`: renderer + sieć w jednym oknie (referencja, sieć, mapa FLIP, metryki) |
| `SIGK___Projekt_3/src/main.py` | wybór zadania: `phong` (generacja zbioru) / `neural` (demo) |
| `nr/scene.py` | kamera, przedziały, losowanie z odrzucaniem scen zdegenerowanych, analityczny ray-tracer (testy) |
| `nr/encoding.py` | kodowania parametrów: `raw`, `camera` (względne), `geometric` (+ mapy pikselowe) |
| `nr/models.py` | generator U-Net i dyskryminator PatchGAN |
| `nr/metrics.py` | FLIP (NVlabs), LPIPS, SSIM, Hausdorff na krawędziach Canny |
| `nr/data.py` | wczytanie zbioru, podział 2200/200/600 |
| `scripts/*.py` | trening, ewaluacja, badanie metryk, rysunki |
