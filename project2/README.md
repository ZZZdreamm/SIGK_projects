# SIGK – projekt 2: Low light image enhancement

Sieć neuronowa (PyTorch) rekonstruująca zdjęcia wykonane przy ekstremalnie małym
oświetleniu (−3 EV) z fizycznym szumem matrycy (Poisson + Gauss).
Dane: RAISE, Nikon D7000, zdjęcia wnętrz. Używamy 100 pierwszych, a ewaluację
robimy na 20 ostatnich.

**Dokumentacja eksperymentów, wyniki i wnioski: [docs/DOKUMENTACJA.md](docs/DOKUMENTACJA.md).**

| Metoda (20 obrazów testowych) | PSNR [dB] ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| baseline (wyrównanie histogramu) | 11,31 | 0,152 | 1,076 |
| network (`isp`) | **29,74** | **0,762** | **0,261** |

## Uruchomienie (uv + just)

```bash
just sync          # uv sync – Python 3.10+, torch z CUDA 12.8 (RTX 50xx)
just download      # 100 plików NEF (~1,9 GB) do data/nef
just prepare       # data/dataset/{gt,input,linear}/*.png + split.json
just train-all     # 6 wariantów sieci (~30 min każdy na RTX 5050)
just evaluate      # results/summary_test.csv, results/table_test.md
just robustness    # results/robustness.csv
just figures       # docs/figures/*.png
# albo wszystko naraz:
just all
```

**Demo na jednym zdjęciu** (np. do prezentacji). Plik RAW jest przyciemniany i
zaszumiany tak samo jak zbiór danych, a wynik pokazuje metryki względem GT:

```bash
just predict data/nef/rc1d2751dt.NEF                 # -> results/demo/rc1d2751dt_comparison.png
just predict ciemne_zdjecie.png                      # gotowe ciemne zdjęcie 8-bit (bez metryk)
just predict data/nef/rc1d2751dt.NEF --ev -4 --seed 1
```

Pojedynczy trening: `just train <nazwa> [opcje train.py]`, np.
`just train test_run --input-mode srgb --epochs 5`.

## Struktura

| Plik | Zawartość |
|---|---|
| `lle/download.py` | pobieranie NEF (kod z instrukcji + pobieranie równoległe, ponawianie) |
| `lle/raw_pipeline.py` | kod z instrukcji z uzupełnionymi TODO + ta sama degradacja w PyTorch (`degrade_torch`) |
| `lle/models.py` | `LowLightNet`: Residual U-Net z uwagą kanałową + opcjonalne „fizyczne” wyrównanie ekspozycji wejścia |
| `lle/baselines.py` | wyrównanie histogramu (baseline) + metody pomocnicze |
| `lle/data.py`, `lle/losses.py`, `lle/metrics.py` | dane, funkcje straty, PSNR/SSIM/LPIPS |
| `scripts/*.py` | pobieranie, generacja zbioru, trening, ewaluacja, odporność, rysunki, demo (`predict.py`) |
