# SIGK – modyfikacja obrazów (DIV2K, PyTorch)

**Dokumentacja eksperymentów z wynikami i wnioskami: [docs/DOKUMENTACJA.md](docs/DOKUMENTACJA.md).**

Projekt realizuje zadania z tematu „Modyfikacja obrazów”. Wszystkie cztery zadania są zaimplementowane, a do oddania wybiera się dwa z nich:

| Zadanie | Konfiguracje (`--task`) | Metoda bazowa |
|---|---|---|
| 2.1 Zwiększanie rozdzielczości | `sr:32`, `sr:64` | `cv2.resize`, `INTER_CUBIC` |
| 2.2 Odszumianie | `denoise:0.01`, `denoise:0.03` | `skimage.restoration.denoise_bilateral` |
| 2.3 Deblurowanie | `deblur:3`, `deblur:5` | `skimage.restoration.richardson_lucy` |
| 2.4 Inpainting* | `inpaint:3`, `inpaint:32` | `cv2.inpaint`, `INPAINT_TELEA` |

Metryki: MSE (w temacie „SNE”), PSNR, SSIM (scikit-image, `data_range=1`) oraz LPIPS (AlexNet). Wszystkie liczone są na **całym** zbiorze testowym. PSNR pojedynczego obrazu jest ograniczony do 100 dB. Bez tego kilka idealnie płaskich fragmentów (np. niebo), które bicubic odtwarza bezbłędnie, daje średnią równą nieskończoności.

## Instalacja

```bash
uv sync            # Python 3.10+, torch z CUDA 12.8 (wymagane dla kart RTX 50xx)
```

## Dane

Potrzebne są obrazy **HR** z DIV2K, czyli `DIV2K_train_HR.zip` i `DIV2K_valid_HR.zip`. Pliki `LR_x8` mają tylko ok. 255×170 px, więc nie da się z nich wyciąć obrazów 256×256.

```bash
# rozpakowane do data/raw/DIV2K_train_HR oraz data/raw/DIV2K_valid_HR
uv run python scripts/prepare_data.py --div2k-dir data/raw --out data/patches
```

Podział danych:

| Zbiór | Źródło | Liczba fragmentów 256×256 |
|---|---|---|
| train | DIV2K_train_HR 0001–0760, 16 losowych wycinków na obraz | 12160 |
| val | DIV2K_train_HR 0761–0800, 4 kafelki na obraz | 160 |
| test | DIV2K_valid_HR 0801–0900, pełna siatka kafelków bez nakładania | 3598 |

Oficjalny zbiór walidacyjny DIV2K służy jako zbiór testowy, bo testowy DIV2K nie ma publicznych obrazów HR. Opcja `--mode resize` zamiast wycinania skaluje cały obraz do 256×256.

Degradacje są nakładane w locie (`sigk/degradations.py`). Na zbiorze testowym są deterministyczne, więc metoda bazowa i sieć dostają dokładnie te same wejścia.

- **SR:** zmniejszanie przez `cv2.resize` z interpolacją **`cv2.INTER_AREA`**, czyli uśrednianiem obszarów, które nie powoduje aliasingu przy dużej skali. Sieć dostaje obraz LR powiększony bikubicznie do 256×256.
- **Szum:** `skimage.util.random_noise(mode="gaussian", var=σ²)`. Funkcja przyjmuje wariancję, stąd σ².
- **Rozmycie:** `cv2.GaussianBlur(img, (k, k), 0)`, gdzie σ wynika z rozmiaru jądra: 0.80 dla 3×3 i 1.10 dla 5×5.
- **Inpainting:** losowe kwadratowe dziury, 700× 3×3 albo 6× 32×32, czyli ok. 9–10% pikseli. Sieć dostaje dodatkowo maskę jako 4. kanał.

## Strojenie metod bazowych

Parametry metod bazowych są dobierane siatką na zbiorze walidacyjnym, nie testowym, i zapisywane w `configs/baselines.json`. `evaluate.py` używa ich automatycznie.

```bash
uv run python scripts/tune_baselines.py
```

Richardson-Lucy dostaje dokładne jądro Gaussa użyte do rozmycia. Obraz jest przed dekonwolucją rozszerzany odbiciem lustrzanym, bo zerowe brzegi w skimage powodują silne artefakty.

## Architektura (`sigk/models.py`)

**ResUNet** to U-Net z 3 poziomami (64→128→256→256 kanałów). Każdy etap enkodera i dekodera składa się z bloków rezydualnych z mechanizmem *channel attention* (jak w RCAN). Ma ok. 10.2 mln parametrów.

Globalne połączenie rezydualne sprawia, że sieć uczy się tylko poprawki do wejścia. Ostatnia warstwa jest zainicjalizowana zerami, więc nieuczona sieć zwraca dokładnie wejście. Przy inpaintingu znane piksele są przepisywane z wejścia, a sieć uzupełnia tylko dziury.

**PlainResNet** (`--model plain`) to jednoskalowa sieć w stylu EDSR do porównań (ablacji).

Funkcja straty to L1, L2 albo Charbonnier, opcjonalnie z perceptualnym LPIPS-VGG (`--lpips-weight`). Optymalizator to AdamW z rozgrzewką i schedulerem cosinusowym oraz mixed precision.

## Uruchomienie

```bash
# jedna konfiguracja
uv run python scripts/train.py    --task sr:64 --epochs 20
uv run python scripts/evaluate.py --task sr:64 --checkpoint runs/resunet_sr_64x64/best.pt

# wszystko: trening + ewaluacja + raport (domyślnie SR i inpainting)
bash scripts/run_all.sh
TASKS="denoise:0.01 denoise:0.03 deblur:3 deblur:5" bash scripts/run_all.sh

# eksperymenty dodatkowe: strata LPIPS dla SR oraz ablacja architektury na sr:64
bash scripts/run_extra.sh

# porównania wizualne wszystkich metod (z powiększonymi wycinkami) oraz tabele i wykresy
uv run python scripts/make_figures.py
uv run python scripts/doc_figures.py   # wykresy analityczne do dokumentacji
uv run python scripts/report.py
```

Na RTX 5060 Ti jedna epoka trwa ok. 5 minut, więc 20 epok to ok. 1.5–2 h na konfigurację.

## Wyniki

| Plik | Zawartość |
|---|---|
| `results/<zadanie>.json` | średnie i odchylenia MSE/PSNR/SSIM/LPIPS dla każdej metody |
| `results/per_image/<zadanie>.csv` | metryki dla każdego obrazu testowego |
| `results/figures/*_examples.png` | porównania wizualne: wejście, metoda bazowa, sieć, oryginał |
| `results/figures/*_bars.png`, `curves_*.png` | wykresy metryk i krzywe uczenia |
| `results/report.md`, `results/tables.tex` | tabele w formacie z tematu (Markdown i LaTeX) |

Nazwy metod w tabelach jednoznacznie opisują eksperyment, np. `richardson_lucy_deblur_gauss_3x3`, `bicubic_sr_32x32` czy `resunet_inpaint_32x32`. Wiersz `degraded_input_*` to metryki samego zdegradowanego wejścia, jako punkt odniesienia.

## Proponowane eksperymenty do dokumentacji

1. ResUNet a metoda bazowa dla obu parametrów każdego zadania (obowiązkowe).
2. Wpływ funkcji straty: `--loss l1` a `--loss l1 --lpips-weight 0.1`. Zwykle PSNR spada nieznacznie, a LPIPS wyraźnie się poprawia.
3. Ablacja architektury: `--model plain` oraz `--no-ca`, czyli bez channel attention.
4. Uczenie na wielu poziomach degradacji naraz, jeśli starczy czasu.

Przykład nazwy przebiegu dla eksperymentu z LPIPS:

```bash
uv run python scripts/train.py --task sr:32 --lpips-weight 0.1 --out runs/resunet-lpips_sr_32x32
```
