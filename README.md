# Sistema de Autocobro para Despensa Comunitaria

Trabajo Práctico Final - Visión por Computadora II (CEIA FIUBA)

Autores: Cattaneo, Ciarrapico, Pryszczuk (Grupo 4)

## Quick Start

```bash
# 1. Clonar e instalar
git clone https://github.com/BelCattaneo/vpc2_autocobro.git
cd vpc2_autocobro
uv sync --all-extras

# 2. Descargar dataset desde Drive (ver sección "Dataset")
# Link: https://drive.google.com/file/d/14hwSRfZ3qvrf3bUaZV5lpnX2ntSKO_dd
# Extraer en data/poc_multiproduct/ manteniendo estructura train/val/test

# 3. Entrenar (POC 10 clases)
uv run python src/train.py --data data/poc_multiproduct/data.yaml

# 4. Evaluar
uv run python src/evaluate.py --model runs/train/<exp>/weights/best.pt --data data/poc_multiproduct/data.yaml

# 5. Demo en vivo
uv run python src/demo.py --model models/best_multiproduct.pt --source 0
```

---

## Descripción

Sistema de detección y clasificación de productos en tiempo real para autocobro en una despensa comunitaria. Utiliza YOLOv10n fine-tuneado sobre un dataset propio de 10 clases de productos, con una lógica de registro estable basada en seguimiento multi-frame mediante IoU.

## Resultados

| Métrica | Validación | Test |
|---------|-----------|------|
| mAP@0.5 | 0.924 | 0.806 |
| mAP@0.5:0.95 | 0.826 | 0.717 |
| Precision | 0.903 | 0.778 |
| Recall | 0.834 | 0.670 |

Evaluado sobre 10 clases con YOLOv10n (2.3M parámetros). Inferencia >15 FPS en laptop, ~10 FPS en Raspberry Pi 5.

## Pipeline

```
Cámara USB → Frame OpenCV → YOLOv10n → ProductTracker (IoU) → Video anotado + JSON
                            (Detección +    (Confirmación
                             Clasificación)   multi-frame)
```

## Caso de Uso

1. El asociado pasa por la estación de autocobro
2. Coloca productos de a uno sobre la mesa de control (se acumulan)
3. El sistema detecta y clasifica cada producto en tiempo real
4. Al finalizar, se genera la lista completa para validación

## Stack Técnico

- Modelo: YOLOv10n (nano, para Raspberry Pi 5)
- Framework: Ultralytics + PyTorch
- Computer Vision: OpenCV
- Dependencias: uv (gestor de paquetes)
- Anotación: Roboflow (formato YOLO v5 PyTorch)

## Estructura del Proyecto

```
vpc2_autocobro/
├── src/
│   ├── demo.py                  # Demo principal con tracking y visualización
│   ├── train.py                 # Entrenamiento YOLOv10
│   ├── evaluate.py              # Evaluación formal (mAP, confusion matrix, PR curves)
│   ├── inference.py             # Inferencia básica
│   ├── extract_frames.py        # Extracción de frames de video
│   └── convert_heic.py          # Conversión HEIC → JPG
├── data/
│   ├── poc_10_clases/           # Dataset producto individual (10 clases)
│   └── poc_multiproduct/        # Dataset multi-producto (10 clases, con oclusiones)
├── models/                      # Modelos entrenados (.pt)
├── notebooks/                   # Notebooks de entrenamiento
├── runs/                        # Outputs (entrenamiento, evaluación, demo)
├── paper/                       # Paper IEEE (LaTeX)
├── docs/                        # Documentación (hardware, presentación)
└── pyproject.toml               # Dependencias
```

## Dataset

El dataset anotado (10 clases, formato YOLO v5 PyTorch) está disponible en Google Drive:

- POC producto individual: [Google Drive - POC v4](https://drive.google.com/file/d/14hwSRfZ3qvrf3bUaZV5lpnX2ntSKO_dd/view?usp=drive_link)
- POC multi-producto: Exportado desde [Roboflow](https://universe.roboflow.com/belns-workspace/poc-zdqcq)

Extraer en `data/poc_10_clases/` o `data/poc_multiproduct/` manteniendo la estructura `train/`, `val/`, `test/` con subdirectorios `images/` y `labels/`.

## Setup

### Requisitos

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Instalación

```bash
uv sync                  # dependencias core
uv sync --all-extras     # + jupyter, pandas, matplotlib, seaborn
```

## Uso

### Entrenar

```bash
# Default: YOLOv10n, 100 epochs, batch 16, 640x640
uv run python src/train.py --data data/poc_multiproduct/data.yaml

# Personalizar
uv run python src/train.py --model yolov10n.pt --epochs 200 --batch 32 --data data/poc_multiproduct/data.yaml
```

### Evaluar

```bash
# Evaluar sobre validación (genera confusion matrix, PR curves, métricas JSON)
uv run python src/evaluate.py --model models/best_multiproduct.pt --data data/poc_multiproduct/data.yaml

# Evaluar sobre test split
uv run python src/evaluate.py --model models/best_multiproduct.pt --data data/poc_multiproduct/data.yaml --split test

# Evaluar y copiar figuras al paper
uv run python src/evaluate.py --model models/best_multiproduct.pt --data data/poc_multiproduct/data.yaml --split test --paper-figures paper/figures
```

Outputs en `evaluate/<nombre>/`: `confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`, `metrics.json`.

### Demo

```bash
# Con webcam
uv run python src/demo.py --model models/best_multiproduct.pt --source 0

# Con video
uv run python src/demo.py --model models/best_multiproduct.pt --source data/videos/IMG_3278.MOV

# Sin guardar video
uv run python src/demo.py --model models/best_multiproduct.pt --source 0 --no-save
```

Visualización:
- Verde: producto confirmado (alta confianza, visto ≥5 frames)
- Amarillo: producto pendiente de confirmación
- Naranja: advertencia (baja confianza, requiere revisión)

### Inferencia rápida

```bash
uv run python src/inference.py --model models/best_multiproduct.pt --source imagen.jpg
```

## Clases (10)

| ID | Clase | ID | Clase |
|----|-------|----|-------|
| 0 | aceite_1l | 5 | leche_entera |
| 1 | aceite_4l | 6 | miel |
| 2 | dulce_de_leche | 7 | polenta |
| 3 | fideos | 8 | tomate |
| 4 | leche_descremada | 9 | yerba_kalena |

## Salida JSON

```json
{
  "timestamp": "2026-04-11T17:30:00",
  "productos": {"miel": 1, "polenta": 1, "fideos": 1},
  "total_items": 3,
  "advertencias": {"aceite_1l": 58}
}
```

## Linting y tests

```bash
uv run ruff check src/        # lint
uv run ruff format src/       # format
uv run pytest                  # tests
```

## Paper IEEE

```bash
cd paper && latexmk -lualatex paper.tex
```

## Equipo

Grupo 4 - CEIA FIUBA
- María Belén Cattaneo
- Nicolás Valentín Ciarrapico
- Sabrina Daiana Pryszczuk

## Referencias

- [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- [RPC: A Large-Scale Retail Product Checkout Dataset](https://arxiv.org/abs/1901.07249)
- [ByteTrack: Multi-Object Tracking](https://arxiv.org/abs/2110.06864)
- [Ultralytics Docs](https://docs.ultralytics.com/)
