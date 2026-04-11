# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time self-checkout system for a community grocery store ("despensa comunitaria") using computer vision. Products placed on a table are detected by a webcam, classified by a fine-tuned YOLOv10 model, and added to a shopping list. Final project for Visión por Computadora II (CEIA FIUBA), deadline April 20, 2026. Authors: Cattaneo, Ciarrapico, Pryszczuk (Grupo 4).

## Assignment (Consigna)

Trabajo práctico grupal (3-4 personas) orientado a aplicar técnicas de visión por computadora a un problema real. Debe usar al menos un modelo de clasificación/detección (YOLO, SSD, etc.). No se requiere implementación completa, pero sí justificación conceptual de las decisiones.

**Entregables y ponderación:**
| Entregable | Peso |
|---|---|
| Paper IEEE (4-6 páginas) | 35% |
| Repositorio GitHub (código + recursos reproducibles) | 25% |
| Demo en vivo (~10 min) + defensa oral con preguntas | 30% |
| Examen con preguntas | 10% |

## Course Context

Repo de la materia: `../vision_computadora_II/` (branch `VpC2_2026`). Los conceptos base del curso que fundamentan este proyecto:

- **Clase 1 — CNNs desde cero:** Arquitecturas convolucionales (Conv2d, MaxPool, Dropout, FC), entrenamiento con PyTorch, métricas (loss, accuracy, F1-macro), matrices de confusión, visualización de feature maps.
- **Clase 2 — Data Augmentation y preprocesamiento:** Augmentations geométricas/fotométricas, balanceo de clases por oversampling sintético, estrategias de resolución (resize vs. crop vs. pad), EDA de datasets (distribución de resoluciones, análisis de color RGB/HSV), limpieza de datos, comparativa de normalización (ImageNet vs. dataset-specific stats).
- **Clase 3 — Transfer Learning:** Tres estrategias (from scratch / fine-tuning / feature extraction con backbone congelado), mecánica de `requires_grad`, arquitecturas clásicas (AlexNet, VGG, Inception, ResNet, EfficientNet), concepto de Negative Transfer.
- **Clases 4-6 (según programa):** Segmentación (UNet, DeepLabV3), Explicabilidad (GradCAM), Modelos generativos (NST, Autoencoders, DCGAN), Hyperparameter tuning.

Stack del curso: PyTorch, torchvision, OpenCV, PIL, sklearn, matplotlib, Google Colab con GPU.

## Commands

**Package manager:** `uv` (astral.sh)

```bash
uv sync                  # install core dependencies
uv sync --all-extras     # include dev extras (numpy, pandas, matplotlib, jupyter)

# Run scripts
uv run python src/convert_heic.py                                              # HEIC → JPG
uv run python src/extract_frames.py --input data/videos/ --output data/frames/ --every 10
uv run python src/train.py --model yolov10n.pt --epochs 100 --batch 16
uv run python src/inference.py --model models/best.pt --source video.mp4
uv run python src/demo.py --model models/best_multiproduct.pt --source 0      # main app, source 0 = webcam

# Notebooks
uv run jupyter notebook

# Linting
uv run ruff check src/
uv run ruff format src/

# Tests (no tests exist yet, but pytest is configured)
uv run pytest

# Build IEEE paper PDF (from /paper directory)
cd paper && make
```

## Architecture

### Pipeline

```
Webcam / Video → OpenCV frame → YOLOv10 (Ultralytics) → ProductTracker → draw_frame() → Display + JSON output
```

### Design choice: single-stage detection + classification

YOLOv10 is fine-tuned on ~10–60 product classes in one stage (no separate detector/classifier). This simplifies the pipeline at the cost of needing labelled data per class.

### Key components

- **`src/demo.py`** — Main production script. Contains:
  - `ProductTracker`: IoU-based multi-frame tracking. Items are "confirmed" after appearing in N consecutive frames (default 5) with confidence ≥ 0.5. Confirmed items persist for up to 10 frames of absence. Detections with 0.15 ≤ conf < 0.5 appear as "warnings" (orange boxes) but are not added to the shopping list.
  - `draw_frame()`: Renders annotated frame + side panel showing confirmed items and warnings.
  - `run_demo()`: Main loop — reads frames, runs inference, updates tracker, writes annotated video + final JSON.

- **`src/train.py`** — Fine-tunes YOLOv10 on a YOLO-format dataset. Saves to `runs/train/<timestamp>/`.

- **`src/inference.py`** — Thin wrapper around `model.predict()` for quick testing on images, videos, or webcam.

- **`src/convert_heic.py`** / **`src/extract_frames.py`** — Data preparation utilities.

### Data layout

Datasets use YOLO format (`images/` + `labels/` + `data.yaml`):
- `data/poc_10_clases/` — 10-class POC dataset (from Google Drive)
- `data/poc_multiproduct/` — Multi-product scenes (10 classes, includes occlusion). Note: `data.yaml` contains a hardcoded path for another team member's machine; update `path:` when training locally.

Trained weights live in `models/` (`best.pt` for single product, `best_multiproduct.pt` for multi).
Training outputs go to `runs/train/`, inference outputs to `runs/detect/`.

### Output format

`run_demo()` writes a JSON file with:
```json
{
  "confirmed_items": {"class_name": count},
  "warnings": ["class_name", ...]
}
```

## Scope boundaries

The following are intentionally out of scope (per `planification.md`):
- Flask web server for tablet UI (`src/server.py` — mentioned in `docs/hardware_setup.md` but not yet implemented)
- Payment integration
- Persistent database
- Production UI

## Paper

The IEEE conference paper is in `/paper/`. Build it with `make` inside that directory (uses `latexmk`). See `paper/PAPER_RULES.md` for formatting rules and the pre-submission checklist of missing content (confusion matrix, final metrics table, comparison table).
