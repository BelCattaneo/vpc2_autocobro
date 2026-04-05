# Sistema de Autocobro para Despensa Comunitaria

Trabajo Práctico Final - Visión por Computadora II (CEIA FIUBA)

## Quick Start (POC - 10 clases)

```bash
# 1. Clonar e instalar
git clone https://github.com/BelCattaneo/vpc2_autocobro.git
cd vpc2_autocobro
uv sync --all-extras

# 2. Descargar dataset desde Drive (ver sección "Descargar dataset anotado")
# Link: https://drive.google.com/file/d/14hwSRfZ3qvrf3bUaZV5lpnX2ntSKO_dd

# 3. Entrenar (POC)
uv run jupyter notebook notebooks/train_single_product.ipynb
# Abrir en VS Code: seleccionar kernel .venv/bin/python y ejecutar celdas
```

---

## Descripción

Sistema de detección y clasificación de productos para autocobro en una despensa comunitaria. Utiliza visión por computadora para identificar productos colocados sobre una mesa de control, generando automáticamente una lista de compra.

## Caso de Uso

1. El asociado arma su pedido
2. Pasa por la estación de autocobro
3. Coloca productos de a uno sobre la mesa (los productos se acumulan)
4. El sistema detecta y clasifica cada producto en tiempo real
5. Al finalizar, se genera la lista completa para validación

## Características

- Detección en tiempo real usando YOLOv10
- Clasificación de ~60 productos de despensa
- Visualización con bounding boxes (verde = identificado, rojo = unknown)
- Salida JSON con estado de la mesa por frame
- Lógica de registro estable para evitar duplicados

## Stack Técnico

- YOLOv10 (detección + clasificación single-stage)
- Ultralytics
- OpenCV
- Python 3.10+
- uv (gestor de dependencias)

## Estructura del Proyecto

```
vpc2_autocobro/
├── data/
│   ├── heic/                     # Imágenes originales HEIC
│   │   ├── aceite_1l/
│   │   ├── aceite_4l/
│   │   └── ...
│   ├── jpg/                      # Imágenes convertidas a JPG
│   │   ├── aceite_1l/
│   │   └── ...
│   ├── videos/                   # Videos de prueba (.MOV)
│   └── poc_10_clases/            # Dataset YOLO (descargar de Drive)
│       ├── data.yaml             # Configuración del dataset
│       ├── train/images/, labels/
│       ├── val/images/, labels/
│       └── test/images/, labels/
├── models/
│   ├── base/
│   │   └── yolov10n.pt           # Modelo base pre-entrenado
│   ├── best.pt                   # POC: productos individuales
│   └── best_multiproduct.pt      # Multi-producto (oclusión)
├── src/
│   ├── convert_heic.py           # Conversión HEIC → JPG
│   ├── train.py                  # Entrenamiento YOLOv10
│   ├── inference.py              # Inferencia básica
│   └── demo.py                   # Demo con tracking y visualización
├── notebooks/
│   ├── train_single_product.ipynb       # POC: productos individuales
│   └── train_multi_and_single_product.ipynb  # Combinado: individual + multi
├── runs/
│   ├── train/                    # Outputs de entrenamiento
│   ├── detect/                   # Outputs de inferencia
│   └── demo/                     # Outputs de demo (videos + JSON)
├── docs/                         # Documentación adicional
├── pyproject.toml                # Dependencias (uv)
├── planification.md              # Planificación detallada
└── README.md
```

## Setup

### Requisitos

- Python 3.10+
- uv (instalación: https://docs.astral.sh/uv/getting-started/installation/)

### Instalación

```bash
cd vpc2_autocobro

# Crear entorno virtual e instalar dependencias
uv sync

# Con dependencias opcionales (pandas, matplotlib, jupyter)
uv sync --all-extras
```

## Flujo de Trabajo POC

### 1. Convertir imágenes HEIC a JPG

```bash
uv run python src/convert_heic.py
```

Esto convierte todas las imágenes de `data/heic/` a `data/jpg/`.

### 2. Descargar dataset anotado

El dataset anotado está disponible en Google Drive:

1. Descargar el zip desde: [Google Drive - Dataset POC](https://drive.google.com/file/d/14hwSRfZ3qvrf3bUaZV5lpnX2ntSKO_dd/view?usp=drive_link)
2. Extraer en el proyecto:
```bash
cd vpc2_autocobro
unzip ~/Downloads/POC.vX.yolov5pytorch.zip -d data/poc_10_clases_tmp
cp -r data/poc_10_clases_tmp/train/* data/poc_10_clases/train/
cp -r data/poc_10_clases_tmp/valid/* data/poc_10_clases/val/
cp -r data/poc_10_clases_tmp/test/* data/poc_10_clases/test/
rm -rf data/poc_10_clases_tmp
```

Estructura esperada después de extraer:
```
data/poc_10_clases/
├── data.yaml
├── train/
│   ├── images/   (102 imágenes)
│   └── labels/   (102 archivos .txt)
├── val/
│   ├── images/   (13 imágenes)
│   └── labels/
└── test/
    ├── images/   (12 imágenes)
    └── labels/
```

Para el responsable del dataset (anotación con Roboflow):
1. Crear cuenta/proyecto en [Roboflow](https://roboflow.com/)
2. Subir carpeta `data/jpg/` organizada por clase
3. Dibujar bounding boxes alrededor de cada producto
4. Exportar en formato "YOLO v5 PyTorch"
5. Subir el zip a Google Drive

### 3. Entrenar modelo

Opción A: Usar notebook (recomendado)
```bash
# POC - solo productos individuales
uv run jupyter notebook notebooks/train_single_product.ipynb

# Después de anotar multi-producto en Roboflow
uv run jupyter notebook notebooks/train_multi_and_single_product.ipynb
```

Opción B: Usar script
```bash
# Entrenamiento con configuración por defecto
uv run python src/train.py

# Personalizar hiperparámetros
uv run python src/train.py --model yolov10s.pt --epochs 200 --batch 32
```

Opciones del script:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| --model | yolov10n.pt | Modelo base (n/s/m/b/l/x) |
| --epochs | 100 | Épocas de entrenamiento |
| --batch | 16 | Tamaño del batch |
| --imgsz | 640 | Tamaño de imagen |
| --data | data/poc_10_clases/data.yaml | Config del dataset |

### 4. Ejecutar inferencia

```bash
# Con imagen
uv run python src/inference.py --model runs/train/<exp>/weights/best.pt --source imagen.jpg

# Con video
uv run python src/inference.py --model runs/train/<exp>/weights/best.pt --source video.mp4

# Con webcam
uv run python src/inference.py --model runs/train/<exp>/weights/best.pt --source 0

# Ajustar threshold de confianza
uv run python src/inference.py --model runs/train/<exp>/weights/best.pt --source imagen.jpg --conf 0.6
```

### 5. Ejecutar demo (recomendado)

El script `demo.py` proporciona una visualización completa con tracking estable y panel lateral:

```bash
# Con video
uv run python src/demo.py --model models/best_multiproduct.pt --source data/videos/IMG_3278.MOV

# Con webcam
uv run python src/demo.py --model models/best_multiproduct.pt --source 0

# Sin guardar video (solo visualización)
uv run python src/demo.py --model models/best_multiproduct.pt --source 0 --no-save
```

Opciones del demo:

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| --model | (requerido) | Path al modelo YOLO (.pt) |
| --source | (requerido) | Video, directorio, o "0" para webcam |
| --conf | 0.5 | Umbral de confianza para detecciones |
| --warn-conf | 0.15 | Umbral para advertencias (productos no reconocidos) |
| --confirm-frames | 5 | Frames consecutivos para confirmar detección |
| --output | auto | Path para JSON de salida |
| --no-save | false | No guardar video de salida |

Visualización:

```
+---------------------------+---------------+
|                           | Productos:    |
|      Video Feed           | leche x2      |
|      con bboxes           | fideos x1     |
|                           | Total: 3      |
|      Verde = confirmado   |               |
|      Amarillo = pendiente | Revisar:      |
|      Naranja = advertencia| ? aceite (32%)|
+---------------------------+---------------+
```

Colores de bounding boxes:
- Verde: producto confirmado (alta confianza, visto N frames)
- Amarillo: producto pendiente de confirmación
- Naranja: advertencia (baja confianza, requiere revisión)

## Clases del POC (10 clases)

| ID | Clase |
|----|-------|
| 0 | aceite_1l |
| 1 | aceite_4l |
| 2 | dulce_de_leche |
| 3 | fideos |
| 4 | leche_descremada |
| 5 | leche_entera |
| 6 | miel |
| 7 | polenta |
| 8 | tomate |
| 9 | yerba_kalena |

## Salida

El sistema produce:

1. Video con bounding boxes y panel lateral de productos
2. JSON con estado final de la mesa

Ejemplo de salida JSON (demo.py):

```json
{
  "timestamp": "2026-04-05T15:31:35.217350",
  "source": "data/videos/IMG_3278.MOV",
  "model": "models/best_multiproduct.pt",
  "conf_threshold": 0.5,
  "warn_threshold": 0.2,
  "frames_processed": 414,
  "productos": {
    "fideos": 1,
    "miel": 1,
    "dulce_de_leche": 1
  },
  "total_items": 3,
  "advertencias": {
    "aceite_1l": 58,
    "leche_descremada": 56
  }
}
```

- `productos`: items confirmados con alta confianza
- `advertencias`: items detectados con baja confianza que requieren revisión manual (número indica en cuántos frames fue visto)

## Criterios de Éxito POC

| Métrica | Umbral mínimo |
|---------|---------------|
| mAP@0.5 | > 0.70 |
| FPS en laptop | > 15 FPS |

## Equipo

Grupo 4 - CEIA FIUBA

## Referencias

- [YOLOv10](https://arxiv.org/abs/2405.14458)
- [RPC Dataset](https://arxiv.org/abs/1901.07249)
- [ByteTrack](https://arxiv.org/abs/2110.06864)
- [Ultralytics Docs](https://docs.ultralytics.com/)
