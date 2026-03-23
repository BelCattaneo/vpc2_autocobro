# Sistema de Autocobro para Despensa Comunitaria

Trabajo Práctico Final - Visión por Computadora II (CEIA FIUBA)

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
- Python 3.x

## Estructura del Proyecto

```
vpc2_autocobro/
├── data/               # Dataset (imágenes y anotaciones)
├── models/             # Modelos entrenados
├── src/                # Código fuente
│   ├── inference.py    # Script principal de inferencia
│   └── ...
├── notebooks/          # Notebooks de experimentación
├── docs/               # Documentación adicional
└── README.md
```

## Uso

```bash
# Inferencia con webcam
python src/inference.py --source 0

# Inferencia con video
python src/inference.py --source video.mp4
```

## Salida

El sistema produce:
1. Video con bounding boxes y lista de productos
2. JSON con detecciones por frame

Ejemplo de salida JSON:
```json
{
    "frame_id": 60,
    "productos_en_mesa": [
        {"product_id": "leche_entera_1l", "confidence": 0.94, "status": "identified"},
        {"product_id": "pan_lactal", "confidence": 0.87, "status": "identified"}
    ],
    "total": 2
}
```

## Equipo

Grupo 4 - CEIA FIUBA

## Referencias

- [YOLOv10](https://arxiv.org/abs/2405.14458)
- [RPC Dataset](https://arxiv.org/abs/1901.07249)
- [ByteTrack](https://arxiv.org/abs/2110.06864)
- [Ultralytics Docs](https://docs.ultralytics.com/)
