# Planificación: Sistema de Autocobro para Despensa Comunitaria

Trabajo Práctico Final - Visión por Computadora II (CEIA FIUBA)
Fecha de entrega: 20 de abril de 2026

---

## Contexto del Proyecto

### Caso de Uso

Sistema de autocobro para una mutual con despensa comunitaria. Los asociados:
1. Arman su pedido
2. Pasan por la estación de autocobro
3. Retiran productos de a uno desde su bolsa/canasto
4. Colocan cada producto sobre la mesa de control (los productos se van acumulando)
5. Al finalizar, todos los productos quedan sobre la mesa para validación final

### Restricciones

- Catálogo acotado: ~60 productos
- Pedidos pequeños: 10-20 productos por pedido
- Dataset: Propio (a generar)

### Alcance del Proyecto

#### Dentro del alcance (lo que hay que entregar)

1. Dataset anotado de los productos de la despensa
2. Modelo entrenado de detección y clasificación
3. Script de inferencia que procese video y devuelva detecciones
4. Visualización básica para la demo (video con bboxes + lista de productos)
5. Paper IEEE documentando metodología y resultados
6. Repositorio con código reproducible

#### Fuera del alcance (no se implementa)

- Integración con sistema de gestión de la mutual
- Base de datos de productos/precios
- Sistema de pagos
- Interfaz de usuario para producción
- Lógica de carrito de compras persistente
- Deployment en hardware específico

---

## Arquitectura del Sistema

### Decisión de Arquitectura: Single-Stage

| Opción | Descripción | Pros | Contras |
|--------|-------------|------|---------|
| A: Single-stage (elegida) | YOLOv10 fine-tuned con 60 clases | Simple, un solo modelo, fácil debug | Menos flexible si hay clases muy similares |
| B: Two-stage | Detector genérico + clasificador separado | Más modular, permite especializar | Mayor complejidad, más latencia |

YOLOv10 hace detección Y clasificación en un solo paso al entrenarse con las 60 clases de productos.

### Pipeline

```
Cámara -> YOLOv10 (detecta + clasifica) -> Lógica de registro -> Output JSON + Video
```

### Componentes

| Etapa | Componente | Descripción |
|-------|------------|-------------|
| 1. Input | Cámara 1080p | Video en tiempo real o archivo |
| 2. Detección + Clasificación | YOLOv10 (60 clases) | Localiza Y clasifica productos en un paso |
| 3. Tracking (opcional) | ByteTrack | Asigna IDs persistentes, solo si baseline insuficiente |
| 4. Output | JSON + Video | Estado de mesa + visualización |

---

## Interfaz del Sistema

### Entrada

```bash
# Opción 1: Archivo de video
uv run python src/inference.py --model models/best.pt --source video.mp4

# Opción 2: Cámara en vivo
uv run python src/inference.py --model models/best.pt --source 0
```

### Salida

#### Video con visualización

- Productos identificados: bbox verde + label con nombre y confidence
- Productos no identificados: bbox rojo + label "Unknown"
- Panel lateral con lista de productos

#### JSON por frame

```json
{
    "frame_id": 60,
    "productos_en_mesa": [
        {"product_id": "leche_entera_1l", "confidence": 0.94, "status": "identified"},
        {"product_id": "pan_lactal", "confidence": 0.87, "status": "identified"},
        {"product_id": null, "confidence": 0.45, "status": "unknown"}
    ],
    "total": 3
}
```

### Criterios de Clasificación

| Condición | Status | Visual |
|-----------|--------|--------|
| confidence >= threshold | identified | bbox verde |
| confidence < threshold | unknown | bbox rojo |

Nota: El threshold se ajusta según curva precision-recall en validación.

---

## Lógica de Registro Estable

Para reducir duplicados y falsos positivos:

| Concepto | Valor sugerido | Descripción |
|----------|----------------|-------------|
| Frames de confirmación | N = 5-10 frames | Producto debe ser visible y estable por N frames antes de registrarse |
| Frames de desaparición | M = 15-30 frames | Producto debe desaparecer por M frames antes de considerarse "retirado" |
| Umbral de movimiento | IoU > 0.7 | Si bbox se mueve poco, es el mismo producto |

---

## Épicas y Tareas

### EPIC-0: Validación de la Solución

Objetivo: Validar que el enfoque técnico funciona antes de escalar.

Tareas:
- 0.1 Investigación y recolección de papers relevantes
- 0.2 Armar setup mínimo de prueba
- 0.3 Conseguir 5-10 productos de prueba
- 0.4 Probar YOLOv10 pre-entrenado (COCO)
- 0.5 Crear mini-dataset (~10 imágenes por producto)
- 0.6 Entrenar modelo mínimo (fine-tuning)
- 0.7 Evaluar viabilidad y documentar aprendizajes
- 0.8 Probar ByteTrack con el mini-dataset
- 0.9 Grabar video corto de prueba end-to-end

Criterios de éxito:

| Métrica | Umbral mínimo |
|---------|---------------|
| mAP@0.5 en mini-dataset | > 0.70 |
| Confusión entre productos similares | < 20% |
| FPS en laptop | > 15 FPS |
| Detección de producto nuevo | > 90% precisión |

### EPIC-1: Dataset

Tareas:
- 1.1 Definir lista de productos del catálogo
- 1.2 Definir setup de captura (cámara, iluminación, fondo)
- 1.3 Capturar imágenes
- 1.4 Anotar imágenes (bounding boxes)
- 1.5 Split train/val/test (70/20/10)
- 1.6 Aplicar data augmentation (mosaic, mixup, flip, rotate)
- 1.7 Validar calidad del dataset

Protocolo de captura:
- Misma mesa, mismo ángulo (cenital), misma iluminación, misma distancia

Tipos de escenas:

| Tipo de escena | Cantidad mínima |
|----------------|-----------------|
| Producto individual | 30-50 por clase |
| Acumulación 2-5 items | 20 escenas |
| Acumulación 5-10 items | 20 escenas |
| Con mano visible | 10 escenas |
| Hard negatives | 10 escenas |

### EPIC-2: Modelo

Tareas:
- 2.1 Seleccionar arquitectura (YOLOv10n/s/m)
- 2.2 Configurar entrenamiento
- 2.3 Entrenar modelo inicial
- 2.4 Evaluar métricas (mAP, precision, recall)
- 2.5 Analizar errores (matriz de confusión)
- 2.6 Iterar: ajustar dataset o modelo
- 2.7 Optimizar para inferencia

### EPIC-3: Integración

Tareas:
- 3.1 Definir arquitectura del sistema
- 3.2 Implementar captura de video
- 3.3 Implementar pipeline de inferencia
- 3.4 Implementar lógica de registro estable
- 3.5 Implementar visualización
- 3.6 Testing end-to-end

### EPIC-4: Documentación

Tareas:
- 4.1 Redactar paper IEEE (4-6 páginas)
- 4.2 Documentar repositorio (README)
- 4.3 Grabar video demo (máximo 2 minutos)
- 4.4 Preparar presentación (10 minutos)

### EPIC-5: Tracking (Opcional)

Solo si hay tiempo después del baseline:
- 5.1 Evaluar si tracking es necesario
- 5.2 Integrar ByteTrack
- 5.3 Comparar métricas vs baseline

---

## Timeline

| Semana | Fechas | Épica | Entregable |
|--------|--------|-------|------------|
| 1 | 24-30 mar | EPIC-0 | Prototipo funcionando, decisión go/no-go |
| 2 | 31 mar - 6 abr | EPIC-1 + EPIC-2 | Dataset completo, modelo inicial |
| 3 | 7-13 abr | EPIC-2 + EPIC-3 | Sistema end-to-end funcionando |
| 4 | 14-19 abr | EPIC-4 | Paper, demo, presentación |
| - | 20 abr | ENTREGA | Deadline |

Fecha de freeze: 13 de abril de 2026
- A partir de esta fecha: no más cambios de arquitectura
- Solo bug fixes, evaluación, paper y demo

---

## Criterios de Éxito

| Criterio | Umbral mínimo | Ideal |
|----------|---------------|-------|
| mAP@0.5 en validación | > 0.70 | > 0.85 |
| Order-level accuracy | > 70% | > 90% |
| FPS en demo | > 15 FPS | > 25 FPS |
| Productos reconocidos | > 50/60 clases | 60/60 clases |

---

## Escenario de Demo

Flujo sugerido (2 minutos máximo):
1. Mostrar mesa vacía con sistema corriendo (5 seg)
2. Agregar 3-4 productos de a uno (45 seg)
3. Mostrar lista de compra acumulándose (10 seg)
4. Agregar 2-3 productos más, incluyendo uno "difícil" (30 seg)
5. Mostrar estado final de la mesa (15 seg)
6. Mostrar JSON de salida y métricas (15 seg)

Productos sugeridos:
- 2-3 fáciles (formas/colores distintos)
- 2-3 medianos
- 1-2 difíciles (similares entre sí)

---

## Métricas a Reportar

Métricas de detección:
- mAP (mean Average Precision)
- Precision / Recall / F1-Score por clase
- Latencia/FPS
- Matriz de confusión

Métricas de sistema:
- Order-level accuracy
- Item registration accuracy
- Duplicate rate
- Miss rate

---

## Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Modelo no distingue productos similares | Media | Medio | Identificar pares problemáticos en EPIC-0 |
| Tiempo insuficiente para tracking | Alta | Bajo | Ya está como opcional en EPIC-5 |
| Paper sin suficiente contenido técnico | Baja | Medio | Papers ya identificados |
| Oclusiones por mano del usuario | Media | Medio | Probar en EPIC-0 |
| Latencia inaceptable | Baja | Alto | Usar YOLOv10n |

---

## Entregables y Pesos

| Entregable | Peso |
|------------|------|
| Paper IEEE (4-6 páginas) | 30% |
| Repositorio GitHub | 20% |
| Presentación + Demo (video 2 min) | 50% |

---

## Referencias

Papers:
- [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- [RPC: A Large-Scale Retail Product Checkout Dataset](https://arxiv.org/abs/1901.07249)
- [ByteTrack: Multi-Object Tracking](https://arxiv.org/abs/2110.06864)
- [RT-DETR: DETRs Beat YOLOs](https://arxiv.org/abs/2304.08069)

Documentación:
- [Ultralytics (YOLOv10)](https://docs.ultralytics.com/)
- [Roboflow (anotación)](https://roboflow.com/)
- [uv (gestor de dependencias)](https://docs.astral.sh/uv/)
- [IEEE Conference Templates](https://www.ieee.org/conferences/publishing/templates.html)
