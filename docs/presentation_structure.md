# Estructura de la Presentación

Visión por Computadora II - CEIA FIUBA
Duración: 10 minutos + video demo de 2 minutos
Peso: 50% de la nota final

---

## Estructura General

| Sección | Tiempo | Slides |
|---------|--------|--------|
| 1. Problema y Contexto | 1 min | 2 |
| 2. Solución Propuesta | 1 min | 2 |
| 3. Dataset | 2 min | 3-4 |
| 4. Modelo | 2 min | 3-4 |
| 5. Sistema de Inferencia | 1.5 min | 2-3 |
| 6. Resultados | 1 min | 2-3 |
| 7. Demo Video | 2 min | 1 (video embed) |
| Buffer/Cierre | 0.5 min | 1 |
| Total | 10 min | ~16-20 slides |

---

## Contenido por Sección

### 1. Problema y Contexto (1 min)

Slide 1: Título
- Detección de Productos en Tiempo Real para Autocobro usando YOLO11
- Autores: María Belén Cattaneo, Nicolás Valentín Ciarrapico, Sabrina Daiana Pryszczuk
- CEIA FIUBA - Visión por Computadora II

Slide 2: El Problema
- Despensa comunitaria de una mutual
- Proceso manual de checkout es lento
- Soluciones comerciales (Amazon Go, etc.) son costosas
- Necesidad: sistema accesible para organizaciones pequeñas

### 2. Solución Propuesta (1 min)

Slide 3: Nuestra Solución
- Sistema de detección en tiempo real
- Cámara cenital sobre mesa de control
- Identifica productos automáticamente
- Genera lista de compras

Slide 4: Por qué YOLO11
- Single-stage: detección + clasificación en un paso
- Tiempo real: >15 FPS
- State-of-the-art en detección de objetos
- Comparación breve con alternativas (RT-DETR, YOLOv8)

### 3. Dataset (2 min)

Slide 5: Productos
- 60 clases de productos de despensa
- Imagen con ejemplos de productos
- Criterios de selección

Slide 6: Setup de Captura
- Foto/diagrama del setup
- Cámara cenital
- Iluminación controlada
- Raspberry Pi + webcam + tablet

Slide 7: Protocolo de Captura
- Tipos de escenas:
  - Individual (30-50 por clase)
  - Múltiples productos (20 escenas)
  - Con oclusiones (10 escenas)
  - Con mano visible (10 escenas)
- Total: ~X imágenes

Slide 8: Anotación y Augmentation
- Herramienta: Roboflow
- Formato: YOLO (bounding boxes)
- Augmentation: mosaic, flip, rotate
- Splits: 70% train, 20% val, 10% test

### 4. Modelo (2 min)

Slide 9: Arquitectura YOLO11
- Diagrama simplificado de la arquitectura
- Variante elegida (n/s/m) y justificación
- Input → Backbone → Neck → Head → Output

Slide 10: Entrenamiento
- Configuración:
  - Epochs: X
  - Batch size: X
  - Learning rate: X
  - Hardware: X
- Tiempo de entrenamiento: X horas

Slide 11: Métricas de Entrenamiento
- Curvas de loss (train/val)
- mAP durante entrenamiento
- Punto de convergencia

Slide 12: Resultados del Modelo
- mAP@0.5: X.XX
- Precision: X.XX
- Recall: X.XX
- FPS: XX

### 5. Sistema de Inferencia (1.5 min)

Slide 13: Pipeline
- Diagrama: Cámara → YOLO → Tracker → Visualización
- Flujo de datos

Slide 14: Lógica de Registro Estable
- Problema: flickering, detecciones intermitentes
- Solución: confirmar después de N frames
- Doble umbral:
  - Alta confianza → confirmado (verde)
  - Baja confianza → revisar (naranja)

Slide 15: Interfaz
- Screenshot de la visualización
- Panel lateral con lista de productos
- Bounding boxes con colores

### 6. Resultados (1 min)

Slide 16: Matriz de Confusión
- Heatmap de confusión entre clases
- Clases problemáticas identificadas

Slide 17: Análisis de Errores
- Productos más confundidos
- Causas (similitud visual, oclusiones)
- Posibles mejoras

### 7. Demo Video (2 min)

Slide 18: Demo en Vivo
- Video embebido o link
- Duración: máximo 2 minutos

Contenido del video:
1. Mesa vacía, sistema corriendo (5 seg)
2. Agregar 3-4 productos de a uno (45 seg)
3. Lista acumulándose en panel lateral (10 seg)
4. Agregar productos difíciles (30 seg)
5. Estado final de la mesa (15 seg)
6. Mostrar JSON de salida (15 seg)

### 8. Cierre (0.5 min)

Slide 19: Conclusiones
- Sistema funcional para autocobro
- mAP alcanzado vs objetivo
- Viable para organizaciones comunitarias

Slide 20: Trabajo Futuro
- Integración con sistema de la mutual
- Más clases de productos
- Deployment en producción

---

## Checklist Pre-Presentación

### Slides
- [ ] Todas las slides tienen poco texto (bullet points)
- [ ] Imágenes/diagramas en cada slide
- [ ] Font legible (mínimo 24pt)
- [ ] Consistencia visual
- [ ] Sin errores de ortografía

### Demo Video
- [ ] Máximo 2 minutos
- [ ] Audio claro (o sin audio)
- [ ] Resolución adecuada
- [ ] Muestra casos exitosos
- [ ] Muestra al menos un caso difícil
- [ ] Comprimido para reproducción fluida

### Técnico
- [ ] Backup del video en USB
- [ ] Backup de slides en PDF
- [ ] Probar reproducción de video en el lugar
- [ ] Tener laptop cargada / cargador

### Ensayo
- [ ] Practicar timing (10 min estrictos)
- [ ] Definir quién presenta cada sección
- [ ] Preparar respuestas para preguntas esperadas

---

## Preguntas Esperadas

1. Por qué eligieron YOLO11 y no otra arquitectura?
2. Cómo manejan productos muy similares?
3. Qué pasa si un producto no está en el dataset?
4. Cuánto tiempo tomó el entrenamiento?
5. Funciona en tiempo real? A cuántos FPS?
6. Cómo se compara con soluciones comerciales?
7. Qué limitaciones tiene el sistema?
8. Cómo escalaría a más productos?

---

## Distribución de Presentadores

| Sección | Presentador |
|---------|-------------|
| 1. Problema | TBD |
| 2. Solución | TBD |
| 3. Dataset | TBD |
| 4. Modelo | TBD |
| 5. Sistema | TBD |
| 6. Resultados | TBD |
| 7. Demo | TBD |
| 8. Cierre | TBD |

---

## Referencias Visuales Necesarias

- [ ] Diagrama de arquitectura del sistema
- [ ] Foto del setup de captura
- [ ] Ejemplos de productos (grid de imágenes)
- [ ] Diagrama de YOLO11 simplificado
- [ ] Curvas de entrenamiento
- [ ] Matriz de confusión
- [ ] Screenshots de la interfaz
- [ ] Video demo (2 min)
