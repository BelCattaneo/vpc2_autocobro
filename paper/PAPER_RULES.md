# Reglas para el Paper IEEE

Guía de verificación para el paper del proyecto de autocobro.

---

## Requisitos del Proyecto

| Requisito | Valor |
|-----------|-------|
| Formato | IEEE Conference |
| Extensión | 4-6 páginas |
| Peso en nota | 30% |
| Idioma | Español (con abstract en inglés opcional) |

---

## Estructura Obligatoria

### 1. Title
- Específico y descriptivo
- Sin palabras innecesarias como "nuevo" o "novel"
- Incluir keywords relevantes (detección, YOLO, autocobro, tiempo real)

### 2. Abstract (máximo 250 palabras)
- Un solo párrafo
- Debe contener:
  - Contexto/problema
  - Solución propuesta
  - Metodología resumida
  - Resultados clave (métricas)
  - Conclusión principal
- Sin abreviaturas no definidas
- Sin referencias ni ecuaciones
- Sin footnotes

### 3. Keywords (3-5)
- Ejemplos: detección de objetos, YOLO11, autocobro, deep learning, visión por computadora

### 4. Introduction
- Contexto del problema (despensa comunitaria, mutual)
- Motivación (soluciones comerciales costosas)
- Contribuciones específicas del trabajo
- Organización del paper (último párrafo)

### 5. Trabajos Relacionados
- Detección de objetos (YOLO family, RT-DETR)
- Datasets de productos retail (RPC)
- Sistemas de checkout existentes
- Mínimo 5-8 referencias relevantes

### 6. Metodología
- Describir con suficiente detalle para replicar
- Incluir:
  - Arquitectura del sistema (diagrama)
  - Dataset (captura, anotación, splits, augmentation)
  - Modelo (selección, configuración, entrenamiento)
  - Pipeline de inferencia (lógica de registro estable)

### 7. Experimentos y Resultados
- Métricas claras (mAP, precision, recall, F1)
- Comparación POC vs dataset final
- Matriz de confusión
- Análisis de errores (clases problemáticas)
- FPS/latencia

### 8. Discusión
- Interpretación de resultados
- Limitaciones del trabajo
- No exagerar la importancia de los resultados

### 9. Conclusiones
- Resumen de hallazgos clave
- Implicaciones para el campo
- Trabajo futuro

### 10. Referencias
- Formato IEEE (números en corchetes [1], [2])
- Ordenadas por aparición en el texto
- Mínimo 10-15 referencias

---

## Formato IEEE

| Elemento | Especificación |
|----------|----------------|
| Columnas | 2 columnas |
| Fuente cuerpo | Times New Roman 10pt |
| Fuente título | 24pt bold |
| Fuente autores | 10pt regular |
| Fuente abstract | 9pt italic |
| Tamaño página | A4 (210mm x 297mm) |
| Márgenes | Top/Bottom: 19mm, Left/Right: 14mm |
| Espacio entre columnas | 5mm |
| Espaciado | Single-space |

---

## Checklist de Verificación

### Antes de entregar

- [ ] Título específico y descriptivo
- [ ] Abstract completo (problema, método, resultados, conclusión)
- [ ] Abstract menor a 250 palabras
- [ ] Keywords incluidas (3-5)
- [ ] Introducción termina con contribuciones claras
- [ ] Metodología suficiente para replicar
- [ ] Todas las figuras tienen caption descriptivo
- [ ] Todas las tablas tienen caption descriptivo
- [ ] Figuras y tablas referenciadas en el texto
- [ ] Métricas del dataset final incluidas
- [ ] Matriz de confusión incluida
- [ ] Comparación con baseline/POC
- [ ] Limitaciones mencionadas
- [ ] Trabajo futuro mencionado
- [ ] Referencias en formato IEEE [n]
- [ ] Referencias ordenadas por aparición
- [ ] Sin números de página (los agrega la conferencia)
- [ ] Sin biografías ni fotos de autores
- [ ] Extensión entre 4-6 páginas
- [ ] Compilado sin errores
- [ ] PDF generado correctamente

### TODOs pendientes en el paper

- [ ] Actualizar abstract con métricas del dataset final (60 clases)
- [ ] Agregar figura de arquitectura del sistema
- [ ] Agregar figura del setup de captura
- [ ] Agregar tabla de clases del dataset
- [ ] Agregar matriz de confusión
- [ ] Agregar curvas precision-recall
- [ ] Agregar tabla comparativa POC vs final
- [ ] Completar sección de resultados con métricas finales

---

## Referencias Obligatorias

Estas referencias deben aparecer en el paper:

1. YOLOv10: Real-Time End-to-End Object Detection
   - https://arxiv.org/abs/2405.14458

2. RPC: A Large-Scale Retail Product Checkout Dataset
   - https://arxiv.org/abs/1901.07249

3. ByteTrack: Multi-Object Tracking (si se usa tracking)
   - https://arxiv.org/abs/2110.06864

4. RT-DETR: DETRs Beat YOLOs (como comparación/contexto)
   - https://arxiv.org/abs/2304.08069

5. Ultralytics YOLO Documentation
   - https://docs.ultralytics.com/

---

## Errores Comunes a Evitar

1. Abstract demasiado vago (sin métricas específicas)
2. Introducción sin contribuciones claras
3. Metodología insuficiente para replicar
4. Resultados sin interpretación
5. Figuras sin caption o no referenciadas
6. Referencias incompletas o mal formateadas
7. Conclusiones que repiten el abstract
8. Exagerar la importancia de los resultados
9. No mencionar limitaciones
10. Usar "nosotros" excesivamente (preferir voz pasiva o impersonal)

---

## Fuentes

- [IEEE Author Center - Structure Your Paper](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/)
- [IEEE Conference Templates](https://www.ieee.org/conferences/publishing/templates.html)
