# Setup de Hardware para Captura y Demo

Documentación del sistema de captura de imágenes y estación de demo.

---

## Arquitectura del Sistema

```
┌─────────────┐     USB      ┌──────────────┐     WiFi      ┌─────────────┐
│   Webcam    │─────────────▶│ Raspberry Pi │◀────────────▶│   Tablet    │
│  (cenital)  │              │  (servidor)  │               │ (interfaz)  │
└─────────────┘              └──────────────┘               └─────────────┘
       │                            │
       │                            │ micro-HDMI
       ▼                            ▼
  ┌─────────┐                ┌──────────────┐
  │  Mesa   │                │   Monitor    │
  │productos│                │   (debug)    │
  └─────────┘                └──────────────┘
```

### Descripción de Componentes

| Componente | Función | Conexión |
|------------|---------|----------|
| Webcam | Captura cenital de la mesa | USB al Raspberry Pi |
| Raspberry Pi | Procesamiento central (captura + inferencia YOLO) | - |
| Tablet | Interfaz de usuario (muestra video + lista) | WiFi (browser) |
| Monitor | Debug y desarrollo (opcional) | micro-HDMI |

---

## Hardware Requerido

### Lista de Compras

| Item | Modelo Recomendado | Precio Aprox | Notas |
|------|-------------------|--------------|-------|
| Raspberry Pi | Pi 5 8GB | $80 USD | Pi 4 8GB como alternativa ($75) |
| Fuente de alimentación | Official 27W USB-C | $12 USD | Importante: usar fuente oficial |
| MicroSD | 32GB+ Class 10 | $10 USD | Recomendado 64GB para logs |
| Cable micro-HDMI | micro-HDMI a HDMI | $8 USD | Pi 4/5 usan micro-HDMI |
| Webcam USB | Logitech C270/C920 | $20-70 USD | 720p mínimo, 1080p ideal |
| Case con ventilador | Official Pi 5 case | $10 USD | Necesario para inferencia continua |

Total estimado: $140-190 USD

### Hardware Disponible (no comprar)

- Tablet (cualquier tablet con browser)
- Monitor HDMI
- Red WiFi

---

## Especificaciones Técnicas

### Raspberry Pi - Comparación de Modelos

| Modelo | RAM | YOLO FPS (YOLO11n) | Recomendación |
|--------|-----|---------------------|---------------|
| Pi 4 4GB | 4GB | ~3-5 FPS | Mínimo viable |
| Pi 4 8GB | 8GB | ~3-5 FPS | Mejor multitarea |
| Pi 5 8GB | 8GB | ~8-12 FPS | Recomendado |

### YOLO - Variantes para Raspberry Pi

| Variante | Parámetros | FPS en Pi 5 | Precisión | Uso |
|----------|------------|-------------|-----------|-----|
| YOLO11n | 2.5M | ~10 FPS | Menor | Inferencia en Pi |
| YOLO11s | 9.4M | ~5 FPS | Media | Con paciencia |
| YOLO11m | 20.1M | ~2 FPS | Mayor | No recomendado |

Nota: Entrenar con cualquier variante en laptop/cloud, luego deployar el `.pt` al Pi.

### Webcam - Requisitos

| Característica | Mínimo | Ideal |
|----------------|--------|-------|
| Resolución | 720p | 1080p |
| FPS | 15 | 30 |
| Enfoque | Fijo OK | Autofocus |
| Campo de visión | 60° | 78° |

---

## Conexiones Físicas

### Webcam a Raspberry Pi

```
Webcam ──[USB-A cable]──▶ Puerto USB del Pi
```

- Plug and play en Linux
- Se detecta como `/dev/video0`
- OpenCV accede con `cv2.VideoCapture(0)`

### Raspberry Pi a Monitor

```
Pi ──[micro-HDMI to HDMI]──▶ Monitor
```

- Pi 4 y Pi 5 tienen puertos micro-HDMI (no HDMI estándar)
- Señal de video digital directa
- Funciona inmediatamente al bootear

### Raspberry Pi a Tablet

```
Pi ◀──[WiFi / misma red]──▶ Tablet (browser)
```

- No hay conexión física directa
- Ambos dispositivos en la misma red WiFi
- Pi ejecuta servidor Flask en puerto 5000
- Tablet abre browser a `http://<ip-del-pi>:5000`

---

## Flujo de Datos

### Diagrama de Señales

```
MUNDO FÍSICO              RASPBERRY PI                      TABLET
     │                         │                              │
     ▼                         ▼                              ▼
┌─────────┐              ┌───────────┐                 ┌───────────┐
│  MESA   │   luz        │  WEBCAM   │                 │  BROWSER  │
│   con   │─────────────▶│  SENSOR   │                 │           │
│productos│   fotones    │           │                 │ ┌───────┐ │
└─────────┘              └─────┬─────┘                 │ │ video │ │
                               │ USB                   │ └───────┘ │
                               │ (píxeles digitales)   │           │
                               ▼                       │ ┌───────┐ │
                         ┌───────────┐                 │ │ lista │ │
                         │  OPENCV   │                 │ └───────┘ │
                         │  captura  │                 └─────┬─────┘
                         │  frame    │                       ▲
                         └─────┬─────┘                       │
                               │ numpy array                 │
                               │ (640x480x3 RGB)             │
                               ▼                             │
                         ┌───────────┐                       │
                         │   YOLO    │                       │
                         │  MODEL    │                       │
                         │           │                       │
                         │ Detecta + │                       │
                         │ Clasifica │                       │
                         └─────┬─────┘                       │
                               │ detecciones                 │
                               │ [{clase, bbox, conf}]       │
                               ▼                             │
                         ┌───────────┐      HTTP/WiFi        │
                         │  FLASK    │───────────────────────┘
                         │  SERVER   │  (JPEG stream +
                         │           │   JSON data)
                         └───────────┘
```

### Transformación de Datos por Etapa

| Etapa | Formato | Tamaño | Descripción |
|-------|---------|--------|-------------|
| Luz en sensor | Fotones | - | Luz física reflejada por productos |
| Webcam digitaliza | Píxeles raw | ~1MB/frame | Sensor convierte luz a señal eléctrica |
| Transferencia USB | Comprimido | ~100KB/frame | MJPEG o raw según webcam |
| OpenCV frame | numpy array | 640×480×3 = 921KB | Imagen BGR en memoria |
| YOLO input | Tensor | 640×640×3 | Redimensionado y normalizado |
| YOLO conv layers | Feature maps | Variable | Patrones detectados jerárquicamente |
| YOLO output | Detecciones | ~1KB | Lista de {class_id, x, y, w, h, conf} |
| Flask response | JPEG + JSON | ~50KB | Comprimido para red |
| WiFi | Ondas de radio | 2.4/5GHz | Paquetes por el aire |
| Tablet renderiza | Píxeles | 1920×1080 | Browser muestra imagen + lista |

---

## Setup del Raspberry Pi

### Instalación de Sistema Operativo

1. Descargar Raspberry Pi Imager
2. Flashear Raspberry Pi OS (64-bit) en microSD
3. Configurar WiFi y SSH durante el flasheo
4. Insertar SD en Pi y bootear

### Instalación de Dependencias

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias de OpenCV
sudo apt install -y python3-opencv python3-pip libatlas-base-dev

# Instalar uv (gestor de dependencias)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clonar repositorio
git clone https://github.com/BelCattaneo/vpc2_autocobro.git
cd vpc2_autocobro

# Instalar dependencias Python
uv sync
```

### Verificar Webcam

```bash
# Listar dispositivos de video
ls /dev/video*

# Probar captura
uv run python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

---

## Protocolo de Captura de Imágenes

### Configuración Física

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Ángulo de cámara | Cenital (90°) | Minimiza oclusiones |
| Distancia cámara-mesa | 50-70cm | Cubre área de trabajo completa |
| Iluminación | Difusa, sin sombras duras | Consistencia entre capturas |
| Fondo | Uniforme (blanco o gris claro) | Facilita detección de bordes |

### Escenas a Capturar por Producto

| Tipo de Escena | Cantidad Mínima | Descripción |
|----------------|-----------------|-------------|
| Individual centrado | 20-30 | Producto solo, centrado |
| Individual variado | 10-20 | Producto solo, distintas posiciones |
| Con otros productos | 10-15 | 2-5 productos en escena |
| Parcialmente ocluido | 5-10 | Producto tapado parcialmente |
| Con mano visible | 5 | Mano colocando/retirando |

### Script de Captura

```bash
# Ejecutar script de captura con preview
uv run python src/capture.py --output data/raw/nombre_producto/

# Controles:
# ESPACIO = capturar imagen
# Q = salir
```

---

## Interfaz Web (Tablet)

### Servidor Flask

El servidor Flask corre en el Raspberry Pi y sirve:

- `/` - Página principal con video y lista
- `/video_feed` - Stream MJPEG del video
- `/shopping_list` - JSON con productos detectados

### Acceso desde Tablet

1. Conectar tablet a la misma red WiFi que el Pi
2. Obtener IP del Pi: `hostname -I`
3. Abrir browser en tablet
4. Navegar a `http://<ip-del-pi>:5000`

---

## Ejecución en Raspberry Pi

### Comandos para Ejecutar

```bash
# SSH al Pi
ssh pi@<ip-del-pi>

# Navegar al proyecto
cd vpc2_autocobro

# Opción 1: Inferencia básica
uv run python src/inference.py --model models/best_multiproduct.pt --source 0

# Opción 2: Demo con tracking (recomendado)
uv run python src/demo.py --model models/best_multiproduct.pt --source 0
```

### Cómo Funciona el Script

El script corre en un loop continuo - no espera ninguna señal o trigger:

```
┌─────────────────────────────────────────┐
│           SCRIPT CORRIENDO              │
│                                         │
│   ┌─────────┐                           │
│   │  START  │                           │
│   └────┬────┘                           │
│        ▼                                │
│   ┌─────────────┐                       │
│   │ Abrir webcam│  cv2.VideoCapture(0)  │
│   └──────┬──────┘                       │
│          ▼                              │
│   ┌──────────────┐                      │
│   │ Leer frame   │◀─────────────┐       │
│   └──────┬───────┘              │       │
│          ▼                      │       │
│   ┌──────────────┐              │       │
│   │ YOLO predict │              │       │
│   └──────┬───────┘              │       │
│          ▼                      │       │
│   ┌──────────────┐              │       │
│   │ Mostrar/     │              │       │
│   │ guardar      │              │       │
│   └──────┬───────┘              │       │
│          │                      │       │
│          └──────────────────────┘       │
│             (loop infinito)             │
│                                         │
│   Presionar 'Q' para detener            │
└─────────────────────────────────────────┘
```

La webcam envía frames constantemente (~30 por segundo). El script los captura tan rápido como puede y ejecuta YOLO en cada uno.

| Acción | Resultado |
|--------|-----------|
| Producto aparece en mesa | YOLO lo detecta, aparece bounding box |
| Producto se retira | Detección desaparece |
| Nada en la mesa | Frame vacío, sin detecciones |

No se necesita "trigger" - el sistema está constantemente observando.

### Antes de Ejecutar: Checklist

```bash
# 1. Clonar repo (si no está)
git clone https://github.com/BelCattaneo/vpc2_autocobro.git
cd vpc2_autocobro

# 2. Instalar dependencias
uv sync

# 3. Copiar modelo entrenado al Pi (desde laptop)
scp models/best_multiproduct.pt pi@<ip-del-pi>:~/vpc2_autocobro/models/

# 4. Verificar que la webcam está detectada
ls /dev/video*
```

### Display: Monitor vs Tablet

El script usa `cv2.imshow()` que necesita un display:

| Método | Comando | Cuándo usar |
|--------|---------|-------------|
| Con monitor HDMI | `uv run python src/demo.py ...` | Pi conectado a monitor |
| Sin monitor (solo guarda video) | `uv run python src/demo.py ... --no-display` | Headless |
| Interfaz tablet | `uv run python src/server.py` | Ver en browser del tablet |

---

## Troubleshooting

### Webcam no detectada

```bash
# Verificar que el dispositivo existe
ls /dev/video*

# Ver logs del kernel
dmesg | grep -i video

# Probar con otro puerto USB
```

### Flask no accesible desde tablet

```bash
# Verificar que Flask escucha en todas las interfaces
# Debe usar host='0.0.0.0', no '127.0.0.1'

# Verificar firewall
sudo ufw status

# Verificar que ambos están en la misma red
ping <ip-del-tablet>
```

### YOLO muy lento en Pi

- Usar YOLO11n (nano) en lugar de versiones más grandes
- Reducir resolución de entrada (320x320 en lugar de 640x640)
- Procesar cada 2-3 frames en lugar de todos

---

## Referencias

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [OpenCV VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [Flask Streaming](https://flask.palletsprojects.com/en/2.0.x/patterns/streaming/)
- [Ultralytics en Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/)
