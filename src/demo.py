#!/usr/bin/env python3
"""
Demo del sistema de autocobro con visualización y tracking estable.

Uso:
    uv run python src/demo.py --model models/best_multiproduct.pt --source video.MOV
    uv run python src/demo.py --model models/best_multiproduct.pt --source 0  # Webcam
    uv run python src/demo.py --model models/best_multiproduct.pt --source video.MOV --conf 0.5
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class ProductTracker:
    """
    Tracker que confirma productos después de N frames consecutivos.
    Evita el flickering y mantiene un conteo estable de productos en la mesa.
    """

    def __init__(self, confirm_frames: int = 5, disappear_frames: int = 10, iou_threshold: float = 0.3):
        self.confirm_frames = confirm_frames
        self.disappear_frames = disappear_frames
        self.iou_threshold = iou_threshold

        self.next_id = 0
        self.pending = {}  # {id: {"class": str, "bbox": tuple, "count": int, "conf": float}}
        self.confirmed = {}  # {id: {"class": str, "bbox": tuple, "conf": float, "last_seen": int}}
        self.frame_count = 0

    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Calcula IoU entre dos bounding boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _match_detection_to_tracked(self, det_bbox: tuple, det_class: str, tracked_items: dict) -> int | None:
        """Encuentra el mejor match en tracked_items para una detección."""
        best_match = None
        best_iou = self.iou_threshold

        for track_id, item in tracked_items.items():
            if item["class"] != det_class:
                continue
            iou = self._calculate_iou(det_bbox, item["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_match = track_id

        return best_match

    def update(self, detections: list[dict]) -> dict[str, int]:
        """
        Actualiza el estado del tracker con nuevas detecciones.

        Args:
            detections: Lista de {"class": str, "bbox": (x1, y1, x2, y2), "conf": float}

        Returns:
            Conteo de productos confirmados: {"leche": 2, "fideos": 1}
        """
        self.frame_count += 1
        matched_confirmed = set()
        matched_pending = set()
        new_detections = []

        for det in detections:
            det_bbox = det["bbox"]
            det_class = det["class"]
            det_conf = det["conf"]

            # Intentar matchear con confirmed
            match_id = self._match_detection_to_tracked(det_bbox, det_class, self.confirmed)
            if match_id is not None:
                self.confirmed[match_id]["bbox"] = det_bbox
                self.confirmed[match_id]["conf"] = det_conf
                self.confirmed[match_id]["last_seen"] = self.frame_count
                matched_confirmed.add(match_id)
                continue

            # Intentar matchear con pending
            match_id = self._match_detection_to_tracked(det_bbox, det_class, self.pending)
            if match_id is not None:
                self.pending[match_id]["bbox"] = det_bbox
                self.pending[match_id]["conf"] = det_conf
                self.pending[match_id]["count"] += 1
                matched_pending.add(match_id)

                # Promover a confirmed si alcanza confirm_frames
                if self.pending[match_id]["count"] >= self.confirm_frames:
                    self.confirmed[match_id] = {
                        "class": self.pending[match_id]["class"],
                        "bbox": self.pending[match_id]["bbox"],
                        "conf": self.pending[match_id]["conf"],
                        "last_seen": self.frame_count,
                    }
                    del self.pending[match_id]
                continue

            # Nueva detección
            new_detections.append(det)

        # Eliminar pending no vistos (ANTES de agregar nuevos)
        pending_to_remove = [pid for pid in self.pending if pid not in matched_pending]
        for pid in pending_to_remove:
            del self.pending[pid]

        # Agregar nuevas detecciones a pending
        for det in new_detections:
            self.pending[self.next_id] = {
                "class": det["class"],
                "bbox": det["bbox"],
                "conf": det["conf"],
                "count": 1,
            }
            self.next_id += 1

        # Eliminar confirmed que no se ven hace muchos frames
        confirmed_to_remove = []
        for cid, item in self.confirmed.items():
            if cid not in matched_confirmed:
                frames_missing = self.frame_count - item["last_seen"]
                if frames_missing > self.disappear_frames:
                    confirmed_to_remove.append(cid)
        for cid in confirmed_to_remove:
            del self.confirmed[cid]

        # Calcular conteo por clase
        counts = defaultdict(int)
        for item in self.confirmed.values():
            counts[item["class"]] += 1

        return dict(counts)

    def get_confirmed_items(self) -> list[dict]:
        """Retorna items confirmados para visualización."""
        return [
            {"id": cid, "class": item["class"], "bbox": item["bbox"], "conf": item["conf"]}
            for cid, item in self.confirmed.items()
        ]

    def get_pending_items(self) -> list[dict]:
        """Retorna items pendientes para visualización."""
        return [
            {"id": pid, "class": item["class"], "bbox": item["bbox"], "conf": item["conf"]}
            for pid, item in self.pending.items()
        ]


def draw_frame(
    frame: np.ndarray,
    confirmed: list[dict],
    pending: list[dict],
    warnings: list[dict],
    counts: dict[str, int],
    panel_width: int = 250,
) -> np.ndarray:
    """
    Dibuja bounding boxes y panel lateral con lista de productos.
    """
    h, w = frame.shape[:2]

    # Crear frame extendido con panel lateral
    extended = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    extended[:, :w] = frame

    # Panel con fondo oscuro
    extended[:, w:] = (40, 40, 40)

    # Dibujar bounding boxes confirmados (verde)
    for item in confirmed:
        x1, y1, x2, y2 = [int(c) for c in item["bbox"]]
        cv2.rectangle(extended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{item['class']} {item['conf']:.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(extended, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(extended, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Dibujar bounding boxes pendientes (amarillo, más delgados)
    for item in pending:
        x1, y1, x2, y2 = [int(c) for c in item["bbox"]]
        cv2.rectangle(extended, (x1, y1), (x2, y2), (0, 200, 255), 1)

    # Dibujar bounding boxes de advertencia (naranja)
    for item in warnings:
        x1, y1, x2, y2 = [int(c) for c in item["bbox"]]
        cv2.rectangle(extended, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = f"? {item['class']} {item['conf']:.0%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(extended, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 5, y1), (0, 140, 255), -1)
        cv2.putText(extended, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Panel lateral: título
    y_offset = 30
    cv2.putText(extended, "Productos:", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 35

    # Línea separadora
    cv2.line(extended, (w + 10, y_offset - 10), (w + panel_width - 10, y_offset - 10), (100, 100, 100), 1)

    # Lista de productos
    if counts:
        sorted_items = sorted(counts.items(), key=lambda x: x[0])
        for class_name, count in sorted_items:
            display_name = class_name.replace("_", " ")
            if count > 1:
                text = f"{display_name} x{count}"
            else:
                text = display_name
            cv2.putText(extended, text, (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
            y_offset += 28
    else:
        cv2.putText(extended, "(ninguno)", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y_offset += 28

    # Total
    total = sum(counts.values())
    y_offset += 20
    cv2.line(extended, (w + 10, y_offset - 15), (w + panel_width - 10, y_offset - 15), (100, 100, 100), 1)
    cv2.putText(extended, f"Total: {total}", (w + 15, y_offset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Sección de advertencias (productos no reconocidos)
    if warnings:
        y_offset += 45
        cv2.putText(extended, "Revisar:", (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        y_offset += 25

        # Agrupar warnings por clase
        warn_counts = defaultdict(list)
        for item in warnings:
            warn_counts[item["class"]].append(item["conf"])

        for class_name in sorted(warn_counts.keys()):
            confs = warn_counts[class_name]
            display_name = class_name.replace("_", " ")
            avg_conf = sum(confs) / len(confs)
            if len(confs) > 1:
                text = f"? {display_name} x{len(confs)} ({avg_conf:.0%})"
            else:
                text = f"? {display_name} ({avg_conf:.0%})"
            cv2.putText(extended, text, (w + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1)
            y_offset += 24

    # Instrucciones al pie
    cv2.putText(extended, "Q: salir", (w + 15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return extended


def run_demo(
    model_path: Path,
    source: str,
    conf_threshold: float = 0.5,
    warn_threshold: float = 0.15,
    confirm_frames: int = 5,
    save_video: bool = True,
    output_json: Path | None = None,
) -> dict:
    """
    Ejecuta la demo con tracking y visualización.

    Args:
        model_path: Path al modelo YOLO
        source: Video file, directorio, o "0" para webcam
        conf_threshold: Umbral de confianza mínimo para detecciones confirmadas
        warn_threshold: Umbral de confianza para advertencias (productos no reconocidos)
        confirm_frames: Frames necesarios para confirmar una detección
        save_video: Si guardar el video con overlay
        output_json: Path para guardar JSON (opcional, se genera automáticamente si None)

    Returns:
        Diccionario con el estado final
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print("=" * 60)
    print("Demo Sistema de Autocobro")
    print("=" * 60)
    print(f"  Modelo:          {model_path}")
    print(f"  Fuente:          {source}")
    print(f"  Confianza:       {conf_threshold}")
    print(f"  Warn threshold:  {warn_threshold}")
    print(f"  Frames confirm:  {confirm_frames}")
    print(f"  Guardar video:   {save_video}")
    print("=" * 60)
    print()

    # Cargar modelo
    model = YOLO(str(model_path))
    class_names = model.names

    # Abrir fuente de video
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_webcam = True
    else:
        cap = cv2.VideoCapture(source)
        is_webcam = False

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_width = 250

    # Configurar output
    output_dir = Path(__file__).parent.parent / "runs" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_writer = None
    if save_video:
        video_path = output_dir / f"{timestamp_str}.mp4"
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (frame_width + panel_width, frame_height),
        )
        print(f"Video se guardará en: {video_path}")

    if output_json is None:
        output_json = output_dir / f"{timestamp_str}.json"

    # Inicializar tracker
    tracker = ProductTracker(confirm_frames=confirm_frames)

    print()
    print("Procesando... (presiona 'q' para terminar)")
    print()

    frame_num = 0
    final_counts = {}
    warning_counts = defaultdict(int)  # Acumular advertencias por clase

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Inferencia con umbral bajo para capturar advertencias
            results = model.predict(frame, conf=warn_threshold, verbose=False)

            # Extraer detecciones y separar en normales y advertencias
            detections = []
            current_warnings = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    det = {
                        "class": class_names[cls_id],
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                    }
                    if conf >= conf_threshold:
                        detections.append(det)
                    else:
                        current_warnings.append(det)
                        warning_counts[class_names[cls_id]] += 1

            # Actualizar tracker (solo con detecciones de alta confianza)
            counts = tracker.update(detections)
            final_counts = counts

            # Obtener items para visualización
            confirmed = tracker.get_confirmed_items()
            pending = tracker.get_pending_items()

            # Filtrar warnings: excluir detecciones que coinciden espacialmente con confirmados
            def overlaps_with_confirmed(warning_det, confirmed_items, iou_thresh=0.3):
                """Check if warning detection overlaps with any confirmed item of same class."""
                for conf_item in confirmed_items:
                    if conf_item["class"] != warning_det["class"]:
                        continue
                    # Calculate IoU
                    box1, box2 = warning_det["bbox"], conf_item["bbox"]
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0
                    if iou > iou_thresh:
                        return True
                return False

            current_warnings = [w for w in current_warnings if not overlaps_with_confirmed(w, confirmed)]

            # Dibujar frame
            display_frame = draw_frame(frame, confirmed, pending, current_warnings, counts, panel_width)

            # Mostrar
            cv2.imshow("Autocobro Demo", display_frame)

            # Guardar frame
            if video_writer is not None:
                video_writer.write(display_frame)

            # Salir con 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    # Filtrar advertencias significativas (vistas en al menos 10% de frames)
    min_warn_frames = max(5, frame_num // 10)
    significant_warnings = {
        cls: count for cls, count in warning_counts.items()
        if count >= min_warn_frames and cls not in final_counts
    }

    # Guardar JSON
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "source": str(source),
        "model": str(model_path),
        "conf_threshold": conf_threshold,
        "warn_threshold": warn_threshold,
        "frames_processed": frame_num,
        "productos": final_counts,
        "total_items": sum(final_counts.values()),
        "advertencias": significant_warnings,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("Resultados finales:")
    print("=" * 60)
    if final_counts:
        for class_name, count in sorted(final_counts.items()):
            print(f"  {class_name}: {count}")
    else:
        print("  (ningún producto detectado)")
    print(f"  Total: {sum(final_counts.values())}")

    if significant_warnings:
        print()
        print("Advertencias (revisar):")
        for class_name, frames_seen in sorted(significant_warnings.items()):
            print(f"  ? {class_name} (visto en {frames_seen} frames)")

    print()
    print(f"JSON guardado en: {output_json}")
    if save_video:
        print(f"Video guardado en: {output_dir / f'{timestamp_str}.mp4'}")
    print("=" * 60)

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="Demo del sistema de autocobro con visualización y tracking"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path al modelo YOLO (.pt)",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Video file, directorio, o '0' para webcam",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Umbral de confianza mínimo (default: 0.5)",
    )
    parser.add_argument(
        "--warn-conf",
        type=float,
        default=0.15,
        help="Umbral para advertencias de productos no reconocidos (default: 0.15)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=5,
        help="Frames consecutivos para confirmar detección (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path para archivo JSON de salida (opcional)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar video de salida",
    )

    args = parser.parse_args()

    run_demo(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        warn_threshold=args.warn_conf,
        confirm_frames=args.confirm_frames,
        save_video=not args.no_save,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
