#!/usr/bin/env python3
"""
Script de evaluación formal para modelos de detección (YOLO, RT-DETR, etc.).

Genera métricas de detección (mAP, precision, recall por clase),
confusion matrix, curvas precision-recall y F1, y guarda todo en JSON.

Uso:
    python evaluate.py --model models/best_multiproduct.pt --data data/poc_multiproduct/data.yaml
    python evaluate.py --model models/best.pt --data data/poc_10_clases/data.yaml --split val
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def evaluate_model(
    model_path: Path,
    data_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    project: str = "evaluate",
    name: str | None = None,
) -> dict:
    """
    Evalúa un modelo de detección y genera métricas y plots.

    Args:
        model_path: Path al modelo entrenado (.pt)
        data_yaml: Path al archivo data.yaml del dataset
        split: Split a evaluar ('val' o 'test')
        imgsz: Tamaño de imagen
        batch: Tamaño del batch
        conf: Umbral de confianza mínimo para detecciones
        iou: Umbral de IoU para NMS
        project: Directorio para guardar resultados
        name: Nombre del experimento (default: timestamp)

    Returns:
        Diccionario con métricas de evaluación
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {data_yaml}")

    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    model_name = Path(model_path).stem
    print(f"Evaluación de modelo: {model_name}")
    print("=" * 60)
    print(f"  Modelo:    {model_path}")
    print(f"  Dataset:   {data_yaml}")
    print(f"  Split:     {split}")
    print(f"  Resultados: {project}/{name}")
    print("=" * 60)
    print()

    model = YOLO(str(model_path))

    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        project=project,
        name=name,
        plots=True,      # genera confusion_matrix.png, PR_curve.png, F1_curve.png
        save_json=True,  # guarda predicciones en formato COCO JSON
        verbose=True,
    )

    # Extraer métricas del objeto results
    metrics = {
        "model": str(model_path),
        "data": str(data_yaml),
        "split": split,
        "timestamp": datetime.now().isoformat(),
        "overall": {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        },
        "per_class": {},
    }

    # Métricas por clase
    class_names = results.names
    if hasattr(results.box, "ap_class_index") and results.box.ap_class_index is not None:
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = class_names[int(class_idx)]
            metrics["per_class"][class_name] = {
                "mAP50": float(results.box.ap50[i]),
                "mAP50_95": float(results.box.ap[i]),
            }

    # Guardar métricas en JSON
    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Imprimir resumen
    print()
    print("=" * 60)
    print("Resultados")
    print("=" * 60)
    print(f"  mAP@0.5:       {metrics['overall']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:  {metrics['overall']['mAP50_95']:.4f}")
    print(f"  Precision:     {metrics['overall']['precision']:.4f}")
    print(f"  Recall:        {metrics['overall']['recall']:.4f}")
    print()

    if metrics["per_class"]:
        print("  Por clase (mAP@0.5):")
        for cls, vals in sorted(metrics["per_class"].items(), key=lambda x: x[1]["mAP50"], reverse=True):
            print(f"    {cls:<25} {vals['mAP50']:.4f}")
        print()

    print(f"  Métricas guardadas en: {metrics_path}")
    print(f"  Plots guardados en:    {output_dir}/")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo de detección y genera métricas formales"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path al modelo entrenado (.pt)"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data/poc_multiproduct/data.yaml"),
        help="Path al data.yaml del dataset (default: data/poc_multiproduct/data.yaml)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split a evaluar (default: val)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen (default: 640)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Tamaño del batch (default: 16)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Umbral de confianza mínimo (default: 0.001)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="Umbral de IoU para NMS (default: 0.6)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="evaluate",
        help="Directorio para guardar resultados (default: evaluate)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Nombre del experimento (default: timestamp)"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
