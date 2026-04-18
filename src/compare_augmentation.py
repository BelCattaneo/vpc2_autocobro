#!/usr/bin/env python3
"""
Comparación de estrategias de data augmentation para YOLOv10n.

Entrena 3 configuraciones sobre el mismo dataset y compara métricas:
  1. Sin augmentation (baseline)
  2. Augmentation por defecto de Ultralytics
  3. Augmentation agresivo (mosaic + mixup + rotación + más HSV jitter)

Uso:
    python compare_augmentation.py --data data/train_v5.yaml
    python compare_augmentation.py --data data/train_v5.yaml --epochs 50
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def get_device() -> str:
    """Auto-detecta el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Configuraciones de augmentation a comparar
CONFIGS = {
    "sin_augmentation": {
        "description": "Sin augmentation — baseline para medir impacto",
        "params": {
            "augment": False,
            "mosaic": 0.0,
            "mixup": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "scale": 0.0,
            "translate": 0.0,
        },
    },
    "default_ultralytics": {
        "description": "Augmentation por defecto de Ultralytics (mosaic, flip, HSV)",
        "params": {
            "mosaic": 1.0,
            "mixup": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "scale": 0.5,
            "translate": 0.1,
        },
    },
    "augmentation_agresivo": {
        "description": "Augmentation agresivo (mosaic + mixup + rotación + HSV fuerte)",
        "params": {
            "mosaic": 1.0,
            "mixup": 0.3,
            "flipud": 0.5,
            "fliplr": 0.5,
            "hsv_h": 0.02,
            "hsv_s": 0.9,
            "hsv_v": 0.5,
            "degrees": 15.0,
            "scale": 0.7,
            "translate": 0.2,
        },
    },
}


def train_config(
    config_name: str,
    config: dict,
    data_yaml: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    project: str,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Entrena una configuración y retorna métricas."""
    if device is None:
        device = get_device()

    print()
    print("=" * 60)
    print(f"Configuración: {config_name}")
    print(f"  {config['description']}")
    print(f"  Device: {device}")
    print(f"  Parámetros: {config['params']}")
    print("=" * 60)

    model = YOLO("yolov10n.pt")

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": config_name,
        "patience": 20,
        "save": True,
        "plots": True,
        "verbose": False,
        "seed": seed,
        "device": device,
        **config["params"],
    }
    if device == "mps":
        train_kwargs["amp"] = False

    results = model.train(**train_kwargs)

    # Evaluar sobre test split — usar save_dir real de Ultralytics
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))

    val_results = best_model.val(
        data=str(data_yaml),
        split="test",
        plots=True,
        verbose=False,
    )

    metrics = {
        "config": config_name,
        "description": config["description"],
        "params": config["params"],
        "validation": {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        },
        "test": {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        },
    }

    print(f"\n  Resultados {config_name}:")
    print(f"    Val  — mAP@0.5: {metrics['validation']['mAP50']:.4f}")
    print(f"    Test — mAP@0.5: {metrics['test']['mAP50']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compara estrategias de data augmentation para YOLOv10n"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data/train_v5.yaml"),
        help="Path al data.yaml (default: data/train_v5.yaml)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Épocas por configuración (default: 100)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen (default: 640)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/augmentation_comparison",
        help="Directorio de resultados (default: runs/augmentation_comparison)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidad (default: 42)",
    )
    args = parser.parse_args()

    device = get_device()

    print("=" * 60)
    print("Comparación de Data Augmentation — YOLOv10n")
    print("=" * 60)
    print(f"  Dataset:  {args.data}")
    print(f"  Épocas:   {args.epochs}")
    print(f"  Device:   {device}")
    print(f"  Configs:  {list(CONFIGS.keys())}")
    print("=" * 60)

    all_metrics = []

    for name, config in CONFIGS.items():
        metrics = train_config(
            config_name=name,
            config=config,
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            seed=args.seed,
            device=device,
        )
        all_metrics.append(metrics)

    # Resumen comparativo
    print()
    print("=" * 60)
    print("RESUMEN COMPARATIVO")
    print("=" * 60)
    print(f"{'Configuración':<28} {'Val mAP@0.5':>12} {'Test mAP@0.5':>13} {'Test P':>8} {'Test R':>8}")
    print("-" * 69)
    for m in all_metrics:
        print(
            f"{m['config']:<28} "
            f"{m['validation']['mAP50']:>12.4f} "
            f"{m['test']['mAP50']:>13.4f} "
            f"{m['test']['precision']:>8.4f} "
            f"{m['test']['recall']:>8.4f}"
        )
    print("=" * 60)

    # Guardar resultados
    output_path = Path(args.project) / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "dataset": str(args.data),
                "epochs": args.epochs,
                "results": all_metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
