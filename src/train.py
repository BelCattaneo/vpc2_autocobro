#!/usr/bin/env python3
"""
Script de entrenamiento para YOLOv10.

Uso:
    python train.py                           # Entrena con configuración por defecto
    python train.py --model yolov10s.pt       # Usa modelo small
    python train.py --epochs 200 --batch 32   # Personaliza hiperparámetros
"""

import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from utils import resolve_data_yaml


BASE_MODELS_DIR = Path(__file__).parent.parent / "models" / "base"


def train_model(
    data_yaml: Path,
    model_name: str = "yolov10n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str | None = None,
    seed: int = 42,
) -> Path:
    """
    Entrena un modelo YOLOv10 en el dataset especificado.

    Args:
        data_yaml: Path al archivo data.yaml del dataset
        model_name: Modelo base a usar (yolov10n/s/m/b/l/x)
        epochs: Número de épocas de entrenamiento
        imgsz: Tamaño de imagen para entrenamiento
        batch: Tamaño del batch
        project: Directorio del proyecto para guardar resultados
        name: Nombre del experimento (default: timestamp)
        seed: Seed para reproducibilidad (default: 42)

    Returns:
        Path al mejor modelo entrenado
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_yaml}")

    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Entrenamiento YOLOv10")
    print("=" * 60)
    print(f"  Modelo base:  {model_name}")
    print(f"  Dataset:      {data_yaml}")
    print(f"  Épocas:       {epochs}")
    print(f"  Imagen:       {imgsz}x{imgsz}")
    print(f"  Batch:        {batch}")
    print(f"  Proyecto:     {project}/{name}")
    print("=" * 60)
    print()

    # Cargar modelo base desde models/base/ si existe, sino descargar
    model_path = BASE_MODELS_DIR / model_name
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        print(f"Modelo base no encontrado en {model_path}, descargando...")
        model = YOLO(model_name)

    data_resolved = resolve_data_yaml(data_yaml)

    results = model.train(
        data=str(data_resolved),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        seed=seed,
    )

    best_model_path = Path(project) / name / "weights" / "best.pt"

    print()
    print("=" * 60)
    print("Entrenamiento completado")
    print("=" * 60)
    print(f"  Mejor modelo: {best_model_path}")
    print()

    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo YOLOv10 para detección de productos"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=Path("data/poc_10_clases/data.yaml"),
        help="Path al archivo data.yaml (default: data/poc_10_clases/data.yaml)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov10n.pt",
        choices=["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10b.pt", "yolov10l.pt", "yolov10x.pt"],
        help="Modelo base a usar (default: yolov10n.pt)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Número de épocas (default: 100)"
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
        "--project",
        type=str,
        default="runs/train",
        help="Directorio del proyecto (default: runs/train)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Nombre del experimento (default: timestamp)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidad (default: 42)"
    )

    args = parser.parse_args()

    best_model = train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )

    print(f"Para usar el modelo entrenado:")
    print(f"  python inference.py --model {best_model} --source <imagen_o_video>")


if __name__ == "__main__":
    main()
