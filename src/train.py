#!/usr/bin/env python3
"""
Script de entrenamiento YOLO.

Uso:
    python train.py                                        # Entrena con configuración por defecto
    python train.py --model yolo11s.pt                     # Usa modelo small
    python train.py --augmentation none                    # Sin data augmentation
    python train.py --augmentation aggressive              # Augmentation agresivo
    python train.py --epochs 200 --batch 32                # Personaliza hiperparámetros
"""

import argparse
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

from utils import AUGMENTATION_CONFIGS, get_device, resolve_data_yaml


BASE_MODELS_DIR = Path(__file__).parent.parent / "models" / "base"


def train_model(
    data_yaml: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/train",
    name: str | None = None,
    seed: int = 42,
    augmentation: str = "default",
) -> Path:
    """
    Entrena un modelo YOLO en el dataset especificado.

    Args:
        data_yaml: Path al archivo data.yaml del dataset
        model_name: Modelo base a usar (yolo11n/s/m/l/x)
        epochs: Número de épocas de entrenamiento
        imgsz: Tamaño de imagen para entrenamiento
        batch: Tamaño del batch
        project: Directorio del proyecto para guardar resultados
        name: Nombre del experimento (default: timestamp)
        seed: Seed para reproducibilidad (default: 42)
        augmentation: Estrategia de augmentation (none/default/aggressive)

    Returns:
        Path al mejor modelo entrenado
    """
    if not data_yaml.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_yaml}")
    if augmentation not in AUGMENTATION_CONFIGS:
        raise ValueError(f"Augmentation '{augmentation}' no válido. Opciones: {list(AUGMENTATION_CONFIGS.keys())}")

    if name is None:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = get_device()
    aug_config = AUGMENTATION_CONFIGS[augmentation]

    print("=" * 60)
    print("Entrenamiento YOLO")
    print("=" * 60)
    print(f"  Modelo base:    {model_name}")
    print(f"  Dataset:        {data_yaml}")
    print(f"  Augmentation:   {augmentation} ({aug_config['description']})")
    print(f"  Device:         {device}")
    print(f"  Épocas:         {epochs}")
    print(f"  Imagen:         {imgsz}x{imgsz}")
    print(f"  Batch:          {batch}")
    print(f"  Seed:           {seed}")
    print(f"  Proyecto:       {project}/{name}")
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

    train_kwargs = {
        "data": str(data_resolved),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "name": name,
        "patience": 20,
        "save": True,
        "plots": True,
        "verbose": True,
        "seed": seed,
        "device": device,
        **aug_config["params"],
    }
    if device == "mps":
        train_kwargs["amp"] = False

    results = model.train(**train_kwargs)

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
        description="Entrena un modelo YOLO para detección de productos"
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
        default="yolo11n.pt",
        help="Modelo base a usar (default: yolo11n.pt)"
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
    parser.add_argument(
        "--augmentation",
        type=str,
        default="default",
        choices=list(AUGMENTATION_CONFIGS.keys()),
        help="Estrategia de data augmentation (default: default)"
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
        augmentation=args.augmentation,
    )

    print(f"Para usar el modelo entrenado:")
    print(f"  python inference.py --model {best_model} --source <imagen_o_video>")


if __name__ == "__main__":
    main()
