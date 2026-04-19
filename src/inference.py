#!/usr/bin/env python3
"""
Script de inferencia YOLO.

Uso:
    python inference.py --model models/best.pt --source imagen.jpg
    python inference.py --model models/best.pt --source video.mp4
    python inference.py --model models/best.pt --source 0  # Webcam
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_inference(
    model_path: Path,
    source: str,
    conf: float = 0.5,
    save: bool = True,
    show: bool = False,
    project: str = "runs/detect",
    name: str = "predict",
) -> list:
    """
    Ejecuta inferencia en imágenes, video o webcam.

    Args:
        model_path: Path al modelo entrenado (.pt)
        source: Imagen, video, directorio, o ID de cámara (0, 1, ...)
        conf: Threshold de confianza mínimo
        save: Guardar resultados
        show: Mostrar resultados en pantalla
        project: Directorio del proyecto para guardar resultados
        name: Nombre del experimento

    Returns:
        Lista de resultados de detección
    """
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    print("=" * 60)
    print("Inferencia YOLO")
    print("=" * 60)
    print(f"  Modelo:     {model_path}")
    print(f"  Fuente:     {source}")
    print(f"  Confianza:  {conf}")
    print(f"  Guardar:    {save}")
    print(f"  Mostrar:    {show}")
    print("=" * 60)
    print()

    model = YOLO(str(model_path))

    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        show=show,
        project=project,
        name=name,
    )

    print()
    print("Detecciones:")
    print("-" * 40)

    for i, result in enumerate(results):
        boxes = result.boxes
        if len(boxes) == 0:
            print(f"  Imagen {i+1}: Sin detecciones")
            continue

        print(f"  Imagen {i+1}:")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])
            print(f"    - {cls_name}: {conf_val:.2%}")

    if save:
        print()
        print(f"Resultados guardados en: {project}/{name}")

    return results


def main():
    # Default project path (absolute)
    default_project = str(Path(__file__).parent.parent / "runs" / "detect")

    parser = argparse.ArgumentParser(
        description="Ejecuta inferencia con un modelo YOLO entrenado"
    )
    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path al modelo entrenado (.pt)"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Imagen, video, directorio, o ID de cámara (0, 1, ...)"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Threshold de confianza mínimo (default: 0.5)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Guardar resultados (default: True)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar resultados"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar resultados en pantalla"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=default_project,
        help="Directorio del proyecto (default: runs/detect)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default="predict",
        help="Nombre del experimento (default: predict)"
    )

    args = parser.parse_args()

    save = not args.no_save

    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save=save,
        show=args.show,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
