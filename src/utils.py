"""Utilidades compartidas para los scripts del proyecto."""

from pathlib import Path
from typing import Any

import torch
import yaml


# Configuraciones de data augmentation reutilizables
AUGMENTATION_CONFIGS: dict[str, dict[str, Any]] = {
    "none": {
        "description": "Sin augmentation",
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
    "default": {
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
    "aggressive": {
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


def get_device() -> str:
    """Auto-detecta el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data_yaml(data_yaml: Path) -> Path:
    """Resuelve path: . en data.yaml al directorio real del archivo.

    Ultralytics resuelve el campo 'path' del data.yaml relativo al cwd,
    no al directorio del yaml. Esta función genera una copia temporal
    con el path absoluto correcto.
    """
    data_yaml = data_yaml.resolve()
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    if config.get("path") == ".":
        config["path"] = str(data_yaml.parent)
        resolved = data_yaml.parent / f".{data_yaml.stem}_resolved.yaml"
        with open(resolved, "w") as f:
            yaml.dump(config, f)
        return resolved
    return data_yaml
