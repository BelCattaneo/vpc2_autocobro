"""Utilidades compartidas para los scripts del proyecto."""

from pathlib import Path

import torch
import yaml


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
