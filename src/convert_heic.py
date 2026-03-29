#!/usr/bin/env python3
"""
Convierte imágenes HEIC a JPG para su uso con YOLO.

Uso:
    python convert_heic.py                    # Convierte data/ -> data_jpg/
    python convert_heic.py --input data --output data_jpg
"""

import argparse
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener


def convert_heic_to_jpg(input_dir: Path, output_dir: Path) -> dict:
    """
    Convierte todas las imágenes HEIC en input_dir a JPG en output_dir.
    Mantiene la estructura de carpetas por clase.

    Args:
        input_dir: Directorio con imágenes HEIC organizadas por clase
        output_dir: Directorio de salida para imágenes JPG

    Returns:
        Dict con estadísticas de conversión
    """
    register_heif_opener()

    stats = {"converted": 0, "skipped": 0, "errors": 0}

    heic_files = list(input_dir.glob("**/*.HEIC")) + list(input_dir.glob("**/*.heic"))

    if not heic_files:
        print(f"No se encontraron archivos HEIC en {input_dir}")
        return stats

    print(f"Encontradas {len(heic_files)} imágenes HEIC")

    for heic_path in heic_files:
        relative_path = heic_path.relative_to(input_dir)
        jpg_path = output_dir / relative_path.with_suffix(".jpg")

        jpg_path.parent.mkdir(parents=True, exist_ok=True)

        if jpg_path.exists():
            print(f"  Saltando (ya existe): {relative_path}")
            stats["skipped"] += 1
            continue

        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)
            print(f"  Convertido: {relative_path}")
            stats["converted"] += 1
        except Exception as e:
            print(f"  Error en {relative_path}: {e}")
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convierte imágenes HEIC a JPG para YOLO"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/heic"),
        help="Directorio de entrada con imágenes HEIC (default: data/heic)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/jpg"),
        help="Directorio de salida para imágenes JPG (default: data/jpg)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: No existe el directorio {args.input}")
        return 1

    print(f"Convirtiendo HEIC -> JPG")
    print(f"  Entrada: {args.input.absolute()}")
    print(f"  Salida:  {args.output.absolute()}")
    print()

    stats = convert_heic_to_jpg(args.input, args.output)

    print()
    print("Resumen:")
    print(f"  Convertidas: {stats['converted']}")
    print(f"  Saltadas:    {stats['skipped']}")
    print(f"  Errores:     {stats['errors']}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())
