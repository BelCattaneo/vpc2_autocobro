#!/usr/bin/env python3
"""
Extrae frames de videos para crear dataset de entrenamiento.

Uso:
    # Extraer 1 frame cada 15 frames (2 fps de un video de 30fps)
    uv run python src/extract_frames.py --input data/videos/leche.MOV --output data/frames/leche/ --every 15

    # Extraer frames de todos los videos en una carpeta
    uv run python src/extract_frames.py --input data/videos/ --output data/frames/ --every 10

    # Extraer máximo 50 frames por video
    uv run python src/extract_frames.py --input data/videos/leche.MOV --output data/frames/leche/ --max-frames 50
"""

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 10,
    max_frames: int | None = None,
) -> int:
    """
    Extrae frames de un video.

    Args:
        video_path: Path al video
        output_dir: Directorio donde guardar los frames
        every_n_frames: Extraer 1 frame cada N frames
        max_frames: Máximo de frames a extraer (None = sin límite)

    Returns:
        Número de frames extraídos
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: No se pudo abrir {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Video: {video_path.name}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"  Extrayendo 1 de cada {every_n_frames} frames...")

    frame_num = 0
    extracted = 0
    video_name = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % every_n_frames == 0:
            output_path = output_dir / f"{video_name}_{frame_num:05d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted += 1

            if max_frames and extracted >= max_frames:
                break

        frame_num += 1

    cap.release()
    print(f"  Extraídos: {extracted} frames -> {output_dir}")
    return extracted


def process_videos(
    input_path: Path,
    output_path: Path,
    every_n_frames: int = 10,
    max_frames: int | None = None,
) -> dict:
    """
    Procesa uno o varios videos.

    Args:
        input_path: Video individual o directorio con videos
        output_path: Directorio de salida
        every_n_frames: Extraer 1 frame cada N frames
        max_frames: Máximo de frames por video

    Returns:
        Diccionario con estadísticas
    """
    video_extensions = {".mov", ".mp4", ".avi", ".mkv", ".MOV", ".MP4"}
    stats = {"videos": 0, "frames": 0}

    if input_path.is_file():
        # Un solo video
        frames = extract_frames(input_path, output_path, every_n_frames, max_frames)
        stats["videos"] = 1
        stats["frames"] = frames

    elif input_path.is_dir():
        # Directorio con videos
        videos = [f for f in input_path.iterdir() if f.suffix in video_extensions]

        if not videos:
            print(f"No se encontraron videos en {input_path}")
            return stats

        print(f"Encontrados {len(videos)} videos en {input_path}")
        print()

        for video in sorted(videos):
            # Crear subdirectorio con el nombre del video
            video_output_dir = output_path / video.stem
            frames = extract_frames(video, video_output_dir, every_n_frames, max_frames)
            stats["videos"] += 1
            stats["frames"] += frames
            print()

    else:
        print(f"Error: {input_path} no existe")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extrae frames de videos para crear dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Video o directorio con videos",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Directorio de salida para los frames",
    )
    parser.add_argument(
        "--every", "-e",
        type=int,
        default=10,
        help="Extraer 1 frame cada N frames (default: 10)",
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Máximo de frames a extraer por video (default: sin límite)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Extractor de Frames para Dataset")
    print("=" * 60)
    print()

    stats = process_videos(
        input_path=args.input,
        output_path=args.output,
        every_n_frames=args.every,
        max_frames=args.max_frames,
    )

    print("=" * 60)
    print(f"Resumen: {stats['videos']} videos, {stats['frames']} frames extraídos")
    print("=" * 60)


if __name__ == "__main__":
    main()
