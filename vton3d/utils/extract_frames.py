# vton3d/utils/extract_frames.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}


@dataclass
class ExtractFramesConfig:
    num_frames: int
    start: int = 0
    end: Optional[int] = None
    ext: str = "png"
    overwrite: bool = False
    rotate: int = 0
    prefix: Optional[str] = None
    clear_output_dir: bool = False

@dataclass
class ExtractFramesResult:
    video_path: Path
    out_images_dir: Path
    requested_frames: int
    saved_frames: int
    start: int
    end: int
    rotate: int
    total_frames: int


def _resolve_rotate_code(rotate: int):
    if rotate not in (0, 90, 180, 270):
        raise ValueError(f"rotate must be one of [0, 90, 180, 270], got {rotate}")
    if rotate == 0:
        return None
    if rotate == 90:
        return cv2.ROTATE_90_CLOCKWISE
    if rotate == 180:
        return cv2.ROTATE_180
    if rotate == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    return None


def list_videos(videos_dir: Path) -> list[Path]:
    videos_dir = videos_dir.expanduser().resolve()
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    return videos


def _clear_dir(dir_path: Path):
    if not dir_path.exists():
        return
    for p in dir_path.iterdir():
        if p.is_file():
            p.unlink()


def extract_frames_from_video(
    video_path: Path,
    out_images_dir: Path,
    cfg: ExtractFramesConfig,
) -> ExtractFramesResult:
    video_path = video_path.expanduser().resolve()
    out_images_dir = out_images_dir.expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if cfg.num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {cfg.num_frames}")

    rotate_code = _resolve_rotate_code(cfg.rotate)

    out_images_dir.mkdir(parents=True, exist_ok=True)

    if cfg.clear_output_dir:
        _clear_dir(out_images_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video seems to have 0 frames: {video_path}")

    end = cfg.end
    if end is None or end > total_frames:
        end = total_frames

    start = cfg.start
    if start < 0 or start >= end:
        cap.release()
        raise ValueError(
            f"Invalid range for {video_path.name}: start={start}, end={end}, total={total_frames}"
        )

    prefix = cfg.prefix or video_path.stem

    digits = max(4, len(str(max(0, end - 1))))

    frame_range = end - start
    if cfg.num_frames >= frame_range:
        target_indices = list(range(start, end))
    else:
        step = frame_range / cfg.num_frames
        indices = []
        for i in range(cfg.num_frames):
            idx = int(start + i * step)
            if idx >= end:
                idx = end - 1
            indices.append(idx)
        target_indices = sorted(set(indices))

    actual_target_count = len(target_indices)

    saved = 0
    target_pos = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    current = start

    while current < end and target_pos < actual_target_count:
        ret, frame = cap.read()
        if not ret:
            break

        target_idx = target_indices[target_pos]
        if current == target_idx:
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            filename = f"{prefix}_{current:0{digits}d}.{cfg.ext.lower()}"
            out_path = out_images_dir / filename

            if out_path.exists() and not cfg.overwrite:
                pass
            else:
                ok = cv2.imwrite(str(out_path), frame)
                if not ok:
                    cap.release()
                    raise RuntimeError(f"Failed to write frame to {out_path}")
                saved += 1

            target_pos += 1

        current += 1

    cap.release()

    return ExtractFramesResult(
        video_path=video_path,
        out_images_dir=out_images_dir,
        requested_frames=cfg.num_frames,
        saved_frames=saved,
        start=start,
        end=end,
        rotate=cfg.rotate,
        total_frames=total_frames,
    )

def extract_frames_to_scene_dir(
    video_path: Path,
    scene_dir: Path,
    cfg: ExtractFramesConfig,
) -> ExtractFramesResult:
    images_dir = scene_dir / "real" / "images"
    return extract_frames_from_video(video_path=video_path, out_images_dir=images_dir, cfg=cfg)
