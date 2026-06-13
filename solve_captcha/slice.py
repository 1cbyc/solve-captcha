"""Split a grid CAPTCHA image into cell crops."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    image: np.ndarray
    x: int
    y: int
    w: int
    h: int


def slice_grid(
    image: np.ndarray,
    rows: int,
    cols: int,
    *,
    padding: int = 0,
) -> list[Cell]:
    if rows < 1 or cols < 1:
        raise ValueError(f"rows/cols must be >= 1, got {rows}x{cols}")

    h, w = image.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    if cell_h < 1 or cell_w < 1:
        raise ValueError(f"image too small for {rows}x{cols} grid ({w}x{h})")

    cells: list[Cell] = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h + padding
            x0 = c * cell_w + padding
            y1 = (r + 1) * cell_h - padding if r < rows - 1 else h
            x1 = (c + 1) * cell_w - padding if c < cols - 1 else w
            crop = image[y0:y1, x0:x1].copy()
            cells.append(Cell(row=r, col=c, image=crop, x=x0, y=y0, w=x1 - x0, h=y1 - y0))

    return cells


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def save_cells(cells: list[Cell], out_dir: str | Path, prefix: str = "cell") -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cell in cells:
        p = out_dir / f"{prefix}_r{cell.row}_c{cell.col}.png"
        cv2.imwrite(str(p), cell.image)
        paths.append(p)
    return paths
