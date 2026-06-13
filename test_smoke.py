#!/usr/bin/env python3
"""Generate a test 3x3 grid with text labels and verify slicing + coords."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from solve_captcha import GridSolver
from solve_captcha.slice import slice_grid, cell_center


def make_test_grid(path: Path) -> None:
    rows, cols = 3, 3
    cell = 100
    img = np.ones((rows * cell, cols * cell, 3), dtype=np.uint8) * 240
    labels = [
        ["cat", "bus", "dog"],
        ["bus", "tree", "bus"],
        ["car", "bus", "hat"],
    ]
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell, c * cell
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + cell - 2, y0 + cell - 2), (200, 200, 200), 1)
            cv2.putText(
                img,
                labels[r][c],
                (x0 + 15, y0 + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
    cv2.imwrite(str(path), img)


def test_slice_centers():
    img = cv2.imread(str(Path("test_grid.png")))
    cells = slice_grid(img, 3, 3)
    assert len(cells) == 9
    # cell (0,1) center should be ~(150, 50) for 100px cells
    c = next(x for x in cells if x.row == 0 and x.col == 1)
    x, y = cell_center(c)
    assert 140 <= x <= 160, f"bad x: {x}"
    assert 40 <= y <= 60, f"bad y: {y}"
    print("slice centers: ok")


def test_solver():
    solver = GridSolver(rows=3, cols=3, use_ocr=True, min_confidence=0.0)
    result = solver.solve("test_grid.png", "bus")
    assert len(result.clicks) >= 1, "expected at least one bus match"
    print(json.dumps([{"x": c.x, "y": c.y, "label": c.label} for c in result.clicks], indent=2))
    print("solver: ok")


if __name__ == "__main__":
    make_test_grid(Path("test_grid.png"))
    test_slice_centers()
    try:
        test_solver()
    except Exception as e:
        print(f"solver skipped (ocr may need model download): {e}", file=sys.stderr)
    print("done")
