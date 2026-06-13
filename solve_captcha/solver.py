"""Grid CAPTCHA solver pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .action import Click, match_predictions
from .infer import Prediction, infer_cells
from .slice import Cell, load_image, save_cells, slice_grid


@dataclass
class SolveResult:
    clicks: list[Click]
    cells: list[Cell]
    predictions: list[Prediction]
    image_shape: tuple[int, int]  # h, w


class GridSolver:
    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        *,
        padding: int = 0,
        onnx_path: str | Path | None = None,
        labels: list[str] | None = None,
        use_ocr: bool = True,
        min_confidence: float = 0.0,
        exact_match: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.padding = padding
        self.onnx_path = onnx_path
        self.labels = labels
        self.use_ocr = use_ocr
        self.min_confidence = min_confidence
        self.exact_match = exact_match

    def solve(
        self,
        image: str | Path | np.ndarray,
        target: str,
        *,
        dump_dir: str | Path | None = None,
    ) -> SolveResult:
        if isinstance(image, (str, Path)):
            img = load_image(image)
        else:
            img = image

        cells = slice_grid(img, self.rows, self.cols, padding=self.padding)
        if dump_dir:
            save_cells(cells, dump_dir)

        preds = infer_cells(
            cells,
            onnx_path=self.onnx_path,
            labels=self.labels,
            use_ocr=self.use_ocr if not self.onnx_path else False,
        )
        clicks = match_predictions(
            cells,
            preds,
            target,
            min_confidence=self.min_confidence,
            exact=self.exact_match,
        )

        h, w = img.shape[:2]
        return SolveResult(clicks=clicks, cells=cells, predictions=preds, image_shape=(h, w))
