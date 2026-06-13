"""Map matched grid cells to click coordinates."""

from dataclasses import dataclass

from .infer import Prediction
from .slice import Cell


@dataclass(frozen=True)
class Click:
    x: int
    y: int
    row: int
    col: int
    label: str
    confidence: float


def cell_center(cell: Cell) -> tuple[int, int]:
    return cell.x + cell.w // 2, cell.y + cell.h // 2


def match_predictions(
    cells: list[Cell],
    predictions: list[Prediction],
    target: str,
    *,
    min_confidence: float = 0.0,
    exact: bool = False,
) -> list[Click]:
    target = target.strip().lower()
    cell_map = {(c.row, c.col): c for c in cells}
    clicks: list[Click] = []

    for pred in predictions:
        if pred.confidence < min_confidence:
            continue
        hit = pred.label == target if exact else target in pred.label
        if not hit:
            continue
        cell = cell_map[(pred.row, pred.col)]
        x, y = cell_center(cell)
        clicks.append(Click(x=x, y=y, row=pred.row, col=pred.col, label=pred.label, confidence=pred.confidence))

    return clicks
