"""Run ONNX or local OCR on grid cell images."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .slice import Cell


@dataclass(frozen=True)
class Prediction:
    row: int
    col: int
    label: str
    confidence: float


class OnnxClassifier:
    def __init__(self, model_path: str | Path, labels: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta()
        raw = meta.custom_metadata_map.get("labels", "")
        self.labels = labels or ([s.strip() for s in raw.split(",") if s.strip()] if raw else [])

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        h = shape[2] if isinstance(shape[2], int) else 224
        w = shape[3] if isinstance(shape[3], int) else 224
        resized = cv2.resize(image, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, 0)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        out = self.session.run(None, {self.input_name: self._preprocess(image)})[0]
        scores = out[0]
        idx = int(np.argmax(scores))
        conf = float(scores[idx])
        if self.labels and idx < len(self.labels):
            return self.labels[idx], conf
        return str(idx), conf


class OcrEngine:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        result, _ = self.engine(image)
        if not result:
            return "", 0.0
        texts = [r[1] for r in result]
        scores = [float(r[2]) for r in result]
        return " ".join(texts).strip().lower(), max(scores)


def infer_cells(
    cells: list[Cell],
    *,
    onnx_path: str | Path | None = None,
    labels: list[str] | None = None,
    use_ocr: bool = False,
) -> list[Prediction]:
    if onnx_path:
        model = OnnxClassifier(onnx_path, labels)
        predict = model.predict
    elif use_ocr:
        model = OcrEngine()
        predict = model.predict
    else:
        raise ValueError("need onnx_path or use_ocr=True")

    preds: list[Prediction] = []
    for cell in cells:
        label, conf = predict(cell.image)
        preds.append(Prediction(row=cell.row, col=cell.col, label=label.lower(), confidence=conf))
    return preds
