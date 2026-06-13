# grid captcha solver

splits a grid image, runs local inference, returns click coords.

## install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## usage

```bash
# OCR (default) — good for text captchas
python solve.py captcha.png "bus"

# 4x4 grid
python solve.py captcha.png "traffic light" -r 4 -c 4

# dump cell crops for debugging
python solve.py captcha.png "a" --dump ./cells

# ONNX classifier
python solve.py captcha.png "cat" --onnx models/classifier.onnx --labels "dog,cat,bus,car"

# fetch from URL
python solve.py https://example.com/captcha.jpg "bus" --download captcha.png
```

output is JSON on stdout:

```json
{
  "clicks": [{"x": 120, "y": 85, "row": 0, "col": 1, "label": "bus", "confidence": 0.91}],
  "predictions": [...]
}
```

## python api

```python
from solve_captcha import GridSolver

solver = GridSolver(rows=3, cols=3)
result = solver.solve("captcha.png", "bus")

for click in result.clicks:
    print(click.x, click.y)
```

## how it works

1. **slice** — OpenCV splits the image into `rows x cols` cells. click point = cell center.
2. **infer** — each cell goes through RapidOCR (default) or your ONNX model.
3. **match** — cells whose label contains the target string become clicks.

## onnx model

expects NCHW float32 input, `[0,1]` normalized RGB. put class names in model metadata key `labels` (comma-separated) or pass `--labels`.

for object detection (YOLO etc), export a classifier per-cell or wrap your own model in `solve_captcha/infer.py`.

## layout

```
solve_captcha/
  slice.py    # grid split
  infer.py    # onnx + ocr
  action.py   # click coords
  solver.py   # pipeline
solve.py      # cli
```
