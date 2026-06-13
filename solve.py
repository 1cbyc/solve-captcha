#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from solve_captcha import GridSolver


def main() -> int:
    p = argparse.ArgumentParser(description="solve grid captchas locally")
    p.add_argument("image", help="captcha image path or URL")
    p.add_argument("target", help="label to click (substring match)")
    p.add_argument("-r", "--rows", type=int, default=3)
    p.add_argument("-c", "--cols", type=int, default=3)
    p.add_argument("--padding", type=int, default=0, help="px trim per cell edge")
    p.add_argument("--onnx", help="path to ONNX classifier")
    p.add_argument("--labels", help="comma-separated class labels (onnx)")
    p.add_argument("--no-ocr", action="store_true", help="disable OCR fallback")
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--exact", action="store_true", help="exact label match")
    p.add_argument("--dump", help="dir to write cell crops")
    p.add_argument("--download", help="save fetched URL image here")
    args = p.parse_args()

    image_path = args.image
    if image_path.startswith(("http://", "https://")):
        image_path = _fetch(image_path, args.download)

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    solver = GridSolver(
        rows=args.rows,
        cols=args.cols,
        padding=args.padding,
        onnx_path=args.onnx,
        labels=labels,
        use_ocr=not args.no_ocr,
        min_confidence=args.min_conf,
        exact_match=args.exact,
    )

    result = solver.solve(image_path, args.target, dump_dir=args.dump)

    out = {
        "target": args.target,
        "grid": f"{args.rows}x{args.cols}",
        "image_size": {"h": result.image_shape[0], "w": result.image_shape[1]},
        "clicks": [{"x": c.x, "y": c.y, "row": c.row, "col": c.col, "label": c.label, "confidence": c.confidence} for c in result.clicks],
        "predictions": [{"row": p.row, "col": p.col, "label": p.label, "confidence": p.confidence} for p in result.predictions],
    }
    print(json.dumps(out, indent=2))
    return 0


def _fetch(url: str, dest: str | None) -> str:
    import urllib.request

    data = urllib.request.urlopen(url, timeout=30).read()
    path = Path(dest) if dest else Path("captcha_download.png")
    path.write_bytes(data)
    return str(path)


if __name__ == "__main__":
    sys.exit(main())
