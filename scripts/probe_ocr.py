"""RapidOCR で test.mp4 の代表フレームを読んで結果をダンプする (動作確認用).

Usage:
    uv run python scripts/probe_ocr.py <frame.png> [<frame.png> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from ch_data_collector.ocr import ocr_image


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        path = Path(arg)
        img = cv2.imread(str(path))
        if img is None:
            print(f"failed to read: {path}")
            continue
        print(f"=== {path.name} ({img.shape[1]}x{img.shape[0]}) ===")
        results = ocr_image(img)
        for r in results:
            x0, y0 = r.box[0]
            x1, y1 = r.box[2]
            print(
                f"  [{x0:>4},{y0:>4} - {x1:>4},{y1:>4}] "
                f"conf={r.confidence:.2f}  text={r.text!r}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
