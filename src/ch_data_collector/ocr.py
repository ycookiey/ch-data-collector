"""EasyOCR (lang=['ja','en']) ラッパ.

技名・ポケモン名を画像のサブ矩形からテキスト抽出する。
EasyOCR はPyTorchベース、初回呼出でモデル自動DL。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import easyocr

from ch_data_collector.screen_layouts import Box


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    box: tuple[tuple[int, int], ...]  # 4頂点の (x, y) (入力画像座標系)


@lru_cache(maxsize=1)
def _engine() -> easyocr.Reader:
    import torch

    return easyocr.Reader(
        ["ja", "en"], gpu=torch.cuda.is_available(), verbose=False
    )


def ocr_image(image: np.ndarray) -> list[OcrResult]:
    engine = _engine()
    raw = engine.readtext(image)
    out: list[OcrResult] = []
    for box, text, conf in raw:
        pts = tuple((int(x), int(y)) for (x, y) in box)
        out.append(OcrResult(text=str(text), confidence=float(conf), box=pts))
    return out


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    return image[box.y : box.y + box.h, box.x : box.x + box.w]


def upscale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC
    )


def ocr_region(
    image: np.ndarray, box: Box, *, upscale_factor: float = 2.0
) -> list[OcrResult]:
    """指定矩形領域をクロップ + 拡大してOCR. 結果は元画像座標系に戻す."""
    sub = crop(image, box)
    if sub.size == 0:
        return []
    if upscale_factor != 1.0:
        sub_for_ocr = upscale(sub, upscale_factor)
    else:
        sub_for_ocr = sub
    raw = _engine().readtext(sub_for_ocr)
    out: list[OcrResult] = []
    for poly, text, conf in raw:
        pts = []
        for x, y in poly:
            ox = int(x / upscale_factor) + box.x
            oy = int(y / upscale_factor) + box.y
            pts.append((ox, oy))
        out.append(
            OcrResult(text=str(text), confidence=float(conf), box=tuple(pts))
        )
    return out


def joined_text(results: list[OcrResult]) -> str:
    """OCR結果を単一文字列に連結 (画面分類用)."""
    return "".join(r.text for r in results)


def recognize_in_regions(
    image: np.ndarray,
    boxes: list[Box],
    *,
    allowlist: str | None = None,
) -> list[list[OcrResult]]:
    """検出器をスキップし、指定矩形群の認識のみ実行する.

    EasyOCR の Reader.recognize() を直接呼ぶ. 内部で各 box が batch_size 単位
    で認識器に渡されるため、6スロット分が1回の GPU 推論にまとまる.

    Args:
      allowlist: 認識する文字を限定する. Champions の技名に出現する
        ひらがな・カタカナ・長音符・小書きに絞ると認識器が漢字や記号への
        誤認識を構造的に避けられる.
    """
    engine = _engine()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    horizontal_list = [
        [b.x, b.x + b.w, b.y, b.y + b.h] for b in boxes
    ]
    raw = engine.recognize(
        gray,
        horizontal_list=horizontal_list,
        free_list=[],
        detail=1,
        batch_size=len(boxes),
        allowlist=allowlist,
    )
    out: list[list[OcrResult]] = [[] for _ in boxes]
    if not raw:
        return out
    # EasyOCR の recognize() は結果を縦位置 (重心 y) でソートして返すことがある
    # (GPU バッチ経路は内部で y ソートする) ため、horizontal_list と必ずしも同順では
    # ない. ここで件数一致時に index 対応で割り当てられるのは、唯一の呼び出し元
    # read_rows が box を上から順 (y 昇順) に渡すため、y ソート後も順序が保たれる
    # からである (重心 y マッチだと重なり領域で手前の box が後続結果を総取りし、その
    # 行が脱落するので index 対応の方が安全).
    # NOTE: y 昇順でない box 群を渡す呼び出し元を追加する場合、ここで poly の重心 y と
    # box を突き合わせる照合が必要 (現状はこの y 昇順の前提に依存している).
    if len(raw) == len(boxes):
        for i, (poly, text, conf) in enumerate(raw):
            pts = (
                tuple((int(x), int(y)) for (x, y) in poly) if poly else ()
            )
            out[i].append(
                OcrResult(text=str(text), confidence=float(conf), box=pts)
            )
        return out
    # 件数が一致しない場合のフォールバック: 重心 y が最も近い box 中心へ割当て
    # (範囲内・最初一致の break ではなく最近傍にすることで重なりでも誤らない).
    for poly, text, conf in raw:
        if not poly:
            continue
        pts_arr = np.asarray(poly, dtype=float)
        cy = float(pts_arr[:, 1].mean())
        best_i, best_d = -1, float("inf")
        for i, b in enumerate(boxes):
            d = abs(cy - (b.y + b.h / 2.0))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            pts = tuple((int(x), int(y)) for (x, y) in poly)
            out[best_i].append(
                OcrResult(text=str(text), confidence=float(conf), box=pts)
            )
    return out
