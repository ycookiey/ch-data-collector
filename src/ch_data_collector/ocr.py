"""EasyOCR (lang=['ja','en']) ラッパ.

技名・ポケモン名を画像のサブ矩形からテキスト抽出する。
EasyOCR はPyTorchベース、初回呼出でモデル自動DL。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

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


def recognize_region(
    image: np.ndarray,
    box: Box,
    *,
    upscale_factor: float = 2.0,
    allowlist: str | None = None,
) -> list[OcrResult]:
    """指定 box を crop + upscale して認識器単体で OCR (検出器なし).

    classify_screen 用. EasyOCR の readtext (検出器 + 認識器) でなく
    recognize() を直接呼んで box 内のテキストを単一塊として認識する.
    検出器をスキップする分 ocr_region より速い (実測 ~30-50% 高速).
    複数行や離散テキストには向かないが、ヘッダのような短い1行テキスト
    (画面分類用キーワード照合) に十分.
    """
    sub = crop(image, box)
    if sub.size == 0:
        return []
    if upscale_factor != 1.0:
        sub_for_ocr = upscale(sub, upscale_factor)
    else:
        sub_for_ocr = sub
    if sub_for_ocr.ndim == 3:
        gray = cv2.cvtColor(sub_for_ocr, cv2.COLOR_BGR2GRAY)
    else:
        gray = sub_for_ocr
    h, w = gray.shape[:2]
    engine = _engine()
    raw = engine.recognize(
        gray,
        horizontal_list=[[0, w, 0, h]],
        free_list=[],
        detail=1,
        batch_size=1,
        allowlist=allowlist,
    )
    out: list[OcrResult] = []
    for poly, text, conf in raw:
        pts: list[tuple[int, int]] = []
        if poly:
            for x, y in poly:
                ox = int(x / upscale_factor) + box.x
                oy = int(y / upscale_factor) + box.y
                pts.append((ox, oy))
        out.append(
            OcrResult(text=str(text), confidence=float(conf), box=tuple(pts))
        )
    return out


def recognize_in_multi_images(
    images: list[np.ndarray],
    boxes_per_image: list[list[Box]],
    *,
    allowlist: str | None = None,
) -> list[list[list[OcrResult]]]:
    """複数 image の行 box 群を vertical stack で連結し1回の認識で batch 処理する.

    GPU 推論の起動 overhead を分散するために read_rows_batched から呼ばれる.
    各 image 内の boxes は y 昇順前提 (recognize_in_regions と同じ). スタック後
    の全体 box 列も y 昇順なので、結果は index 対応で各 image・各 box に戻せる.

    返り値: out[image_index][box_index] = list[OcrResult]
    """
    if not images:
        return []
    if len(images) != len(boxes_per_image):
        raise ValueError("images と boxes_per_image の長さが一致しない")

    # 全 image を grayscale 化して縦に連結. それぞれの box に y_offset を足す.
    gray_list: list[np.ndarray] = []
    offsets: list[int] = []  # 各 image の y 開始位置
    cur_y = 0
    for img in images:
        if img.ndim == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img
        offsets.append(cur_y)
        gray_list.append(g)
        cur_y += g.shape[0]
    # 横幅は image ごとに違うことが想定されないが、念のため最大幅で揃える
    max_w = max(g.shape[1] for g in gray_list)
    padded = []
    for g in gray_list:
        if g.shape[1] < max_w:
            pad = np.zeros((g.shape[0], max_w - g.shape[1]), dtype=g.dtype)
            padded.append(np.concatenate([g, pad], axis=1))
        else:
            padded.append(g)
    stacked = np.concatenate(padded, axis=0)

    # box index → (image_index, local_box_index)
    flat_boxes: list[tuple[int, int, list[int]]] = []
    for i, (off, boxes) in enumerate(zip(offsets, boxes_per_image)):
        for j, b in enumerate(boxes):
            flat_boxes.append(
                (i, j, [b.x, b.x + b.w, b.y + off, b.y + b.h + off])
            )
    if not flat_boxes:
        return [[[] for _ in bs] for bs in boxes_per_image]

    horizontal_list = [hl for (_, _, hl) in flat_boxes]
    engine = _engine()
    raw = engine.recognize(
        stacked,
        horizontal_list=horizontal_list,
        free_list=[],
        detail=1,
        batch_size=len(flat_boxes),
        allowlist=allowlist,
    )

    out: list[list[list[OcrResult]]] = [
        [[] for _ in bs] for bs in boxes_per_image
    ]
    if not raw:
        return out
    if len(raw) == len(flat_boxes):
        # 全体 y 昇順前提で index 対応 (recognize_in_regions と同じロジック)
        for k, (poly, text, conf) in enumerate(raw):
            i, j, _ = flat_boxes[k]
            pts = (
                tuple((int(x), int(y)) for (x, y) in poly) if poly else ()
            )
            out[i][j].append(
                OcrResult(text=str(text), confidence=float(conf), box=pts)
            )
        return out
    # 件数不一致時のフォールバック: 重心 y で最近傍 box へ割り当て
    for poly, text, conf in raw:
        if not poly:
            continue
        pts_arr = np.asarray(poly, dtype=float)
        cy = float(pts_arr[:, 1].mean())
        best_k, best_d = -1, float("inf")
        for k, (_, _, hl) in enumerate(flat_boxes):
            box_cy = (hl[2] + hl[3]) / 2.0
            d = abs(cy - box_cy)
            if d < best_d:
                best_d, best_k = d, k
        if best_k >= 0:
            i, j, _ = flat_boxes[best_k]
            pts = tuple((int(x), int(y)) for (x, y) in poly)
            out[i][j].append(
                OcrResult(text=str(text), confidence=float(conf), box=pts)
            )
    return out


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


class Recognizer(Protocol):
    """複数 image の行 box 群をバッチで認識する抽象 OCR API.

    実装を差し替えることで、推論バックエンドを切り替えられる. 入出力は
    image-major の三重リストで、image[i] の box[j] に対する認識結果
    (list[OcrResult]) を返す. allowlist は実装にヒントとして渡され、
    実装ごとに尊重するか無視するかを決める.
    """

    def recognize_batch(
        self,
        images: list[np.ndarray],
        boxes_per_image: list[list[Box]],
        *,
        allowlist: str | None = None,
    ) -> list[list[list[OcrResult]]]:
        ...


class EasyOcrRecognizer:
    """汎用テキスト認識 (EasyOCR) ベースの Recognizer 実装.

    任意のテキストを認識して raw text + confidence を返す. 後段の fuzzy
    match で技名へ正規化する経路と組み合わせて使う. allowlist は
    EasyOCR.recognize の引数として渡され文字種制約として作用する.
    """

    def recognize_batch(
        self,
        images: list[np.ndarray],
        boxes_per_image: list[list[Box]],
        *,
        allowlist: str | None = None,
    ) -> list[list[list[OcrResult]]]:
        return recognize_in_multi_images(
            images, boxes_per_image, allowlist=allowlist
        )


class IdentityRecognizer:
    """別の Recognizer を wrap してそのまま渡す identity (no-op) wrapper.

    Recognizer 抽象化の動作確認・テスト用. recognize_batch は inner に
    そのまま委譲する. fallback chain や confidence-based gating を後付け
    する decorator のひな型としても流用できる.
    """

    def __init__(self, inner: Recognizer) -> None:
        self.inner = inner

    def recognize_batch(
        self,
        images: list[np.ndarray],
        boxes_per_image: list[list[Box]],
        *,
        allowlist: str | None = None,
    ) -> list[list[list[OcrResult]]]:
        return self.inner.recognize_batch(
            images, boxes_per_image, allowlist=allowlist
        )


_DEFAULT_RECOGNIZER: Recognizer | None = None


def get_default_recognizer() -> Recognizer:
    """既定の Recognizer を返す (シングルトン). PipelineConfig.recognizer
    で明示指定がない場合のフォールバック."""
    global _DEFAULT_RECOGNIZER
    if _DEFAULT_RECOGNIZER is None:
        _DEFAULT_RECOGNIZER = EasyOcrRecognizer()
    return _DEFAULT_RECOGNIZER
