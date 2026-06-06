"""画面分類: 詳細画面 / 技一覧 / その他.

ヘッダ領域 (画面上部固定座標) を OCR してテキスト内容で判定する.
高速化のため、各 kind を OCR で初めて検出した瞬間にヘッダ画像を
テンプレ画像として記憶し、以降は cv2.matchTemplate で OCR を回避する.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from ch_data_collector.ocr import crop, joined_text, ocr_region
from ch_data_collector.screen_layouts import Layout


class ScreenKind(Enum):
    DETAIL = "detail"
    MOVE_LIST = "move_list"
    OTHER = "other"


# ヘッダで現れる代表的なキーワード (誤読バリエーション込み)
DETAIL_KEYWORDS = ("能力", "ポイント", "ボイント", "ポイン", "ボイン")
MOVE_LIST_KEYWORDS = (
    "教える技",
    "教えるわざ",
    "選んでくださ",
    "技を",
    "わざを",
)


def classify_screen(image: np.ndarray, layout: Layout) -> ScreenKind:
    results = ocr_region(image, layout.header, upscale_factor=2.0)
    text = joined_text(results)
    if any(kw in text for kw in MOVE_LIST_KEYWORDS):
        return ScreenKind.MOVE_LIST
    if any(kw in text for kw in DETAIL_KEYWORDS):
        return ScreenKind.DETAIL
    return ScreenKind.OTHER


@dataclass
class TemplateClassifier:
    """ヘッダ領域のテンプレマッチで画面分類を高速化する.

    DETAIL / MOVE_LIST の両方が OCR ベースで一度でも分類されると、
    その時のヘッダ画像をテンプレとして記憶する. 以降は cv2.matchTemplate
    で類似度を測り、OCR を呼ばず分類する (数十倍高速).
    """

    score_threshold: float = 0.85
    templates: dict[ScreenKind, np.ndarray] = field(default_factory=dict)

    def is_ready(self) -> bool:
        return (
            ScreenKind.DETAIL in self.templates
            and ScreenKind.MOVE_LIST in self.templates
        )

    def remember(
        self, kind: ScreenKind, image: np.ndarray, layout: Layout
    ) -> None:
        if kind == ScreenKind.OTHER:
            return
        if kind in self.templates:
            return
        # ヘッダ領域をクロップしてテンプレ化
        self.templates[kind] = crop(image, layout.header).copy()

    def classify(self, image: np.ndarray, layout: Layout) -> ScreenKind | None:
        """テンプレに十分一致する kind を返す. 確信が無ければ None.

        None は「テンプレで判定不能」を意味し、呼び出し側で OCR 分類へ
        フォールバックする。これにより (1) 最初に記憶した非典型ヘッダが
        低品質テンプレ化しても全フレームが誤って OTHER 固定されず OCR が
        再判定でき、(2) しきい値ちょうど/僅差のフレームも OTHER と断定せず
        OCR に委ねられる。
        """
        header = crop(image, layout.header)
        best_kind: ScreenKind | None = None
        best_score = self.score_threshold
        for kind, tpl in self.templates.items():
            if tpl.shape != header.shape:
                continue
            res = cv2.matchTemplate(header, tpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_kind = kind
        return best_kind
