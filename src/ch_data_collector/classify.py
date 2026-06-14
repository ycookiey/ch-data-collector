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

from ch_data_collector.ocr import crop, joined_text, recognize_region
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
    # 検出器をスキップし認識器のみでヘッダを読む (ocr_region 比 ~30-50% 速い).
    # ヘッダは「教える技」「能力」等の短い1行テキストなので検出器不要.
    results = recognize_region(image, layout.header, upscale_factor=2.0)
    text = joined_text(results)
    if any(kw in text for kw in MOVE_LIST_KEYWORDS):
        return ScreenKind.MOVE_LIST
    if any(kw in text for kw in DETAIL_KEYWORDS):
        return ScreenKind.DETAIL
    return ScreenKind.OTHER


_HEADER_THUMB_SIZE = (40, 12)
# ヘッダ thumb 同士の MSE しきい値. 動的要素 (HP gauge 等) の揺れを許容しつつ
# 異 kind の混同を防ぐ. classify は最小 MSE が この値未満なら hit と判定し、
# remember は既存テンプレの最小 MSE が この値以上なら新規追加する.
_HEADER_MSE_THRESHOLD = 5.0
# kind ごとに保持するテンプレ thumb の最大数. DETAIL の HP gauge 動的要素や
# OTHER 多様性 (box / menu / transition) をカバーする上限.
_TEMPLATE_LIMIT_PER_KIND = 32


@dataclass
class TemplateClassifier:
    """ヘッダ thumb (40x12) の MSE で画面分類するマルチテンプレ DB.

    各 kind ごとに最大 _TEMPLATE_LIMIT_PER_KIND 個の thumb を保持し、
    新フレームの thumb と全テンプレの MSE を計算して最小値の kind を返す.

    旧実装 (kind あたり 1 テンプレ + matchTemplate) は DETAIL ヘッダ内の動的
    要素 (HP gauge / 状態異常アイコン) でしきい値を切らず大量に OCR フォール
    バックしていた (測定で classify_screen 113/232 = 49% が DETAIL での
    フォールバック). マルチテンプレに変えて MSE 比較に切り替えることで
    OTHER のバリエーション (box / menu / transition / black) も同じ枠で扱える.

    比較は 40x12 = 480 px の MSE で 1 ペア あたり <0.01ms. 上限 32 でも
    classify は <1ms/frame.
    """

    mse_threshold: float = _HEADER_MSE_THRESHOLD
    template_limit: int = _TEMPLATE_LIMIT_PER_KIND
    thumbs: dict[ScreenKind, list[np.ndarray]] = field(default_factory=dict)

    def is_ready(self) -> bool:
        # OTHER は遅延学習で良い (DETAIL/MOVE_LIST が両方揃ったら有効化、
        # OTHER は最初の数フレームを OCR で判定して remember で溜める).
        return (
            ScreenKind.DETAIL in self.thumbs
            and ScreenKind.MOVE_LIST in self.thumbs
        )

    def _header_thumb(
        self, image: np.ndarray, layout: Layout
    ) -> np.ndarray | None:
        b = layout.header
        sub = image[b.y : b.y + b.h, b.x : b.x + b.w]
        if sub.size == 0:
            return None
        return cv2.resize(sub, _HEADER_THUMB_SIZE, interpolation=cv2.INTER_AREA)

    def remember(
        self, kind: ScreenKind, image: np.ndarray, layout: Layout
    ) -> None:
        thumb = self._header_thumb(image, layout)
        if thumb is None:
            return
        existing = self.thumbs.setdefault(kind, [])
        if len(existing) >= self.template_limit:
            return
        # 既存テンプレと十分近ければ重複なので追加しない
        thumb_i16 = thumb.astype(np.int16)
        for ref in existing:
            diff = float(np.abs(thumb_i16 - ref.astype(np.int16)).mean())
            if diff < self.mse_threshold:
                return
        existing.append(thumb)

    def classify(self, image: np.ndarray, layout: Layout) -> ScreenKind | None:
        """テンプレに十分一致する kind を返す. 確信が無ければ None.

        None は「テンプレで判定不能」を意味し、呼び出し側で OCR 分類へ
        フォールバックする (新しいヘッダパターンの初出を学習する経路).
        """
        thumb = self._header_thumb(image, layout)
        if thumb is None:
            return None
        thumb_i16 = thumb.astype(np.int16)
        best_kind: ScreenKind | None = None
        best_diff = self.mse_threshold
        for kind, refs in self.thumbs.items():
            for ref in refs:
                diff = float(
                    np.abs(thumb_i16 - ref.astype(np.int16)).mean()
                )
                if diff < best_diff:
                    best_diff = diff
                    best_kind = kind
        return best_kind
