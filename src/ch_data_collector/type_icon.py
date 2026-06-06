"""技一覧スロットのタイプアイコン色から技タイプを判定する.

技名 OCR の fuzzy match が、誤読で別の実在技に着地するケース
(例: 「ぼうふう」→OCR「ほうふく」→実在技 報復にマッチ) を、
タイプという独立した手がかりで弾くためのタイブレーカ用モジュール.

判定方式: アイコンの固定位置 (screen_layouts の type_icon_*) の中央値色を
参照色テーブルに最近傍マッチする. 彩度マスク方式は紫背景やカーソル光彩を
誤って拾うため使わない.

参照色:
  REF_BGR は全18タイプを実機フレームから実測した値 (収集元: ファイアロー /
  ガブリアス / ヤミラミ / リザードン / ゲンガー の各動画。各タイプの技を高信頼で
  OCR 一致した行のアイコン色中央値)。全18タイプをタイブレーカに使う。
"""

from __future__ import annotations

import numpy as np

from ch_data_collector.screen_layouts import Layout

# 参照色 (B, G, R). 全18タイプ実機フレームからの実測値 (各タイプの技を高信頼で
# OCR一致した行のアイコン色中央値, n=サンプル数).
# 収集元: ファイアロー / ガブリアス / ヤミラミ / リザードン / ゲンガー の各動画.
REF_BGR: dict[str, tuple[int, int, int]] = {
    "あく": (64, 60, 79),        # n=551
    "ノーマル": (161, 160, 161),  # n=541
    "ほのお": (42, 19, 215),      # n=306
    "ひこう": (238, 187, 134),    # n=216
    "ゴースト": (112, 54, 109),   # n=132
    "はがね": (185, 164, 102),    # n=131
    "ドラゴン": (221, 88, 85),    # n=99
    "かくとう": (9, 123, 241),    # n=89
    "エスパー": (119, 41, 224),   # n=76
    "じめん": (38, 75, 138),      # n=67
    "フェアリー": (238, 88, 232),  # n=50
    "くさ": (45, 178, 71),        # n=33
    "いわ": (132, 169, 176),      # n=33
    "むし": (34, 172, 147),       # n=28
    "みず": (235, 128, 48),       # n=15
    "でんき": (10, 197, 244),     # n=15
    "どく": (199, 45, 138),       # n=10
    "こおり": (251, 228, 79),     # n=9
}

# 最近傍とみなす BGR ユークリッド距離の2乗の上限.
# 実測一致は数〜十数で収まる. JPEG ノイズを見込んで余裕を持たせる.
_MAX_DIST2 = 2500  # = 50^2


def _sample_color(image: np.ndarray, layout: Layout, row_y: int) -> np.ndarray | None:
    x0 = layout.type_icon_x
    x1 = x0 + layout.type_icon_w
    y0 = row_y + layout.type_icon_dy
    y1 = y0 + layout.type_icon_h
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return np.median(patch.reshape(-1, 3), axis=0)


def observed_slot_type(
    image: np.ndarray, layout: Layout, row_y: int
) -> str | None:
    """スロットのタイプアイコン色から技タイプを返す.

    最近傍の参照色が距離しきい値 (_MAX_DIST2) 内ならそのタイプ名を返す.
    距離が大きい (アイコン外をサンプルした等) 場合は None を返し、
    タイブレーカ不使用 (name-only) となる.
    """
    color = _sample_color(image, layout, row_y)
    if color is None:
        return None
    best: str | None = None
    best_d2 = float("inf")
    for t, (b, g, r) in REF_BGR.items():
        d2 = (color[0] - b) ** 2 + (color[1] - g) ** 2 + (color[2] - r) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = t
    if best is not None and best_d2 <= _MAX_DIST2:
        return best
    return None
