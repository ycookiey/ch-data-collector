"""画面内の固定領域座標 (Switch 720p ベース).

実フレーム (test.mp4 = 1280x720) のOCR結果から決定した値.
解像度が異なる場合は resolve_layout() で換算する.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def scaled(self, scale_x: float, scale_y: float) -> "Box":
        return Box(
            x=int(self.x * scale_x),
            y=int(self.y * scale_y),
            w=max(1, int(self.w * scale_x)),
            h=max(1, int(self.h * scale_y)),
        )


@dataclass(frozen=True)
class Layout:
    width: int
    height: int

    # 画面分類用 (ヘッダ)
    #   詳細画面: 「能力ポイント」が ~(100,116) - (208,140)
    #   技一覧:   「教える技を選んでください」が ~(110,104) - (396,134)
    header: Box

    # ボックス画面: 右パネルの「種族名行」(色付きバーの大名前=ニックネームの直下に
    #   ある小さい左寄せテキストが種族名). ニックネーム付きでも常にここは種族名なので
    #   この行だけを読む (バーの大名前=ニックは別ポケ名のことがあり使わない).
    #   実測(480p): x≈545, y≈77, w≈210, h≈22 → 720p基準に換算.
    box_name: Box

    # 技一覧: 6行スロットの技名領域
    #   実測: 行1 (147,231)..(327,269), 行間66-67px, 6行
    move_slot_x: int
    move_slot_w: int
    move_slot_h: int
    move_slot_ys: tuple[int, ...]

    # 技一覧: タイプアイコン色サンプル位置 (技名の左の色付きアイコン).
    #   実測: アイコン色帯は x=86-94 が安定 (中央 x=98-110 は白シンボル,
    #   x<82 はカーソル選択行の黄緑光彩が混入). 行中央±6px をサンプルする.
    #   サンプル矩形 = (type_icon_x, move_slot_ys[i] + type_icon_dy)
    #                  サイズ (type_icon_w, type_icon_h)
    type_icon_x: int
    type_icon_w: int
    type_icon_dy: int
    type_icon_h: int

    def __post_init__(self) -> None:
        # 行ピッチ (move_slot_ys[1]-[0]) を detect_row_tops/read_rows が使うため
        # 2要素以上を要求する. 1要素以下のレイアウトは IndexError を招く.
        if len(self.move_slot_ys) < 2:
            raise ValueError(
                "move_slot_ys は2要素以上必要 (行ピッチ算出に使う)"
            )


LAYOUT_720P = Layout(
    width=1280,
    height=720,
    header=Box(x=80, y=95, w=400, h=50),
    box_name=Box(x=817, y=113, w=315, h=38),
    # スロットは実測 (x=147-330, y=231-270) ベース. 余白を残して認識器に
    # 入力を渡す方が短い技名 (ねごと等) の精度が安定する.
    move_slot_x=142,
    move_slot_w=228,
    move_slot_h=50,
    move_slot_ys=(220, 287, 354, 421, 488, 555),
    type_icon_x=86,
    type_icon_w=9,
    type_icon_dy=19,
    type_icon_h=12,
)


def resolve_layout(width: int, height: int) -> Layout:
    """フレーム解像度に応じてレイアウトを返す. 720p以外はスケール換算."""
    if (width, height) == (LAYOUT_720P.width, LAYOUT_720P.height):
        return LAYOUT_720P
    sx = width / LAYOUT_720P.width
    sy = height / LAYOUT_720P.height
    return Layout(
        width=width,
        height=height,
        header=LAYOUT_720P.header.scaled(sx, sy),
        box_name=LAYOUT_720P.box_name.scaled(sx, sy),
        move_slot_x=int(LAYOUT_720P.move_slot_x * sx),
        move_slot_w=max(1, int(LAYOUT_720P.move_slot_w * sx)),
        move_slot_h=max(1, int(LAYOUT_720P.move_slot_h * sy)),
        move_slot_ys=tuple(int(y * sy) for y in LAYOUT_720P.move_slot_ys),
        type_icon_x=int(LAYOUT_720P.type_icon_x * sx),
        type_icon_w=max(1, int(LAYOUT_720P.type_icon_w * sx)),
        type_icon_dy=int(LAYOUT_720P.type_icon_dy * sy),
        type_icon_h=max(1, int(LAYOUT_720P.type_icon_h * sy)),
    )
