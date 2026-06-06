"""技プール構築: 技一覧画面の各行を実位置で動的クロップ → OCR → fuzzy match.

連続スクロール (左右長押し) に対応するため固定スロット座標は使わず、各フレームで
行を実位置に合わせてクロップする:
  1. detect_row_tops でタイプアイコン列から各行の縦位置を粗検出する
  2. _text_band_center で各行の技名 (暗文字帯) の縦中心に箱を合わせる
  3. recognize_in_regions でまとめて認識し、低 confidence のスロットのみ
     検出器あり ocr_region で再走する (ハイブリッド)
  4. タイプアイコン色をタイブレーカに fuzzy_match_move で MOVE_DATA へ正規化する

一周検出やスロット集合の追跡は行わない (技名の収集・確定は pipeline 側の投票が
担う)。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np

from ch_data_collector.master_data import MasterData, Move
from ch_data_collector.ocr import OcrResult, ocr_region, recognize_in_regions
from ch_data_collector.screen_layouts import Box, Layout
from ch_data_collector.type_icon import observed_slot_type


@dataclass(frozen=True)
class MoveCandidate:
    move: Move
    score: float


# タイプタイブレーカの加点幅. 名前類似度 (SequenceMatcher ratio, 0..1) にこの値を
# 加えて順位付けする. 0.15 は OCR 誤読で生じる僅差 (例 0.75 vs 0.85) は同タイプ側へ
# 倒しつつ、名前が明確に優る候補 (score 差 > 0.15) は観測タイプが誤っても守る妥協値.
_TYPE_MATCH_BONUS = 0.15


def fuzzy_match_move(
    ocr_text: str,
    master: MasterData,
    top_k: int = 3,
    observed_type: str | None = None,
) -> list[MoveCandidate]:
    if not ocr_text:
        return []
    # 文字種正規化を OCR/master 両側に適用してから類似度計算.
    # 「ねっぷう / ねつぷう」「フェイント / フエイント」等の読みブレを吸収.
    normalized_ocr = _kana_normalize(ocr_text)
    cands: list[MoveCandidate] = []
    for m in master.moves:
        normalized_master = _kana_normalize(m.name)
        score = SequenceMatcher(
            None, normalized_ocr, normalized_master
        ).ratio()
        cands.append(MoveCandidate(move=m, score=score))
    if observed_type:
        # タイプタイブレーカ (soft): 観測タイプ一致候補に名前類似度へ加点する.
        # 誤読が別タイプの実在技に着地する事故 (ぼうふう→ほうふく等) を弾くのが狙い.
        # ハード分割 (同タイプを score 不問で常に上位) にすると、アイコン色の誤判定で
        # observed_type が間違ったとき、正しく読めた高スコアの別タイプ技を低スコアの
        # 同タイプ技で上書きしてしまう. 加点方式なら名前類似が僅差のときだけタイプで
        # 決まり、名前が明らかに優る候補は守られる (加点は順位付けのみで、accept 判定に
        # 使う MoveCandidate.score は据え置き).
        cands.sort(
            key=lambda c: c.score
            + (_TYPE_MATCH_BONUS if c.move.type == observed_type else 0.0),
            reverse=True,
        )
    else:
        cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:top_k]


def _best_text(results: list[OcrResult]) -> str:
    """1スロット内の最高 confidence のテキストを返す (空なら空文字)."""
    if not results:
        return ""
    return max(results, key=lambda r: r.confidence).text.strip()


def normalize_slot_text(
    text: str,
    master: MasterData,
    *,
    accept_threshold: float = 0.7,
    observed_type: str | None = None,
) -> str:
    """OCR text を MOVE_DATA に fuzzy match し、確信があれば正規化名を返す.

    確信なしなら raw を返す (ループ検出時の一意性確保のため空にしない).
    observed_type 指定時はタイプタイブレーカで候補を絞る.
    """
    if not text:
        return ""
    cands = fuzzy_match_move(
        text, master, top_k=1, observed_type=observed_type
    )
    if cands and cands[0].score >= accept_threshold:
        return cands[0].move.name
    return text


# recognize の結果が信頼できると見なす最低 confidence.
# これを下回るスロットは検出器あり ocr_region で再走する (ハイブリッド).
_RECOGNIZE_TRUST_CONFIDENCE = 0.5


def _build_allowlist() -> str:
    """Champions 技名に出現する文字種を allowlist として固定する.

    ひらがな全種 + カタカナ全種 + 小書き + 長音符 + ヴ + 半角/全角数字 + 括弧 +
    master に実在するラテン大文字 (DDラリアット / Gのちから / Vジェネレート).
    漢字と小文字英字は除外して認識器がそれらへ誤マッチするのを構造的に防ぐ.

    NOTE: master 出現文字だけに絞るとなぜか「ねごと」等の精度が落ちる
    再現がある (おそらく認識器が許可文字内で誤読を作る経路が変わるため).
    実用上はひら/カタ全種を許す方が安定する.
    全角数字は「１０００まんボルト」が母集合に在るため必須 (欠くと認識器が出力
    できず別技へ誤吸収される). NFKC 畳み込み (_kana_normalize) と併せて半角/全角
    どちらの読みでも master に一致させる.
    """
    hiragana = "".join(chr(c) for c in range(0x3041, 0x3097))
    katakana = "".join(chr(c) for c in range(0x30A1, 0x30FB))
    fullwidth_digits = "".join(chr(c) for c in range(0xFF10, 0xFF1A))
    latin_upper = "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
    return (
        hiragana + katakana + latin_upper
        + "ー・0123456789" + fullwidth_digits + "()（）"
    )


_TECHNIQUE_ALLOWLIST = _build_allowlist()


# 読みブレ吸収用の文字種正規化テーブル.
# 小書き → 大書き, 濁点/半濁点 → 清音, 一部の同形混同を統一. fuzzy match 前に
# OCR/master 両側に適用することで「ね"っ"ぷう/ね"つ"ぷう」「フェ/フエ」や
# 濁点誤読「ねごと→ねこと」(ご→こ) 等の差を消す.
# NOTE: 濁点/半濁点の畳み込みは master 960技で衝突ゼロを確認済み (一意性を
# 損なわない). 濁点だけ違う別技が無いため、誤読耐性を上げても誤マッチを増やさない.
_NORMALIZE_MAP: dict[str, str] = {
    "ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ",
    "ッ": "ツ",
    "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ",
    "ゃ": "や", "ゅ": "ゆ", "ょ": "よ",
    "っ": "つ",
    "ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お",
}


def _add_dakuten_folding(table: dict[str, str]) -> None:
    """濁点・半濁点付き仮名を清音へ畳む対応を table に追加する (ひら/カタ両方)."""
    # 各ペアは (濁/半濁音, 清音) の並び.
    hira = "がかぎきぐくげけごこざさじしずすぜせぞそだたぢちづつでてどと" \
           "ばはびひぶふべへぼほぱはぴひぷふぺへぽほ"
    kata = "ガカギキグクゲケゴコザサジシズスゼセゾソダタヂチヅツデテドト" \
           "バハビヒブフベヘボホパハピヒプフペヘポホヴウ"
    for seq in (hira, kata):
        for i in range(0, len(seq), 2):
            table.setdefault(seq[i], seq[i + 1])


_add_dakuten_folding(_NORMALIZE_MAP)
_NORMALIZE_TABLE = str.maketrans(_NORMALIZE_MAP)


@lru_cache(maxsize=8192)
def _kana_normalize(s: str) -> str:
    # master 技名/ポケモン名は毎フレーム・毎行の fuzzy match で繰り返し正規化される.
    # 入力文字列ごとにメモ化し、固定の master 側は実質1回だけ正規化する.
    # NFKC で全角/半角の差 (全角数字・全角英字・全角括弧) を畳んでから小書き/濁点を
    # 畳む. これで「１０００まんボルト」が OCR の半角読みと一致する (master 955技で
    # NFKC 追加による新規衝突ゼロを確認済み).
    return unicodedata.normalize("NFKC", s).translate(_NORMALIZE_TABLE)


def _resolve_slot_text(bucket: list[OcrResult]) -> tuple[str, float]:
    if not bucket:
        return "", 0.0
    best = max(bucket, key=lambda r: r.confidence)
    return best.text.strip(), float(best.confidence)


def detect_row_tops(image: np.ndarray, layout: Layout) -> list[tuple[int, int]]:
    """各行のタイプアイコン中心を縦に検出し、(text_top, icon_top) のリストを返す.

    タイプアイコン (技名の左の色付き角丸ブロック) は各行に1個あり、その縦中心は
    技名テキストの縦中心に一致する。アイコンの縦位置に箱を合わせれば、連続
    スクロールで行が整列していなくても技名を見切らずにクロップできる
    (明度バンドの中心はテキスト位置とズレ見切れる問題を回避)。

    アイコン列を縦走査し「アイコンらしさ (彩度が高い=色付き or 明るく低彩度=灰)」の
    プロファイルに spacing 周期の櫛を当てて位相を決め、各アイコン中心を得る。
    返す各要素 = (text_top, icon_top)。icon_top = アイコン中心 - move_slot_h//2
    (move_slot_ys と同基準, 未クランプ。タイプ色サンプル用)。text_top は icon_top を
    リスト領域上端へクランプした値 (最上段の技名見切れ防止用のテキストクロップ基準)。
    画面内に行高ぶん完全に収まる行のみ返す。
    """
    spacing = layout.move_slot_ys[1] - layout.move_slot_ys[0]
    row_h = layout.move_slot_h
    vtop = max(0, layout.move_slot_ys[0] - spacing)
    vbot = min(image.shape[0], layout.move_slot_ys[-1] + row_h + spacing)
    # アイコン列 (左右に少し余裕を持たせる)
    ix0 = max(0, layout.type_icon_x - 2)
    ix1 = layout.type_icon_x + layout.type_icon_w + 2
    col = image[vtop:vbot, ix0:ix1]
    if col.size == 0:
        return []
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean(axis=1)
    val = hsv[:, :, 2].mean(axis=1)
    # アイコンらしさ: 色付き (彩度高) または 灰アイコン (明るく低彩度).
    iconness = ((sat > 50) | ((val > 120) & (sat < 45))).astype(float)
    n = len(iconness)
    best_off, best_score = 0, -1.0
    # 位相スコアは固定分母 (最多歯数) で正規化する. len(vals) で割ると走査窓の端で
    # 歯が1本減る位相が 7/7=1.0 のように過大評価され、真の位相 (7/8) を逆転して
    # 全行が縦にズレることがあるため.
    max_teeth = len(range(0, n, spacing))
    for off in range(spacing):
        vals = [iconness[i] for i in range(off, n, spacing)]
        score = sum(vals) / max_teeth if max_teeth else 0.0
        if score > best_score:
            best_score, best_off = score, off
    tops: list[tuple[int, int]] = []
    # アイコン中心が技行範囲内 (先頭行〜末尾行) にあるもののみ.
    center_min = layout.move_slot_ys[0] + row_h // 2 - spacing // 2
    center_max = layout.move_slot_ys[-1] + row_h // 2 + spacing // 2
    top_limit = layout.move_slot_ys[0]  # 技リスト領域の上端 (これより上はヘッダ)
    for i in range(best_off, n, spacing):
        if iconness[i] < 0.5:
            continue  # アイコンが無い位置 (行間/ヘッダ/フッタ) は除外
        yc = i + vtop  # アイコン縦中心
        if yc < center_min or yc > center_max:
            continue
        icon_top = yc - row_h // 2  # 行上端 (move_slot_ys 同基準, 未クランプ)
        # 最上段はヘッダ近傍で comb が上に引っ張られ ~10px 高く出ることがある.
        # 行はリスト領域上端より上には存在しないので、そこへクランプして最上段技
        # (例: 一番上の ギガインパクト) のテキスト見切れを防ぐ. クランプはテキスト
        # クロップ基準 (text_top) のみに適用し、タイプ色サンプルは実アイコン位置
        # (icon_top) を使う — クランプ値を色に流用すると最上段で patch がアイコン下の
        # 背景へずれ、型タイブレーカが誤る/無効化するため.
        text_top = icon_top
        if text_top < top_limit:
            text_top = top_limit
        if text_top >= vtop and text_top + row_h <= vbot:
            tops.append((text_top, icon_top))
    return tops


def _text_band_center(
    image: np.ndarray, layout: Layout, yt: int, spacing: int
) -> int | None:
    """行ピッチ窓 [yt, yt+spacing] 内の技名 (暗文字) 帯の縦中心を返す.

    技名は明るい行矩形上の暗い文字。各yの暗ピクセル率プロファイルで、しきい値を
    超える連続帯の中心 = テキストの縦中心。見つからなければ None (空スロット)。
    """
    x0 = layout.move_slot_x
    x1 = x0 + layout.move_slot_w
    y0 = max(0, yt)
    y1 = min(image.shape[0], yt + spacing)
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return None
    gray = (
        cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        if region.ndim == 3
        else region
    )
    # 各yが文字行か. 短い技名 (ねごと等) は幅広い箱に対し暗ピクセル率が低いため
    # しきい値は低めにする (高いと短名の帯が検出されず行ごと脱落する).
    dark = (gray < 120).mean(axis=1) > 0.02
    # 最長の連続帯 (= 主たる技名) の中心を採る. idx[0]〜idx[-1] 全span だと、
    # 粗い行検出のズレで窓に入った隣接行の文字断片を巻き込み中心が偏るため.
    best_len, best_s, best_e = 0, 0, 0
    s: int | None = None
    arr = list(dark) + [False]
    for i, v in enumerate(arr):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s > best_len:
                best_len, best_s, best_e = i - s, s, i - 1
            s = None
    if best_len == 0:
        return None
    return (best_s + best_e) // 2 + y0


def read_rows(
    image: np.ndarray,
    layout: Layout,
    master: MasterData | None = None,
    *,
    accept_threshold: float = 0.7,
) -> list[str]:
    """表示中の各行を実位置で読み取り、正規化済み技名のリストを返す.

    固定スロット位置でなく detect_row_tops で検出した実位置を使うため、
    連続スクロール中の中途半端なフレームでも正しく読める。空読みは除外。
    """
    tops = detect_row_tops(image, layout)
    if not tops:
        return []
    spacing = layout.move_slot_ys[1] - layout.move_slot_ys[0]
    # 各行を技名テキストの暗文字帯の縦中心に合わせて箱を作る. アイコン/行矩形の
    # 位置オフセットに依存せず、テキストそのものに整列するので見切れない
    # (固定余白などのアドホックな調整が不要). アイコンcombの粗い行上端 yt の
    # ピッチ窓から暗文字帯中心を求め、そこへ move_slot_h の箱を中心配置する。
    # テキストが見つからない行 (空スロット) は除外。
    name_boxes: list[Box] = []
    icon_tops: list[int] = []  # タイプ色サンプル基準 (未クランプの実アイコン行上端)
    for text_top, icon_top in tops:
        tctr = _text_band_center(image, layout, text_top, spacing)
        if tctr is None:
            continue
        by = max(0, tctr - layout.move_slot_h // 2)
        name_boxes.append(
            Box(layout.move_slot_x, by, layout.move_slot_w, layout.move_slot_h)
        )
        icon_tops.append(icon_top)
    if not name_boxes:
        return []
    bucket_results = recognize_in_regions(
        image, name_boxes, allowlist=_TECHNIQUE_ALLOWLIST
    )
    out: list[str] = []
    for i in range(len(name_boxes)):
        text, conf = _resolve_slot_text(bucket_results[i])
        if conf < _RECOGNIZE_TRUST_CONFIDENCE:
            fallback = ocr_region(image, name_boxes[i], upscale_factor=2.0)
            fb_text = _best_text(fallback)
            if fb_text:
                text = fb_text
        if master is not None and text:
            # タイプアイコン色は実アイコン位置由来の行上端 (icon_top, 未クランプ) を
            # 基準にサンプルする. type_icon_dy は move_slot_ys[i] (固定スロット上端)
            # からの較正値で、icon_top も「move_slot_ys 同基準」のアイコン中心由来。
            # 暗文字帯中心由来の箱上端 by を使うとスクロール中途で帯中心とアイコン
            # 中心がずれた行でサンプルがアイコン外に落ち、誤タイプ判定→誤マッチ着地を
            # 招く。最上段のテキストクロップ用クランプ (text_top) も色には使わない。
            obs_type = observed_slot_type(image, layout, icon_tops[i])
            text = normalize_slot_text(
                text,
                master,
                accept_threshold=accept_threshold,
                observed_type=obs_type,
            )
        if text:
            out.append(text)
    return out


def resolve_moves(
    raw_names: Iterable[str],
    master: MasterData,
    *,
    accept_threshold: float = 0.7,
) -> tuple[list[Move], list[tuple[str, list[MoveCandidate]]]]:
    """OCR raw text群を MOVE_DATA へ fuzzy match.

    Returns:
      (確定した技リスト, 曖昧なもの: [(raw_text, top候補...), ...])
    """
    accepted: list[Move] = []
    accepted_ids: set[int] = set()
    ambiguous: list[tuple[str, list[MoveCandidate]]] = []
    for raw in raw_names:
        cands = fuzzy_match_move(raw, master, top_k=3)
        if not cands:
            continue
        top = cands[0]
        if top.score >= accept_threshold:
            if top.move.id not in accepted_ids:
                accepted.append(top.move)
                accepted_ids.add(top.move.id)
        else:
            ambiguous.append((raw, cands))
    return accepted, ambiguous
