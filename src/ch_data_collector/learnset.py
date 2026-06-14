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
import weakref
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

import cv2
import numpy as np
from rapidfuzz import process as _rf_process
from rapidfuzz.distance import Levenshtein as _Lev

from ch_data_collector.master_data import MasterData, Move
from ch_data_collector.ocr import (
    OcrResult,
    Recognizer,
    crop,
    get_default_recognizer,
    ocr_region,
    recognize_in_multi_images,
    recognize_in_regions,
)
from ch_data_collector.screen_layouts import Box, Layout
from ch_data_collector.type_icon import observed_slot_type


@dataclass(frozen=True)
class MoveCandidate:
    move: Move
    score: float


# タイプタイブレーカの加点幅. 名前類似度 (Levenshtein normalized similarity, 0..1) にこの値を
# 加えて順位付けする. 0.15 は OCR 誤読で生じる僅差 (例 0.75 vs 0.85) は同タイプ側へ
# 倒しつつ、名前が明確に優る候補 (score 差 > 0.15) は観測タイプが誤っても守る妥協値.
_TYPE_MATCH_BONUS = 0.15


@dataclass
class _MoveIndex:
    """master 1 つ分の技名照合インデックス (正規化済み名 + 結果メモ).

    同じ OCR テキストの fuzzy match はフレームを跨いで何度も繰り返されるため、
    (text, top_k, observed_type) で結果をメモ化する。ユニーク入力は 1 動画あたり
    高々数百件なのでサイズ上限は設けない。
    """

    moves: tuple[Move, ...]
    normalized: tuple[str, ...]
    cache: dict[
        tuple[str, int, str | None], tuple[MoveCandidate, ...]
    ] = field(default_factory=dict)


_move_indexes: weakref.WeakKeyDictionary[MasterData, _MoveIndex] = (
    weakref.WeakKeyDictionary()
)


def _move_index(master: MasterData) -> _MoveIndex:
    idx = _move_indexes.get(master)
    if idx is None:
        idx = _MoveIndex(
            moves=master.moves,
            normalized=tuple(_kana_normalize(m.name) for m in master.moves),
        )
        _move_indexes[master] = idx
    return idx


def fuzzy_match_move(
    ocr_text: str,
    master: MasterData,
    top_k: int = 3,
    observed_type: str | None = None,
) -> list[MoveCandidate]:
    """OCR テキストを全技マスタへ fuzzy match し top_k 候補を返す.

    スコアは文字種正規化 (_kana_normalize) 後の Levenshtein 正規化類似度.
    位置依存の編集距離なので、連続部分文字列を共有しても位置がずれた候補 (例:
    'なゆきり' vs 'こなゆき'=共通"なゆき"だが全体としては別物) を低く評価する.
    LCS 系 (SequenceMatcher/Indel) は連続部分文字列重視で位置ずれに鈍感で、
    OCR 1 文字ブレが別技に誤マッチする事故 (なゆきり→こなゆき等) を起こしていた.
    rapidfuzz の cdist で全候補を一括計算する (C++ 実装で素朴ループより十分速い).
    """
    if not ocr_text:
        return []
    idx = _move_index(master)
    key = (ocr_text, top_k, observed_type)
    cached = idx.cache.get(key)
    if cached is not None:
        return list(cached)
    normalized_ocr = _kana_normalize(ocr_text)
    scores = _rf_process.cdist(
        [normalized_ocr],
        idx.normalized,
        scorer=_Lev.normalized_similarity,
        dtype=np.float64,
    )[0]
    if observed_type:
        # タイプタイブレーカ (soft): 観測タイプ一致候補に名前類似度へ加点する.
        # 誤読が別タイプの実在技に着地する事故 (ぼうふう→ほうふく等) を弾くのが狙い.
        # ハード分割 (同タイプを score 不問で常に上位) にすると、アイコン色の誤判定で
        # observed_type が間違ったとき、正しく読めた高スコアの別タイプ技を低スコアの
        # 同タイプ技で上書きしてしまう. 加点方式なら名前類似が僅差のときだけタイプで
        # 決まり、名前が明らかに優る候補は守られる (加点は順位付けのみで、accept 判定に
        # 使う MoveCandidate.score は据え置き).
        bonus = np.fromiter(
            (
                _TYPE_MATCH_BONUS if m.type == observed_type else 0.0
                for m in idx.moves
            ),
            dtype=np.float64,
            count=len(idx.moves),
        )
    else:
        bonus = np.zeros(len(idx.moves))
    sort_keys = scores + bonus
    # 上位 top_k を安定ソート (同点は master 順を保持) で抽出.
    # np.argsort は score 昇順なので符号反転し、stable で同点 master 順.
    order = np.argsort(-sort_keys, kind="stable")[:top_k]
    result = tuple(
        MoveCandidate(move=idx.moves[int(i)], score=float(scores[int(i)]))
        for i in order
    )
    idx.cache[key] = result
    return list(result)


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


# 行クロップキャッシュのサムネ寸法 (w, h) とセル分割数 (cols, rows).
# 一致判定は「セル平均絶対差の最大値」(cellmax)。全体平均だと
# 「きあいだめ/きあいだま」のような 1 グリフ局所差 (め/ま) がノイズに埋もれて
# 別技を誤一致させる (実測 min 1.63 で平均差閾値を下回り技を取り逃した)。
# セル単位の最大値なら局所差が薄まらず、実測で別技ペア最小 39.5 vs 閾値 12.0
# (3.3 倍マージン)。同一技のノイズ・±1px 整列揺れは全セルに薄く分散するため
# 閾値内に収まる。一致しない場合は再 OCR されるだけで精度影響はない。
_ROW_THUMB_SIZE = (96, 24)
_ROW_CACHE_GRID = (12, 3)
_ROW_CACHE_CELL_DIFF_THRESHOLD = 12.0


@dataclass
class RowTextCache:
    """整列済み行クロップ → OCR 生テキストのキャッシュ (1セグメント分).

    行クロップは技名テキストの暗文字帯中心に整列するため、同じ技の行は
    フレームを跨いでほぼ同一画像になる。縮小サムネのセル平均差最大値 (cellmax)
    が閾値未満の既知クロップは再認識せず前回のテキストを使う。連続スクロール中
    はスロット領域全体が毎フレーム変化して pipeline 側のフレーム差分スキップが
    効かないが、行単位では同一内容が数十フレーム続くため、ここで認識回数を削る。

    キャッシュするのは OCR 生テキストのみ。タイプ色サンプルと fuzzy match は
    フレーム毎に行う (fuzzy match は別途メモ化されるので実コストはない)。

    照合は全エントリとの差分を numpy 一括計算で行う (Python ループの線形探索は
    エントリが数百件に育つと OCR の節約分を食い潰す)。
    """

    texts: list[str] = field(default_factory=list)
    # スクロール中の整列揺れで同一技にも複数エントリが育つ. 異常な動画でも
    # メモリと照合コストが暴れない上限 (通常 1 セグメントの技は数十種).
    max_entries: int = 1024
    _thumbs: np.ndarray | None = None  # (capacity, h, w) int16
    _count: int = 0

    def lookup(self, thumb: np.ndarray) -> str | None:
        if self._count == 0:
            return None
        diffs = np.abs(self._thumbs[: self._count] - thumb)
        cols, rows = _ROW_CACHE_GRID
        n, h, w = diffs.shape
        cellmax = (
            diffs.reshape(n, rows, h // rows, cols, w // cols)
            .mean(axis=(2, 4))
            .max(axis=(1, 2))
        )
        best = int(np.argmin(cellmax))
        if float(cellmax[best]) < _ROW_CACHE_CELL_DIFF_THRESHOLD:
            return self.texts[best]
        return None

    def store(self, thumb: np.ndarray, text: str) -> None:
        if self._count >= self.max_entries:
            return
        if self._thumbs is None or self._count == len(self._thumbs):
            cap = 64 if self._thumbs is None else len(self._thumbs) * 2
            grown = np.zeros((cap, *thumb.shape), dtype=np.int16)
            if self._thumbs is not None:
                grown[: self._count] = self._thumbs[: self._count]
            self._thumbs = grown
        self._thumbs[self._count] = thumb
        self.texts.append(text)
        self._count += 1


def _row_thumb(image: np.ndarray, box: Box) -> np.ndarray:
    sub = crop(image, box)
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
    return cv2.resize(
        gray, _ROW_THUMB_SIZE, interpolation=cv2.INTER_AREA
    ).astype(np.int16)


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
    ocr_cache: RowTextCache | None = None,
) -> list[str]:
    """表示中の各行を実位置で読み取り、正規化済み技名のリストを返す.

    固定スロット位置でなく detect_row_tops で検出した実位置を使うため、
    連続スクロール中の中途半端なフレームでも正しく読める。空読みは除外。
    ocr_cache を渡すと、過去に読んだ行とほぼ同一のクロップは再認識しない。
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
    # 正規最下段スロットの bottom を超える行は除外 (画面下端の UI/背景グラデが
    # クロップ下端に混入し、fuzzy match が別技に誤着地する事象を物理的に防ぐ。
    # スクロール過渡で最下段に半分入りかけのフレームを弾く。union で他フレームに
    # 同技が出れば失われない)。
    bottom_limit = layout.move_slot_ys[-1] + layout.move_slot_h
    for text_top, icon_top in tops:
        tctr = _text_band_center(image, layout, text_top, spacing)
        if tctr is None:
            continue
        by = max(0, tctr - layout.move_slot_h // 2)
        if by + layout.move_slot_h > bottom_limit:
            continue
        name_boxes.append(
            Box(layout.move_slot_x, by, layout.move_slot_w, layout.move_slot_h)
        )
        icon_tops.append(icon_top)
    if not name_boxes:
        return []
    # キャッシュ照合: 既知クロップの行は認識・fallback とも省く. miss 行だけを
    # y 昇順のまま recognize_in_regions に渡す (同関数は y 昇順前提).
    raw_texts: list[str | None] = [None] * len(name_boxes)
    thumbs: list[np.ndarray | None] = [None] * len(name_boxes)
    miss: list[int] = list(range(len(name_boxes)))
    if ocr_cache is not None:
        miss = []
        for i, b in enumerate(name_boxes):
            thumbs[i] = _row_thumb(image, b)
            cached = ocr_cache.lookup(thumbs[i])
            if cached is not None:
                raw_texts[i] = cached
            else:
                miss.append(i)
    if miss:
        bucket_results = recognize_in_regions(
            image, [name_boxes[i] for i in miss], allowlist=_TECHNIQUE_ALLOWLIST
        )
        for j, i in enumerate(miss):
            text, conf = _resolve_slot_text(bucket_results[j])
            if conf < _RECOGNIZE_TRUST_CONFIDENCE:
                fallback = ocr_region(image, name_boxes[i], upscale_factor=2.0)
                fb_text = _best_text(fallback)
                if fb_text:
                    text = fb_text
            raw_texts[i] = text
            if ocr_cache is not None and thumbs[i] is not None:
                ocr_cache.store(thumbs[i], text)
    out: list[str] = []
    for i in range(len(name_boxes)):
        text = raw_texts[i] or ""
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


def read_rows_batched(
    images: list[np.ndarray],
    layout: Layout,
    master: MasterData | None = None,
    *,
    accept_threshold: float = 0.7,
    ocr_cache: RowTextCache | None = None,
    recognizer: Recognizer | None = None,
    row_sink=None,
) -> list[list[str]]:
    """複数 image の行を batch 認識で読み取り、各 image ごとの技名リストを返す.

    read_rows と挙動互換で、複数 image の row 認識を1回の batch 呼び出しに
    統合する. 起動 overhead が image 数分の1に減りスループットが上がる.
    cache hit は image ごとに行うので、buffer 内で同一クロップが続いても
    二重認識しない. recognizer 未指定時は get_default_recognizer() を使う.

    row_sink: 各行の (image, text, observed_type) を受け取る optional callable.
    確定済の行 (text 非空) のみ通知される. None なら no-op.
    """
    if not images:
        return []

    spacing = layout.move_slot_ys[1] - layout.move_slot_ys[0]
    bottom_limit = layout.move_slot_ys[-1] + layout.move_slot_h

    per_boxes: list[list[Box]] = []
    per_icon_tops: list[list[int]] = []
    per_thumbs: list[list[np.ndarray | None]] = []
    per_texts: list[list[str | None]] = []
    per_miss: list[list[int]] = []

    for image in images:
        tops = detect_row_tops(image, layout)
        if not tops:
            per_boxes.append([])
            per_icon_tops.append([])
            per_thumbs.append([])
            per_texts.append([])
            per_miss.append([])
            continue
        boxes: list[Box] = []
        icon_tops: list[int] = []
        for text_top, icon_top in tops:
            tctr = _text_band_center(image, layout, text_top, spacing)
            if tctr is None:
                continue
            by = max(0, tctr - layout.move_slot_h // 2)
            if by + layout.move_slot_h > bottom_limit:
                continue
            boxes.append(
                Box(
                    layout.move_slot_x,
                    by,
                    layout.move_slot_w,
                    layout.move_slot_h,
                )
            )
            icon_tops.append(icon_top)
        thumbs: list[np.ndarray | None] = [None] * len(boxes)
        cached_texts: list[str | None] = [None] * len(boxes)
        miss: list[int] = list(range(len(boxes)))
        if ocr_cache is not None:
            miss = []
            for i, b in enumerate(boxes):
                thumbs[i] = _row_thumb(image, b)
                cached = ocr_cache.lookup(thumbs[i])
                if cached is not None:
                    cached_texts[i] = cached
                else:
                    miss.append(i)
        per_boxes.append(boxes)
        per_icon_tops.append(icon_tops)
        per_thumbs.append(thumbs)
        per_texts.append(cached_texts)
        per_miss.append(miss)

    # batch recognize: cache miss だけ集めて 1 回で
    miss_imgs: list[np.ndarray] = []
    miss_boxes: list[list[Box]] = []
    miss_image_idx: list[int] = []
    for img_i, miss in enumerate(per_miss):
        if miss:
            miss_imgs.append(images[img_i])
            miss_boxes.append([per_boxes[img_i][i] for i in miss])
            miss_image_idx.append(img_i)

    if miss_imgs:
        rec = recognizer if recognizer is not None else get_default_recognizer()
        results_per_image = rec.recognize_batch(
            miss_imgs, miss_boxes, allowlist=_TECHNIQUE_ALLOWLIST
        )
        for k, img_i in enumerate(miss_image_idx):
            miss = per_miss[img_i]
            image = images[img_i]
            boxes = per_boxes[img_i]
            thumbs = per_thumbs[img_i]
            cached_texts = per_texts[img_i]
            results = results_per_image[k]
            for j, i in enumerate(miss):
                text, conf = _resolve_slot_text(results[j])
                if conf < _RECOGNIZE_TRUST_CONFIDENCE:
                    fallback = ocr_region(
                        image, boxes[i], upscale_factor=2.0
                    )
                    fb_text = _best_text(fallback)
                    if fb_text:
                        text = fb_text
                cached_texts[i] = text
                if ocr_cache is not None and thumbs[i] is not None:
                    ocr_cache.store(thumbs[i], text)

    out: list[list[str]] = []
    for img_i, image in enumerate(images):
        names: list[str] = []
        boxes = per_boxes[img_i]
        icon_tops = per_icon_tops[img_i]
        cached_texts = per_texts[img_i]
        for i in range(len(boxes)):
            text = cached_texts[i] or ""
            obs_type = None
            if master is not None and text:
                obs_type = observed_slot_type(image, layout, icon_tops[i])
                text = normalize_slot_text(
                    text,
                    master,
                    accept_threshold=accept_threshold,
                    observed_type=obs_type,
                )
            if text:
                names.append(text)
                if row_sink is not None:
                    b = boxes[i]
                    crop = image[b.y : b.y + b.h, b.x : b.x + b.w]
                    row_sink(crop, text, obs_type)
        out.append(names)
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
