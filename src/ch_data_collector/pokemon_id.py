"""ポケモン特定: ボックス画面の種族名 OCR → POKEMON_DATA fuzzy match.

ポケモンの特定はボックス画面右パネルの「種族名行」だけで行う。詳細画面の
名前はニックネームのことがあり、別ポケモンの名前を付けられていると誤特定する
ため特定には使わない。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from ch_data_collector.learnset import _kana_normalize
from ch_data_collector.master_data import MasterData, Pokemon
from ch_data_collector.ocr import ocr_region
from ch_data_collector.screen_layouts import Layout


@dataclass(frozen=True)
class PokemonCandidate:
    pokemon: Pokemon
    score: float


# OCR はカタカナを字形の似た漢字へ誤読することがある (ミ→三, ロ→口, カ→力 等)。
# ポケモン種族名はカタカナ/ひらがなのみで漢字を含まないため、種族名照合の前に
# これら漢字をカタカナへ畳む。技名照合 (_kana_normalize) には波及させず、ここだけで
# 行う (技名は漢字を含み得るため全体に入れると誤照合を招く)。
# 例: ミミロップ箱種族名が「三三ロップ」と読まれ 0.60 で閾値割れ → 畳んで 1.00。
_OCR_KANJI_TO_KANA = str.maketrans(
    {
        "三": "ミ",
        "口": "ロ",
        "力": "カ",
        "工": "エ",
        "八": "ハ",
        "卜": "ト",
        "夕": "タ",
        "千": "チ",
        "才": "オ",
        "厶": "ム",
        "二": "ニ",
    }
)


def _normalize_ocr_species(ocr_name: str) -> str:
    """種族名 OCR テキストを照合用に正規化する (漢字字形フォールド + かな正規化)."""
    return _kana_normalize(ocr_name.translate(_OCR_KANJI_TO_KANA))


def _base_form_tokens(name: str) -> tuple[str, str]:
    """master ポケモン名を (ベース名, フォームトークン) に分割する.

    フォーム持ちは ``ベース名(トークン)`` 形式 (例 ``ギルガルド(シールド)``)。
    ``マスカーニャ`` のような非フォームは (名前, "") を返す。NFKC で全角括弧
    ``（）`` も半角に畳んでから判定する。
    """
    n = unicodedata.normalize("NFKC", name)
    if "(" in n and n.rstrip().endswith(")"):
        i = n.index("(")
        return n[:i].strip(), n[i + 1 : n.rindex(")")].strip()
    return n.strip(), ""


def fuzzy_match_pokemon(
    ocr_name: str,
    master: MasterData,
    *,
    top_k: int = 5,
) -> list[PokemonCandidate]:
    if not ocr_name:
        return []
    norm_ocr = _normalize_ocr_species(ocr_name)
    cands: list[PokemonCandidate] = []
    for p in master.pokemon:
        # 濁点/半濁点/小書きを畳んで OCR の読みブレ (グ↔ク, ガ↔カ 等) を吸収
        full = SequenceMatcher(None, norm_ocr, _kana_normalize(p.name)).ratio()
        base, token = _base_form_tokens(p.name)
        # フォーム持ちは特定の処理を入れる. ゲームのボックス種族名行は
        # 「ベース名 フォーム名」と表示する (例「ギルガルド シールドフォルム」)
        # 一方 master 名は「ギルガルド(シールド)」で末尾表記が大きく違うため、
        # 全文字列の ratio が下がり (実測 0.62) 採用閾値 0.7 未満で正解が捨てられる.
        # フォームトークンが OCR に含まれる = そのフォームが指定されている場合に
        # 限り、ベース名を OCR 先頭と照合して高スコアを出し閾値を越えさせる.
        # トークン不在なら full のまま (低い) にして別フォーム/ベースを誤って
        # 押し上げない (トークン一致 かつ ベース先頭一致 の二重ゲート).
        if token:
            ntoken = _kana_normalize(token)
            if ntoken and ntoken in norm_ocr:
                nbase = _kana_normalize(base)
                prefix = (
                    SequenceMatcher(None, nbase, norm_ocr[: len(nbase)]).ratio()
                    if nbase
                    else 0.0
                )
                name_score = max(full, prefix)
            else:
                name_score = full
        else:
            name_score = full
        cands.append(PokemonCandidate(pokemon=p, score=name_score))
    cands.sort(key=lambda c: c.score, reverse=True)
    return cands[:top_k]


def read_box_species(
    image: np.ndarray,
    layout: Layout,
    master: MasterData,
    *,
    top_k: int = 5,
) -> list[PokemonCandidate]:
    """ボックス画面右パネルの種族名行を読みポケモン候補を返す.

    詳細画面はニックネーム付きだと種族名でなくニックネームが映るため、ポケモンの
    特定はこの種族名行で行う。box_name 域 (種族名行) を OCR し、各テキスト断片を
    pokemon.json に fuzzy match して最良候補を返す。
    """
    results = ocr_region(image, layout.box_name, upscale_factor=3.0)
    texts = [t for r in results if len(t := r.text.strip()) >= 2]
    cands: list[PokemonCandidate] = []
    for text in texts:
        cands += fuzzy_match_pokemon(text, master, top_k=top_k)
    # フォーム持ちの種族名行はゲーム表示「ベース名 フォーム名」が OCR で別断片に
    # 割れることがある (例 'ギルガルド' + 'シールドフォルム')。断片単位だと
    # fuzzy_match_pokemon のフォームトークン判定がベース名と別文字列になり効かず、
    # 正解が採用閾値未満で捨てられる。連結文字列でも照合してトークンを揃える。
    if len(texts) >= 2:
        cands += fuzzy_match_pokemon(" ".join(texts), master, top_k=top_k)
    cands.sort(key=lambda c: c.score, reverse=True)
    # ポケモンID重複を除き最良のみ残す
    seen: set[int] = set()
    out: list[PokemonCandidate] = []
    for c in cands:
        if c.pokemon.id in seen:
            continue
        seen.add(c.pokemon.id)
        out.append(c)
    return out[:top_k]
