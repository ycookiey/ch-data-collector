"""towakey/pokedex (MIT) の waza_list.json から全技 moves.json を生成する.

公開リポジトリ towakey/pokedex の各世代 `waza_list.json` を union し、
ゲーム内日本語表記の全技辞書を作る。Champions 実装技の「絞り込み根拠」を
持たずに済み (= 公開リポジトリの全技を収集した、と説明できる)、
威力分割 (おはかまいり) や派生 ((N発)) も源流に存在しない。

技名で fuzzy match するため type/pp は照合に使わないが、Move dataclass の
互換のため保持する (type はソースの日本語表記のまま).

Usage:
    uv run python scripts/build_moves_from_pokedex.py <pokedex-src-dir>
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

MASTER_DIR = Path(__file__).resolve().parents[1] / "data" / "master"

# towakey/pokedex の世代ディレクトリを古い順に並べる. waza_list.json を union
# する際に新しい世代の情報で上書き (後勝ち) するための順序定義.
# アルファベット順 glob だと Black_White が先頭に来てしまい、先勝ちすると
# タイプ変更技 (つきのひかり: Gen2-5 ノーマル → Gen6+ フェアリー 等) が旧タイプの
# まま残る. 末尾ほど新しく最優先 (Champions は最新作 Scarlet_Violet に最も近い).
GEN_ORDER = [
    "Red_Green_Blue_Pikachu",
    "Gold_Silver_Crystal",
    "Ruby_Sapphire_Emerald",
    "FireRed_LeafGreen",
    "Diamond_Pearl_Platinum",
    "HeartGold_SoulSilver",
    "Black_White",
    "Black2_White2",
    "X_Y",
    "OmegaRuby_AlphaSapphire",
    "Sun_Moon",
    "UltraSun_UltraMoon",
    "Sword_Shield",
    "LegendsArceus",
    "Scarlet_Violet",
    "LegendsZA",
]

# towakey/pokedex に無いが Champions に実在する技の手動補完. towakey は SV 不在ポケの
# 技を一部欠く (例: ガラルマッギョのトラバサミ). 公開ソースから再生成する原則の最小例外.
SUPPLEMENT_MOVES = [
    {"name": "トラバサミ", "type": "くさ", "pp": 15},
]


def _gen_rank(path: Path) -> int:
    """世代ディレクトリの新しさ順位 (大きいほど新しい).

    GEN_ORDER 外の未知世代は最古扱い (-1) とし、既知の最新世代の情報が
    上書きで勝つようにする (未知世代に最新タイプを奪われない安全側).
    """
    gen = path.parent.name
    return GEN_ORDER.index(gen) if gen in GEN_ORDER else -1


def parse_pp(raw: object) -> int:
    """pp は ' 10' のような前置空白付き文字列のことがある."""
    try:
        return int(str(raw).strip())
    except (ValueError, TypeError):
        return 0


def normalize_move_key(name: str) -> str:
    """技名の表記ゆれ吸収キー (同一技の世代間表記ゆれを同一視する).

    towakey/pokedex は複数世代を収録するため同じ技が全角/半角・スペース有無で
    揺れる (`１０まんボルト`↔`10まんボルト` 等)。これを潰したキーで群化し、群ごとに
    最新世代の綴りへ集約する。OCR 結果の fuzzy match 先がブレないよう master に
    同一技の別表記を残さない。

    ひらがな↔カタカナのカナ種の違い (`ねこにこばん`↔`ネコにこばん`) は別表記として
    両方残す (公式綴りが世代で変わった事例であり、別技を誤併合するリスクも避ける)。
    """
    s = unicodedata.normalize("NFKC", name)
    return s.replace(" ", "").replace("　", "")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1]) / "pokedex"
    if not src.exists():
        print(f"pokedex dir not found: {src}")
        return 2

    union: dict[str, dict] = {}  # 技名 -> info. 世代を古→新で処理し新情報で上書き.
    best_rank: dict[str, int] = {}  # 技名 -> 登場する最新世代の順位.
    for waza_list_path in sorted(src.glob("*/waza_list.json"), key=_gen_rank):
        rank = _gen_rank(waza_list_path)
        wl = json.loads(waza_list_path.read_text(encoding="utf-8")).get(
            "waza_list", {}
        )
        for _game, moves in wl.items():
            if not isinstance(moves, dict):
                continue
            for name, info in moves.items():
                # 後勝ち: 新しい世代の dict 情報で上書きする. これでタイプ変更技
                # (つきのひかり: Gen2-5 ノーマル → Gen6+ フェアリー 等) が最新作の
                # タイプになる. dict でない/空のゲームは type/pp を持たないので、
                # まだ何も無い時だけ空で登録する (ゴミは後段の real_names で除外).
                if isinstance(info, dict) and info:
                    union[name] = info
                    if rank > best_rank.get(name, -2):
                        best_rank[name] = rank
                else:
                    union.setdefault(name, {})

    # dict 情報を持たない (= どのゲームでも文字列だった) キーはスキーマ違いで
    # 紛れ込んだフィールド名等のゴミなので除外する.
    real_names = [n for n in union if union[n]]

    # 表記ゆれキーで群化し、各群から正規表記 1 件 (最新世代優先) を選ぶ. 同順位は
    # NFKC 正規形・スペース無しを優先して決定的に選ぶ.
    def _pick_key(name: str) -> tuple:
        nfkc = unicodedata.normalize("NFKC", name)
        return (
            best_rank.get(name, -2),
            nfkc == name,
            " " not in name and "　" not in name,
            name,
        )

    groups: dict[str, str] = {}
    collapsed: dict[str, list[str]] = {}
    for name in real_names:
        k = normalize_move_key(name)
        collapsed.setdefault(k, []).append(name)
        if k not in groups or _pick_key(name) > _pick_key(groups[k]):
            groups[k] = name
    for k, variants in sorted(collapsed.items()):
        if len(variants) > 1:
            chosen = groups[k]
            dropped = sorted(v for v in variants if v != chosen)
            print(f"  表記ゆれ集約: {chosen!r} ← {dropped}")

    canonical = sorted(groups.values())
    # towakey 欠落の Champions 実在技を補完 (既存と重複しないもののみ)
    existing = set(canonical)
    for s in SUPPLEMENT_MOVES:
        if s["name"] not in existing:
            union[s["name"]] = {"type": s["type"], "pp": s["pp"]}
            canonical.append(s["name"])
    canonical = sorted(canonical)
    moves_out = [
        {
            "id": i,
            "name": name,
            "type": str(union[name].get("type", "")),
            "pp": parse_pp(union[name].get("pp")),
        }
        for i, name in enumerate(canonical, start=1)
    ]

    out_path = MASTER_DIR / "moves.json"
    out_path.write_text(
        json.dumps(moves_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"全技 {len(moves_out)} 件を {out_path} に書き出した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
