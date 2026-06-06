"""collector のポケモン名 → 母集合名の整合を検証する.

collector (data/master/pokemon.json) の各ポケモン名が、手動対応表
(data/champions_name_map.json) を適用した後で pokemon-champions-data の収録対象
一覧 (母集合) に必ず落ちることを確認する。母集合のフォーム改名・collector マスタの
更新・対応表のタイポといった「silent drift」を、collector 側で早期に検出する
(母集合リポジトリの CI 失敗を待たない)。

検査項目:
  - 対応表のキーが collector の実在ポケモン名か (タイポ・マスタ更新ずれ)
  - 対応表の値が母集合に実在するか (母集合のフォーム改名)
  - 非メガの全 collector 名が対応表適用後に母集合へ落ちるか (フォーム差分の取りこぼし)

メガ進化はボックス画面に基本フォームしか格納されず種族名特定で出力され得ないため
検査対象から除く (理由は data/champions_name_map.json の _comment 参照)。

Usage:
    uv run python scripts/check_champions_names.py <母集合 pokemon.json のパス>
    # 例: uv run python scripts/check_champions_names.py \
    #       ../pokemon-champions-data/data/pokemon.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COLLECTOR_POKEMON = _REPO_ROOT / "data" / "master" / "pokemon.json"
_NAME_MAP = _REPO_ROOT / "data" / "champions_name_map.json"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    champ_path = Path(sys.argv[1])
    if not champ_path.exists():
        print(f"母集合 pokemon.json が見つからない: {champ_path}", file=sys.stderr)
        return 2

    collector = json.loads(_COLLECTOR_POKEMON.read_text(encoding="utf-8"))
    name_map = json.loads(_NAME_MAP.read_text(encoding="utf-8")).get("map", {})
    champ = json.loads(champ_path.read_text(encoding="utf-8"))
    if not isinstance(champ, list) or not all(isinstance(n, str) for n in champ):
        print("母集合 pokemon.json は文字列の配列である必要がある", file=sys.stderr)
        return 2
    champ_set = set(champ)
    collector_names = {p["name"] for p in collector}

    errors: list[str] = []

    # 対応表キーが collector の実在名か
    for k in name_map:
        if k not in collector_names:
            errors.append(
                f"対応表のキーが collector に無い: {k!r} "
                "(タイポ、または data/master/pokemon.json の更新ずれ)"
            )
    # 対応表の値が母集合に実在するか
    for k, v in name_map.items():
        if v not in champ_set:
            errors.append(
                f"対応表の写像先が母集合に無い: {k!r} -> {v!r} "
                "(母集合のフォーム改名の可能性。対応表を更新)"
            )

    # 非メガの全 collector 名が対応表適用後に母集合へ落ちるか
    unmapped: list[str] = []
    for p in collector:
        if p.get("category") == "mega":
            continue
        resolved = name_map.get(p["name"], p["name"])
        if resolved not in champ_set:
            unmapped.append(f"{p['name']!r} -> {resolved!r}")
    if unmapped:
        errors.append(
            "対応表適用後も母集合に落ちない名前があります "
            f"({len(unmapped)} 件) — 対応表に写像を追加してください: {unmapped}"
        )

    if errors:
        print(f"整合検証 失敗 ({len(errors)} 件):")
        for e in errors:
            print(f"  - {e}")
        return 1

    mapped = sum(1 for p in collector if p["name"] in name_map)
    print(
        f"整合OK: collector {len(collector_names)} 種 (うち対応表で写像 {mapped} 種) / "
        f"母集合 {len(champ_set)} 種、非メガの全名が母集合へ落ちる"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
