"""collector の pokemon.json が pokemon-champions-data (母集合) と一致するか検証.

collector の `data/master/pokemon.json` は母集合 (`pokemon-champions-data/data/pokemon.json`)
の文字列リストを直接コピーしたもの。collector 側で手を入れる必要がない (= 母集合と
完全一致) ことを CI/手動で確認する。両者の差分があれば collector 側を再同期する。

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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    champ_path = Path(sys.argv[1])
    if not champ_path.exists():
        print(f"母集合 pokemon.json が見つからない: {champ_path}", file=sys.stderr)
        return 2

    collector = json.loads(_COLLECTOR_POKEMON.read_text(encoding="utf-8"))
    champ = json.loads(champ_path.read_text(encoding="utf-8"))

    if not isinstance(collector, list) or not all(isinstance(n, str) for n in collector):
        print("collector pokemon.json は文字列の配列である必要がある", file=sys.stderr)
        return 1
    if not isinstance(champ, list) or not all(isinstance(n, str) for n in champ):
        print("母集合 pokemon.json は文字列の配列である必要がある", file=sys.stderr)
        return 2

    only_collector = sorted(set(collector) - set(champ))
    only_champ = sorted(set(champ) - set(collector))

    if only_collector or only_champ:
        print("差分あり (collector を母集合に再同期する必要あり):")
        if only_collector:
            print(f"  collector のみ ({len(only_collector)}件): {only_collector}")
        if only_champ:
            print(f"  母集合のみ ({len(only_champ)}件): {only_champ}")
        return 1

    print(f"整合OK: collector {len(collector)} 種 = 母集合 {len(champ)} 種 (完全一致)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
