"""複数の収集結果 json を統合し提出用 json を生成する.

Usage:
    uv run python scripts/merge_results.py result_a.json result_b.json -o submit_YYYYMMDD.json

- ``(unknown*)`` キー (種族未特定セグメント) は除外
- 同名ポケモンは技を union (初出順を保持)
- 既収録 (../pokemon-champions-data/data/collected.jsonl) に無いポケモンだけの
  新規版 ``<out>_new.json`` も併せて出力する
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_COLLECTED = (
    Path(__file__).resolve().parents[2]
    / "pokemon-champions-data"
    / "data"
    / "collected.jsonl"
)


def _load_collected() -> set[str]:
    if not _COLLECTED.exists():
        print(f"note: {_COLLECTED} not found; _new.json は全件と同じになる")
        return set()
    names: set[str] = set()
    for line in _COLLECTED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.update(json.loads(line))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, list[str]] = {}
    for path in args.results:
        data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
        for name, moves in data.items():
            if name.startswith("(unknown"):
                continue
            if name in merged:
                seen = set(merged[name])
                merged[name].extend(m for m in moves if m not in seen)
            else:
                merged[name] = list(moves)

    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(merged)} pokemon)")

    collected = _load_collected()
    new = {k: v for k, v in merged.items() if k not in collected}
    new_path = args.output.with_stem(args.output.stem + "_new")
    new_path.write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {new_path} ({len(new)} new pokemon)")
    skipped = sorted(k for k in merged if k in collected)
    if skipped:
        print(f"already collected (excluded from _new): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
