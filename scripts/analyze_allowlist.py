"""master技名に実際に出現する文字を集計して allowlist の絞り込み余地を見る."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ch_data_collector.learnset import _TECHNIQUE_ALLOWLIST


def char_category(c: str) -> str:
    code = ord(c)
    if 0x3041 <= code <= 0x3096:
        return "hiragana"
    if 0x30A1 <= code <= 0x30FA:
        return "katakana"
    if c == "ー":
        return "chouon"
    if c == "ヴ":
        return "v_kana"
    if c == "・":
        return "middle_dot"
    if c.isdigit():
        return "digit"
    if c in "()（）":
        return "paren"
    if c.isspace():
        return "space"
    return "other"


def main() -> int:
    master_path = Path(__file__).resolve().parents[1] / "data" / "master" / "moves.json"
    moves: list[dict] = json.loads(master_path.read_text(encoding="utf-8"))

    counter: Counter[str] = Counter()
    for m in moves:
        name = m["name"]
        for c in name:
            counter[c] += 1

    by_cat: dict[str, list[tuple[str, int]]] = {}
    for c, n in counter.items():
        cat = char_category(c)
        by_cat.setdefault(cat, []).append((c, n))

    total_chars = len(counter)
    print(f"=== master技名 ({len(moves)}技) の出現文字種 ({total_chars} 種) ===\n")
    for cat in sorted(by_cat.keys()):
        chars = sorted(by_cat[cat], key=lambda x: -x[1])
        chars_str = "".join(c for c, _ in chars)
        print(f"[{cat}] {len(chars)}種  {chars_str}")
        # 高頻度上位5
        top5 = ", ".join(f"{c}:{n}" for c, n in chars[:5])
        print(f"    top5: {top5}\n")

    # 現在の allowlist と比較 (本番側 learnset._build_allowlist の結果を直接参照し
    # 二重管理を避ける).
    current_allowlist = _TECHNIQUE_ALLOWLIST
    used = set(counter.keys())
    available = set(current_allowlist)
    unused = sorted(available - used)
    print(f"=== allowlist サイズ比較 ===")
    print(f"現在 allowlist: {len(available)} 文字")
    print(f"実出現: {len(used)} 文字")
    print(f"削れる: {len(unused)} 文字")
    print(f"削減対象 (現allowlistにあるが master 技名で未使用): {''.join(unused)}")

    # 提案 allowlist
    suggested = "".join(sorted(used))
    print(f"\n=== 提案 allowlist ({len(suggested)}文字) ===")
    print(suggested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
