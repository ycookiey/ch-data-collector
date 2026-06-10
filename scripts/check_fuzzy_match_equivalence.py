"""fuzzy_match_move (上界プレフィルタ + 打ち切り) と総当たり素朴版の完全一致検証.

fuzzy_match_move は Indel 類似度 (rapidfuzz) を SequenceMatcher ratio の上界として
使い計算を打ち切るが、返す候補・スコア・順位は総当たりと完全一致するのが前提。
このスクリプトは master 全技名 + ランダム変異文字列 + タイプタイブレーカ込みで
その前提を検証する (照合ロジック変更時に流す)。

    uv run python scripts/check_fuzzy_match_equivalence.py
"""

import random
from difflib import SequenceMatcher

from ch_data_collector.learnset import (
    _TYPE_MATCH_BONUS,
    _kana_normalize,
    fuzzy_match_move,
)
from ch_data_collector.master_data import load_master_data

master = load_master_data(None)
types = sorted({m.type for m in master.moves}) + [None]


def naive(ocr_text, top_k=3, observed_type=None):
    nq = _kana_normalize(ocr_text)
    cands = [
        (m, SequenceMatcher(None, nq, _kana_normalize(m.name)).ratio())
        for m in master.moves
    ]
    if observed_type:
        cands.sort(
            key=lambda t: t[1]
            + (_TYPE_MATCH_BONUS if t[0].type == observed_type else 0.0),
            reverse=True,
        )
    else:
        cands.sort(key=lambda t: t[1], reverse=True)
    return cands[:top_k]


rng = random.Random(42)
kana = list(
    "あいうえおかきくけこがぎぐげごさしすせそたちつてとなにぬねのはひふへほ"
    "ばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをん"
    "アイウエオカキクケコガギグゲゴサシスセソタチツテトナニヌネノハヒフヘホ"
    "バビブベボパピプペポマミムメモヤユヨラリルレロワヲンー"
)

cases = []
names = [m.name for m in master.moves]
cases += names[:200]
for name in rng.sample(names, 300):
    s = list(name)
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(["sub", "delete", "ins"])
        if op == "sub" and s:
            s[rng.randrange(len(s))] = rng.choice(kana)
        elif op == "delete" and len(s) > 1:
            s.pop(rng.randrange(len(s)))
        else:
            s.insert(rng.randrange(len(s) + 1), rng.choice(kana))
    cases.append("".join(s))
cases += ["かつたへパーー", "ーーー", "ア", "ねこと", "シャトーホール"]

bad = 0
for text in cases:
    ot = rng.choice(types)
    for top_k in (1, 3):
        got = fuzzy_match_move(text, master, top_k=top_k, observed_type=ot)
        want = naive(text, top_k=top_k, observed_type=ot)
        ok = len(got) == len(want) and all(
            g.move.id == w[0].id and g.score == w[1]
            for g, w in zip(got, want)
        )
        if not ok:
            bad += 1
            print("MISMATCH:", repr(text), "type=", ot, "k=", top_k)
            print("  got :", [(g.move.name, round(g.score, 4)) for g in got])
            print("  want:", [(w[0].name, round(w[1], 4)) for w in want])
print(f"{len(cases) * 2} 比較, 不一致 {bad} 件")
if bad:
    raise SystemExit(1)
