"""ch-data-collector CLI エントリポイント."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import questionary

from ch_data_collector.master_data import load_master_data
from ch_data_collector.pipeline import (
    PipelineConfig,
    PokemonResult,
    Segment,
    collect_segment,
    index_segments,
    segment_from_dict,
    segment_to_dict,
)


def _video_resolution(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        # 開けない/メタデータ不正な動画は get() が 0 を返し、(0,0) という偽の
        # 解像度グループに紛れ込む (後段 extract_frames で初めて RuntimeError に
        # なり "0x0" の不可解なメッセージでバッチ全体が落ちる). ここで明示エラーに
        # する.
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise RuntimeError(f"could not read resolution (got {w}x{h}): {path}")
        return (w, h)
    finally:
        cap.release()


def _group_by_resolution(videos: list[Path]) -> dict[tuple[int, int], list[Path]]:
    """動画を解像度でグループ化する.

    pipeline は最初のフレームで layout を固定し画面分類テンプレも解像度依存のため、
    解像度の異なる動画を1回の run_pipeline に混ぜると後続が座標ズレで総崩れする。
    解像度ごとに run_pipeline を分けることでこれを防ぐ (同名ポケモンは出力時に
    集合マージされる)。
    """
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for v in videos:
        groups[_video_resolution(v)].append(v)
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ch-data-collector",
        description="ポケモンチャンピオンズの技プールを動画から半自動収集する",
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("learnset.json"),
    )
    parser.add_argument("--master-dir", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="処理開始秒 (連結後の通算秒). 指定区間だけを部分処理する",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="処理終了秒 (連結後の通算秒, この秒以降は処理しない)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="fuzzy match 採用閾値"
    )
    parser.add_argument(
        "--index-fps",
        type=float,
        default=5.0,
        help="フェーズ1 (境界検出+種族名特定) の fps. 既定 5 (低fpsで足りる)",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="既存の segments.json を読みフェーズ1を省く (1匹再収集の高速化)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="収集対象を絞る. セグメント番号 (1始まり) か種族名のカンマ区切り",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="対話確認しない (CI向け)。種族特定は top1、曖昧な技 (閾値未満) は不採用",
    )
    return parser


def _select_pokemon(result: PokemonResult, no_prompt: bool) -> str:
    cands = result.candidates
    if not cands:
        return "(unknown)"
    top = cands[0]
    if top.score >= 0.85 or no_prompt:
        return top.pokemon.name
    choices = [f"{c.pokemon.name}  (score={c.score:.2f})" for c in cands]
    choices.append("(skip)")
    ts = result.detail_frame_ts
    ts_s = f"~{ts:.1f}s" if ts is not None else "時刻不明"
    answer = questionary.select(
        f"ポケモン候補を選んでください (検出時刻 {ts_s})",
        choices=choices,
    ).ask()
    if answer is None or answer == "(skip)":
        return "(unknown)"
    idx = choices.index(answer)
    return cands[idx].pokemon.name


def _resolve_ambiguous_moves(
    ambiguous: list[tuple[str, list]],
    no_prompt: bool,
) -> list[str]:
    # --no-prompt では曖昧技 (accept_threshold 未満) を採用しない。低 score の OCR
    # 誤読 (ゴミ raw が score 0.4-0.57 で最近傍技に着地) を top1 強制採用すると確定技に
    # 偽技が混入する (実測: ピクシー raw='かつたへパーー'→オーバーヒート 0.43 等)。確定技
    # (>=accept_threshold) は別経路で出力済み。曖昧技は対話時に選ばれたものだけ拾う
    # (種族特定 _select_pokemon は 1 件確定が要るので no-prompt でも top1 を採るが、技は
    # 複数かつ取りこぼし安全側でよく非対称)。
    if no_prompt:
        return []
    out: list[str] = []
    for raw, cands in ambiguous:
        if not cands:
            continue
        choices = [
            f"{c.move.name}  (score={c.score:.2f})" for c in cands
        ] + ["(skip)"]
        answer = questionary.select(
            f"OCR={raw!r} の候補を選択",
            choices=choices,
        ).ask()
        if answer and answer != "(skip)":
            idx = choices.index(answer)
            out.append(cands[idx].move.name)
    return out


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _emit_result(
    output: dict[str, list[str]],
    r: PokemonResult,
    seg_no: int,
    no_prompt: bool,
) -> str:
    """1セグメントの結果を output に統合し進捗を表示する (逐次出力用)."""
    print(f"\n--- segment {seg_no} ---")
    print(f"  detail_frame_ts: {r.detail_frame_ts}")
    print(f"  move_list_frames: {r.move_list_frame_count}")
    print(f"  raw move names ({len(r.raw_move_names)}):")
    for n in r.raw_move_names:
        print(f"    - {n!r}")
    name = _select_pokemon(r, no_prompt)
    confirmed = [m.name for m in r.moves]
    confirmed += _resolve_ambiguous_moves(r.ambiguous_moves, no_prompt)
    # 確定技と曖昧解決が同一技を含むことがあるので順序保持で重複除去する.
    confirmed = _dedupe(confirmed)
    # 同一ポケモン重複は集合マージ. "(unknown)" は別個体衝突を避けセグメント毎に
    # 一意キーにする (別ポケの技が合算されるのを防ぐ).
    if name == "(unknown)":
        output[f"(unknown {seg_no})"] = confirmed
    elif name in output:
        existing = set(output[name])
        for m in confirmed:
            if m not in existing:
                output[name].append(m)
                existing.add(m)
    else:
        output[name] = confirmed
    print(f"  → {name}: {len(confirmed)} moves")
    return name


def _merge_second_pass(
    output: dict[str, list[str]],
    seg_order: list[tuple[int, str]],
    *,
    min_coverage: float = 0.9,
) -> None:
    """同一個体の 2 周目以降のセグメント (unknown) を直前の確定ポケへ統合する.

    技教え画面を複数回開く / 1 周目を閉じて開き直す等で、box を経由しない周回が
    unknown になることがある。確定ポケを anchor として保持し、それに続く unknown を
    順に anchor へ包含判定 → 成立すれば union 統合する。統合後 anchor の技集合は
    広がるので、3 周目以降は「それまでの統合結果」と比較され、何周でも畳み込める。

    判定: 間に別ポケの box 選択 (= 別の確定セグメント) が挟まらない隣接区間で、
    技集合が対称的にほぼ包含 (小さい側の min_coverage 以上が大きい側に含まれる) なら
    同一個体とみなす。どちらの周回が広くてもよい (union なので結果は全技)。包含が
    崩れる別ポケは統合せず unknown のまま残し可視化する (黙ったデータ汚染を防ぐ安全弁。
    anchor は維持し、さらなる周回に備える)。
    """
    anchor: str | None = None  # 連続 unknown 区間の統合先 (直前の確定ポケ名)
    for seg_no, name in seg_order:
        if name != "(unknown)":
            anchor = name if name in output else None
            continue
        uk_key = f"(unknown {seg_no})"
        if uk_key not in output or anchor is None or anchor not in output:
            continue
        uk_moves = set(output[uk_key])
        anchor_moves = set(output[anchor])
        small, big = sorted((uk_moves, anchor_moves), key=len)
        if not small:
            continue
        coverage = len(small & big) / len(small)
        if coverage < min_coverage:
            continue  # 別個体の疑い → unknown を残す (anchor は維持)
        merged = output[anchor] + [m for m in output[uk_key] if m not in anchor_moves]
        output[anchor] = merged
        del output[uk_key]
        print(
            f"  周回結合: {uk_key} → {anchor} "
            f"(被覆率{coverage:.2f}, {len(anchor_moves)}+{len(uk_moves)}→{len(merged)})"
        )


def _write_output(output: dict[str, list[str]], path: Path) -> None:
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _segments_sidecar(output: Path) -> Path:
    return Path(str(output) + ".segments.json")


def _write_output_segments(
    path: Path,
    index_fps: float,
    group_specs: list[tuple[list[Path], tuple[int, int], list[Segment]]],
) -> None:
    """フェーズ1の結果を segments.json に保存する (--segments で再利用)."""
    data = {
        "index_fps": index_fps,
        "groups": [
            {
                "videos": [str(v) for v in vids],
                "resolution": [res[0], res[1]],
                "segments": [segment_to_dict(s) for s in segs],
            }
            for vids, res, segs in group_specs
        ],
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _select_segments(
    group_specs: list[tuple[list[Path], tuple[int, int], list[Segment]]],
    only: str | None,
) -> set[int]:
    """--only を解釈し収集する全体セグメント番号 (1始まり) の集合を返す.

    only が None なら全件。トークンは番号 (1始まり) か種族名。
    """
    flat: list[Segment] = [s for _, _, segs in group_specs for s in segs]
    if only is None:
        return set(range(1, len(flat) + 1))
    wanted: set[int] = set()
    tokens = [t.strip() for t in only.split(",") if t.strip()]
    names: dict[str, int] = {}
    for i, s in enumerate(flat):
        col = s.pokemon[0].pokemon.name if s.pokemon else "(unknown)"
        names[col] = i + 1
    for tok in tokens:
        if tok.isdigit():
            wanted.add(int(tok))
        elif tok in names:
            wanted.add(names[tok])
        else:
            print(f"--only: 不明な指定を無視: {tok!r}", file=sys.stderr)
    return wanted


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for v in args.videos:
        if not v.exists():
            print(f"video not found: {v}", file=sys.stderr)
            return 2
    if args.fps <= 0:
        print(f"--fps must be positive (got {args.fps})", file=sys.stderr)
        return 2
    if (
        args.start is not None
        and args.end is not None
        and args.end <= args.start
    ):
        print(
            f"--end must be greater than --start (got {args.start}, {args.end})",
            file=sys.stderr,
        )
        return 2

    master = load_master_data(args.master_dir)
    print(
        f"loaded master: {len(master.moves)} moves, "
        f"{len(master.pokemon)} pokemon"
    )

    config = PipelineConfig(
        fps=args.fps,
        frames_dir=args.frames_dir,
        accept_threshold=args.threshold,
        verbose=args.verbose,
        start=args.start,
        end=args.end,
        index_fps=args.index_fps,
    )

    # --- 2-phase: フェーズ1 index → フェーズ2 collect (逐次出力) ---
    # group_specs: [(videos, (w,h), [Segment...]) ...]. 解像度ごとに区間 ts が
    # その group の連結タイムライン基準になるため group 単位で扱う.
    group_specs: list[tuple[list[Path], tuple[int, int], list[Segment]]] = []
    classifiers: dict[int, object] = {}

    seg_path = args.segments or _segments_sidecar(args.output)
    if args.segments is not None:
        # 既存 index を読み込みフェーズ1を省く (1匹再収集の高速化)
        data = json.loads(args.segments.read_text(encoding="utf-8"))
        for gi, g in enumerate(data["groups"]):
            vids = [Path(p) for p in g["videos"]]
            segs = [segment_from_dict(d, master) for d in g["segments"]]
            group_specs.append((vids, tuple(g["resolution"]), segs))
        print(f"loaded {args.segments} (index 再利用)")
    else:
        # フェーズ1: 解像度ごとに index
        try:
            groups = _group_by_resolution(args.videos)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        for gi, ((w, h), vids) in enumerate(groups.items()):
            print(
                f"phase1 index: {len(vids)} video(s) @ {w}x{h} "
                f"@ {config.index_fps}fps..."
            )
            segs, classifier = index_segments(vids, master, config)
            group_specs.append((vids, (w, h), segs))
            classifiers[gi] = classifier
        # segments.json を保存 (再実行用)
        _write_output_segments(seg_path, config.index_fps, group_specs)
        print(f"wrote {seg_path}")

    total = sum(len(segs) for _, _, segs in group_specs)
    print(f"detected {total} pokemon segment(s)")

    wanted = _select_segments(group_specs, args.only)

    # フェーズ2: セグメント毎に collect → 逐次 output 書き込み
    output = {}
    seg_order: list[tuple[int, str]] = []
    seg_no = 0
    for gi, (vids, _res, segs) in enumerate(group_specs):
        classifier = classifiers.get(gi)
        for seg in segs:
            seg_no += 1
            if seg_no not in wanted:
                continue
            r = collect_segment(vids, master, config, seg, classifier)
            name = _emit_result(output, r, seg_no, args.no_prompt)
            seg_order.append((seg_no, name))
            _write_output(output, args.output)

    # 同一個体の 2 周目 (unknown) を直前の確定ポケへ統合してから最終書き込み
    _merge_second_pass(output, seg_order)
    _write_output(output, args.output)

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
