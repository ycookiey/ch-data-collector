"""動画 → ポケモン別技プール の end-to-end パイプライン.

1動画(または連結された動画群)を処理し、検出されたポケモン毎に
技プールを返す。複数ポケモン対応のため、各フレームをヘッダで分類して
以下のステートマシンで処理する:

  - MOVE_LIST: 各行を実位置で動的検出して OCR し、読めた技名に投票する
  - DETAIL  : 新しいポケモンの詳細画面 = セグメント境界。直前までの投票を
              commit し、直近に読んだボックス種族名で個体を確定する
  - OTHER   : ボックス・遷移画面。右パネルの種族名を読み pending として保持

セグメント境界は「DETAIL の再来」と「動画末尾」。一周検出は行わない
(連続スクロール対応のため固定スロットでなく動的行検出 + 投票方式)。

高速化:
  - MOVE_LIST 中はスロット領域の縮小サムネ差分 (MSE) が小さければ read_rows
    をスキップする (静止フレームの重複 OCR を省く)。分類自体は毎フレーム行い、
    MOVE_LIST→DETAIL/OTHER の遷移を取りこぼさない。
  - OTHER 中はボックス種族名領域の差分が小さければ read_box_species を省く。
  - 画面分類は両 kind のヘッダテンプレが揃えば matchTemplate で OCR を回避し、
    テンプレで判定不能なフレームのみ OCR にフォールバックする。
  - スクロール中で領域全体の差分が大きいフレームも、行クロップ単位の
    キャッシュ (learnset.RowTextCache) で既読行の再認識を省く。
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 技を採用する最小得票数. 既定 1 = 投票フィルタは無効で、1フレームでも読めた技は
# 採用する. 精度は行クロップの整列で担保し多数決には依存しない方針.
# 2 以上にすると複数フレームで読めた技だけを採用する閾値として働く.
_MIN_VOTES = 1

# ボックス画面で読んだ種族名を採用する最低スコア.
_BOX_SPECIES_MIN = 0.7

import threading
from queue import Queue
from typing import Iterator

import cv2
import numpy as np

from ch_data_collector.classify import (
    ScreenKind,
    TemplateClassifier,
    classify_screen,
)
from ch_data_collector.extract import extract_frames
from ch_data_collector.learnset import (
    RowTextCache,
    read_rows,
    resolve_moves,
)
from ch_data_collector.master_data import MasterData, Move
from ch_data_collector.pokemon_id import (
    PokemonCandidate,
    read_box_species,
)
from ch_data_collector.screen_layouts import Box, Layout, resolve_layout


_SLOT_THUMB_W = 32
_SLOT_THUMB_H = 80  # スロット領域は縦長 (6行 × 65px)
_SLOT_DIFF_THRESHOLD = 2.0

# decoder スレッドが先回りデコードして積むフレームキューのサイズ。
# GPU OCR (~60ms) と CPU デコード (~10-20ms) を重ねて GPU 待ち時間を埋める。
# 大きすぎると frame.image (numpy 配列) のメモリが嵩むので 8 程度に抑える。
_DECODE_QUEUE_MAX = 8


def _prefetch_frames(frames: Iterator) -> Iterator:
    """generator のデコードを別スレッドで先回りする (queue で backpressure).

    main スレッドが OCR で待っている間に decoder スレッドが次のフレームを
    読んでおく。GPU 推論中は GIL が解放されるので Python スレッドでも
    デコード時間と OCR 時間を重ねられる。"""
    q: Queue = Queue(maxsize=_DECODE_QUEUE_MAX)
    sentinel = object()

    def producer() -> None:
        try:
            for f in frames:
                q.put(f)
        except BaseException as e:
            q.put(("__exc__", e))
        finally:
            q.put(sentinel)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__exc__":
            raise item[1]
        yield item


def _slot_region(image: np.ndarray, layout: Layout) -> np.ndarray:
    y0 = layout.move_slot_ys[0]
    y1 = layout.move_slot_ys[-1] + layout.move_slot_h
    x0 = layout.move_slot_x
    x1 = layout.move_slot_x + layout.move_slot_w
    return image[y0:y1, x0:x1]


def _slot_region_thumb(image: np.ndarray, layout: Layout) -> np.ndarray:
    return cv2.resize(
        _slot_region(image, layout),
        (_SLOT_THUMB_W, _SLOT_THUMB_H),
        interpolation=cv2.INTER_AREA,
    )


def _region_thumb(
    image: np.ndarray, box: Box, size: tuple[int, int] = (40, 12)
) -> np.ndarray | None:
    """矩形領域を縮小サムネ化する (フレーム間差分での再 OCR skip 判定用)."""
    sub = image[box.y : box.y + box.h, box.x : box.x + box.w]
    if sub.size == 0:
        return None
    return cv2.resize(sub, size, interpolation=cv2.INTER_AREA)


def _frames_similar(
    a: np.ndarray | None,
    b: np.ndarray | None,
    threshold: float = _SLOT_DIFF_THRESHOLD,
) -> bool:
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
    return diff < threshold


@dataclass
class PokemonResult:
    candidates: list[PokemonCandidate]
    moves: list[Move]
    ambiguous_moves: list[tuple[str, list]] = field(default_factory=list)
    raw_move_names: list[str] = field(default_factory=list)
    detail_frame_ts: float | None = None
    move_list_frame_count: int = 0


@dataclass
class Segment:
    """フェーズ1 (index) が検出する1ポケモン分の区間.

    pokemon = ボックス種族名で特定した候補 (未特定は空). movelist_start/end は
    技一覧フレームの ts 範囲 (index_fps での粗い境界)。フェーズ2 (collect) は
    この範囲を full fps で再走査して技を読む。
    """

    pokemon: list[PokemonCandidate]
    detail_ts: float | None
    movelist_start: float | None
    movelist_end: float | None
    move_list_frame_count: int = 0


@dataclass
class PipelineConfig:
    fps: float = 5.0
    frames_dir: Path | None = None
    accept_threshold: float = 0.7
    verbose: bool = False
    start: float | None = None
    end: float | None = None
    # フェーズ1 (index) のフレームレート. 境界検出と種族名特定のみ行うので低fpsで
    # 足りる (DETAIL/ボックス画面は数秒持続する)。フェーズ2 (collect) は config.fps.
    index_fps: float = 5.0


def run_pipeline(
    videos: list[Path],
    master: MasterData,
    config: PipelineConfig,
) -> list[PokemonResult]:
    """2-phase 処理: フェーズ1 で境界検出+種族名特定、フェーズ2 で技を読む."""
    segments, classifier = index_segments(videos, master, config)
    return [
        collect_segment(videos, master, config, seg, classifier)
        for seg in segments
    ]


def segment_to_dict(seg: Segment) -> dict:
    """Segment を segments.json 保存用の素の dict にする (再実行で読み戻す)."""
    return {
        "pokemon": [
            {"id": c.pokemon.id, "name": c.pokemon.name, "score": c.score}
            for c in seg.pokemon
        ],
        "detail_ts": seg.detail_ts,
        "movelist_start": seg.movelist_start,
        "movelist_end": seg.movelist_end,
        "move_list_frame_count": seg.move_list_frame_count,
    }


def segment_from_dict(d: dict, master: MasterData) -> Segment:
    """segments.json の dict を Segment に復元する (pokemon は id で master 照合)."""
    by_id = {p.id: p for p in master.pokemon}
    pokemon = [
        PokemonCandidate(pokemon=by_id[c["id"]], score=c["score"])
        for c in d.get("pokemon", [])
        if c["id"] in by_id
    ]
    return Segment(
        pokemon=pokemon,
        detail_ts=d.get("detail_ts"),
        movelist_start=d.get("movelist_start"),
        movelist_end=d.get("movelist_end"),
        move_list_frame_count=d.get("move_list_frame_count", 0),
    )


def index_segments(
    videos: list[Path],
    master: MasterData,
    config: PipelineConfig,
) -> tuple[list[Segment], TemplateClassifier]:
    """フェーズ1: 低fpsで画面分類+種族名特定のみ行いセグメント境界を検出する.

    技一覧の OCR はしない。各セグメントの (種族名, detail_ts, 技一覧 ts 範囲) を
    返す。学習済み TemplateClassifier も返し、フェーズ2 で再利用して分類 OCR を
    省く。境界ロジックは process_frames と同じ (DETAIL 再来 = 境界、OTHER で
    ボックス種族名を pending 保持、最新優先)。
    """
    frames = _prefetch_frames(
        extract_frames(
            videos,
            fps=config.index_fps,
            start=config.start,
            end=config.end,
        )
    )
    segments: list[Segment] = []
    layout: Layout | None = None
    classifier = TemplateClassifier()

    pending_box: list[PokemonCandidate] = []
    cur_pokemon: list[PokemonCandidate] | None = None
    cur_detail_ts: float | None = None
    cur_ml_start: float | None = None
    cur_ml_end: float | None = None
    cur_ml_count = 0
    have_open = False

    def open_segment() -> None:
        nonlocal have_open, cur_pokemon, cur_detail_ts
        nonlocal cur_ml_start, cur_ml_end, cur_ml_count
        have_open = True
        cur_pokemon = None
        cur_detail_ts = None
        cur_ml_start = None
        cur_ml_end = None
        cur_ml_count = 0

    def close_segment() -> None:
        nonlocal have_open, pending_box
        # 技一覧フレームが無い区間は出さない (process_frames の「votes 空なら
        # commit しない」と対称。DETAIL フラッシュで開いた空区間を捨てる)。
        if have_open and cur_ml_count > 0:
            segments.append(
                Segment(
                    pokemon=cur_pokemon or [],
                    detail_ts=cur_detail_ts,
                    movelist_start=cur_ml_start,
                    movelist_end=cur_ml_end,
                    move_list_frame_count=cur_ml_count,
                )
            )
        elif have_open and cur_pokemon and not pending_box:
            # 空区間捨てる時、box species (cur_pokemon) は時間的にこの区間と独立
            # した手前のOTHER期間で確定した情報なので次区間に持ち越す
            # (DETAIL→1FのOTHER誤判定→DETAIL の bunny hop で unknown 化するのを防ぐ)。
            # OTHER 期間で別 box へ移動した場合は pending_box が新 box で上書き
            # されているのでこの分岐に入らず誤特定にならない。
            pending_box = cur_pokemon
        have_open = False

    prev_box_thumb: np.ndarray | None = None
    cached_kind: ScreenKind = ScreenKind.OTHER
    prev_header_thumb: np.ndarray | None = None
    prev_classify_kind: ScreenKind | None = None

    for frame in frames:
        if layout is None:
            h, w = frame.image.shape[:2]
            layout = resolve_layout(w, h)

        # ヘッダ領域が前フレームと事実上同じなら分類結果を流用
        # (matchTemplate も OCR フォールバックも両方スキップ).
        # 連続フレームのヘッダは事実上同一なので OTHER 含む全 kind で hit する。
        cur_header_thumb = _region_thumb(frame.image, layout.header)
        kind: ScreenKind | None = None
        if prev_classify_kind is not None and _frames_similar(
            prev_header_thumb, cur_header_thumb
        ):
            kind = prev_classify_kind
        else:
            if classifier.is_ready():
                kind = classifier.classify(frame.image, layout)
            if kind is None:
                kind = classify_screen(frame.image, layout)
                classifier.remember(kind, frame.image, layout)
            prev_classify_kind = kind
        prev_header_thumb = cur_header_thumb
        prev_kind = cached_kind
        cached_kind = kind

        if kind == ScreenKind.DETAIL:
            prev_box_thumb = None
            if prev_kind != ScreenKind.DETAIL:
                close_segment()
                open_segment()
                if pending_box:
                    cur_pokemon = pending_box
                    cur_detail_ts = frame.timestamp
                pending_box = []
        elif kind == ScreenKind.MOVE_LIST:
            prev_box_thumb = None
            if not have_open:
                # DETAIL 前に技一覧が始まる (動画途中開始等) → unknown 区間を開く
                open_segment()
            if cur_ml_start is None:
                cur_ml_start = frame.timestamp
            cur_ml_end = frame.timestamp
            cur_ml_count += 1
        else:
            if float(frame.image.mean()) > 40:
                cur_box_thumb = _region_thumb(frame.image, layout.box_name)
                if not _frames_similar(prev_box_thumb, cur_box_thumb):
                    box_cands = read_box_species(frame.image, layout, master)
                    if box_cands and box_cands[0].score >= _BOX_SPECIES_MIN:
                        pending_box = box_cands
                prev_box_thumb = cur_box_thumb
            else:
                prev_box_thumb = None

    close_segment()
    if config.verbose:
        print(f"index: {len(segments)} segment(s) @ {config.index_fps}fps")
    return segments, classifier


def collect_segment(
    videos: list[Path],
    master: MasterData,
    config: PipelineConfig,
    seg: Segment,
    classifier: TemplateClassifier | None = None,
) -> PokemonResult:
    """フェーズ2: 1セグメントの技一覧範囲を full fps で再走査し技を読む.

    分類は残す (warm な classifier を再利用)。技一覧でないフレーム (境界の
    遷移・ポップアップ等) に read_rows を当てて偽の技を拾わないため。範囲は
    index_fps の粗さを吸収するよう少し外側に広げる (非技一覧フレームは分類で
    弾かれるので広げても安全)。
    """
    if seg.movelist_start is None or seg.movelist_end is None:
        return PokemonResult(
            candidates=seg.pokemon,
            moves=[],
            ambiguous_moves=[],
            raw_move_names=[],
            detail_frame_ts=seg.detail_ts,
            move_list_frame_count=0,
        )
    if classifier is None:
        classifier = TemplateClassifier()

    margin = 2.0 / config.index_fps
    start = max(0.0, seg.movelist_start - margin)
    end = seg.movelist_end + margin
    frames = _prefetch_frames(
        extract_frames(
            videos,
            fps=config.fps,
            out_dir=config.frames_dir,
            start=start,
            end=end,
        )
    )

    layout: Layout | None = None
    votes: Counter[str] = Counter()
    move_list_frames = 0
    prev_slot_thumb: np.ndarray | None = None
    classify_ocr_count = 0
    prev_header_thumb: np.ndarray | None = None
    prev_classify_kind: ScreenKind | None = None
    # 行クロップ単位の OCR キャッシュ. 連続スクロール中はスロット領域全体の
    # 差分スキップが効かないが、行単位では同一クロップが続くためここで削る.
    row_cache = RowTextCache()

    for frame in frames:
        if layout is None:
            h, w = frame.image.shape[:2]
            layout = resolve_layout(w, h)
        # ヘッダ領域が前フレームと事実上同じなら分類結果を流用
        # (matchTemplate も OCR フォールバックも両方スキップ).
        cur_header_thumb = _region_thumb(frame.image, layout.header)
        kind: ScreenKind | None = None
        if prev_classify_kind is not None and _frames_similar(
            prev_header_thumb, cur_header_thumb
        ):
            kind = prev_classify_kind
        else:
            if classifier.is_ready():
                kind = classifier.classify(frame.image, layout)
            if kind is None:
                kind = classify_screen(frame.image, layout)
                classify_ocr_count += 1
                classifier.remember(kind, frame.image, layout)
            prev_classify_kind = kind
        prev_header_thumb = cur_header_thumb
        if kind != ScreenKind.MOVE_LIST:
            prev_slot_thumb = None
            continue
        cur_slot_thumb = _slot_region_thumb(frame.image, layout)
        if _frames_similar(prev_slot_thumb, cur_slot_thumb):
            prev_slot_thumb = cur_slot_thumb
            move_list_frames += 1
            continue
        prev_slot_thumb = cur_slot_thumb
        names = read_rows(
            frame.image,
            layout,
            master,
            accept_threshold=config.accept_threshold,
            ocr_cache=row_cache,
        )
        for n in names:
            votes[n] += 1
        move_list_frames += 1

    voted = [n for n, c in votes.items() if c >= _MIN_VOTES]
    accepted, ambiguous = resolve_moves(
        voted, master, accept_threshold=config.accept_threshold
    )
    if config.verbose:
        print(
            f"  collect: classify_ocr={classify_ocr_count}, "
            f"move_list_frames={move_list_frames}, raw={len(voted)}"
        )
    return PokemonResult(
        candidates=seg.pokemon,
        moves=accepted,
        ambiguous_moves=ambiguous,
        raw_move_names=voted,
        detail_frame_ts=seg.detail_ts,
        move_list_frame_count=move_list_frames,
    )
