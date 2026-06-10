"""動画からのフレーム抽出.

OpenCV (cv2.VideoCapture) で各動画をデコードし、指定 fps に間引いて
フレームを yield する。複数動画は時系列連結として扱い、timestamp は
連結後の通算秒を持つ。
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRef:
    index: int          # 抽出時のフレーム連番 (0始まり)
    timestamp: float    # 動画頭からの秒 (近似)
    image: np.ndarray   # BGR (OpenCV標準)


def _safe_fps(raw: float) -> float:
    """CAP_PROP_FPS の値を健全な fps へ正規化する.

    壊れたメタデータでは 0 / 負値 / nan が返ることがある。`x or 30.0`
    だけでは nan (truthy) を取りこぼし、後続の int(round(nan)) が
    ValueError でクラッシュするため、有限かつ正の値のみ採用する。
    """
    if not math.isfinite(raw) or raw <= 0:
        return 30.0
    return raw


def extract_frames(
    video_paths: list[Path],
    fps: float = 5.0,
    out_dir: Path | None = None,
    start: float | None = None,
    end: float | None = None,
) -> Iterator[FrameRef]:
    """動画を fps に間引きしてフレームを yield する.

    out_dir 指定時は PNG 保存もする (デバッグ用)。複数ファイルは連結扱いし、
    timestamp は連結後の通算秒を持つ。

    start/end (連結後の通算秒) を指定するとその区間 [start, end) だけを返す。
    start より前は seek で読み飛ばし (頭からのデコードを省く)、end 到達で打ち切る
    (ポケモン単位の部分再処理を高速化する)。間引きグリッドはフレーム0基準で
    一定なので、範囲を変えても同じ timestamp のフレームが選ばれる。
    """
    # ターゲット fps の値域検証. _safe_fps は動画側 src_fps しか健全化しないため、
    # ここで 0/負/非有限を弾かないと後段の src_fps / fps が ZeroDivisionError 等で落ちる.
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    if start is not None and end is not None and end <= start:
        raise ValueError(f"end must be greater than start (got {start}, {end})")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    cumulative_offset = 0.0
    global_index = 0
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(f"failed to open video: {video_path}")
            src_fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
            step = max(1, int(round(src_fps / fps)))

            # メタデータのフレーム数 (0/-1/VFR で不正なことがある). 健全な時のみ
            # 区間外動画の丸ごとスキップと seek 後のオフセット前進に使う.
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_dur = total / src_fps if total > 0 else None

            # この動画が start より前に完全に終わるなら丸ごと読み飛ばす
            # (duration が分かる時のみ. 不明ならデコードして範囲フィルタに委ねる).
            if (
                start is not None
                and video_dur is not None
                and cumulative_offset + video_dur <= start
            ):
                cumulative_offset += video_dur
                continue

            # start が この動画の途中なら seek でローカル開始フレームへ飛ぶ.
            frame_idx = 0
            if start is not None and start > cumulative_offset:
                local_start = int((start - cumulative_offset) * src_fps)
                if local_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, local_start)
                    # seek は codec により最寄り keyframe に着地するため実位置を
                    # 取り直す (ts のズレ防止). 取得不能なら要求値で代用.
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_idx = pos if pos > 0 else local_start

            stopped_at_end = False
            while True:
                ts = cumulative_offset + frame_idx / src_fps
                if end is not None and ts >= end:
                    stopped_at_end = True
                    break
                # 間引きグリッドはフレーム0基準. start 前の keyframe 余剰分は
                # ts >= start で除外する (seek が start 手前に着地しても正確).
                wanted = frame_idx % step == 0 and (start is None or ts >= start)
                if wanted:
                    ret, frame = cap.read()
                else:
                    # 間引きで捨てるフレームは grab() のみ (デコードは進めるが
                    # retrieve のフレーム構築・コピーを省く). 低 fps 指定時の
                    # 読み飛ばしコストを下げる.
                    ret = cap.grab()
                if not ret:
                    break
                if wanted:
                    if out_dir is not None:
                        cv2.imwrite(
                            str(out_dir / f"f{global_index:05d}.png"), frame
                        )
                    yield FrameRef(
                        index=global_index, timestamp=ts, image=frame
                    )
                    global_index += 1
                frame_idx += 1
            # 通算タイムラインは単調増加. end で打ち切ったら以降の動画も範囲外な
            # ので抽出ごと終了する.
            if stopped_at_end:
                return
            # 通算オフセットは実際にデコードした総フレーム数で進める. コンテナ
            # メタデータ (CAP_PROP_FRAME_COUNT) は 0/-1/VFR で不正なことがあり、
            # それを使うと連結動画の timestamp が巻き戻る. frame_idx は seek 時も
            # 絶対フレーム番号なので末尾で動画長を表す.
            cumulative_offset += frame_idx / src_fps
        finally:
            # consumer 側の例外・早期 break・GeneratorExit で yield 中断しても
            # VideoCapture ハンドルを必ず解放する (try/finally が無いと未解放のまま
            # ジェネレータが放棄され、バッチ処理でハンドル/メモリがリークする).
            cap.release()
