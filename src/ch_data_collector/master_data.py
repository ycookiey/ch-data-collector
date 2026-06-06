"""Champions マスタデータ読込.

`data/master/{moves,pokemon}.json` を起動時に読み、メモリ上に保持する。
- moves.json は towakey/pokedex (MIT) の全世代 waza_list を union して
  scripts/build_moves_from_pokedex.py で生成する.
- pokemon.json は pokemon-champions-data (CC0) の data/pokemon.json をそのまま
  コピーした文字列リスト. ボックス画面の種族名特定にしか使わないため id と
  name のみを保持する.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Move:
    id: int
    name: str
    type: str
    pp: int


@dataclass(frozen=True)
class Pokemon:
    id: int
    name: str


# eq=False: 同一性ベースの等価/ハッシュにする. learnset の照合インデックスが
# master を WeakKeyDictionary のキーにするため、値ベースだと参照のたびに
# 全 moves/pokemon の再ハッシュが走る (1回数百µs × 毎行毎フレームで効く).
@dataclass(frozen=True, eq=False)
class MasterData:
    moves: tuple[Move, ...]
    pokemon: tuple[Pokemon, ...]


def load_master_data(master_dir: Path | None = None) -> MasterData:
    if master_dir is None:
        # repo_root/data/master をデフォルト解決
        master_dir = Path(__file__).resolve().parents[2] / "data" / "master"
    moves_path = master_dir / "moves.json"
    pokemon_path = master_dir / "pokemon.json"
    if not moves_path.exists() or not pokemon_path.exists():
        raise FileNotFoundError(
            f"master data not found under {master_dir}. "
            "moves.json は towakey/pokedex から、pokemon.json は "
            "pokemon-champions-data/data/pokemon.json から生成できる "
            "(詳細は README 参照)."
        )
    moves_raw = json.loads(moves_path.read_text(encoding="utf-8"))
    pokemon_raw = json.loads(pokemon_path.read_text(encoding="utf-8"))
    moves = tuple(Move(**m) for m in moves_raw)
    pokemon = tuple(Pokemon(id=i, name=n) for i, n in enumerate(pokemon_raw, start=1))
    return MasterData(moves=moves, pokemon=pokemon)
