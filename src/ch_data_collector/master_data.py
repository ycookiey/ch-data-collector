"""Champions マスタデータ読込.

`data/master/{moves,pokemon}.json` を起動時に読み、メモリ上に保持する。
moves.json は towakey/pokedex (MIT) の全世代 waza_list を union して
scripts/build_moves_from_pokedex.py で生成する (生成手順は README 参照)。
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
    dex_no: int
    name: str
    type1: str
    type2: str | None
    ability1: str
    ability2: str | None
    hidden_ability: str | None
    category: str
    base_species_id: int | None


@dataclass(frozen=True)
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
            "moves.json は towakey/pokedex から生成できる: "
            "uv run python scripts/build_moves_from_pokedex.py <pokedex-src-dir> "
            "(詳細は README 参照)."
        )
    moves_raw = json.loads(moves_path.read_text(encoding="utf-8"))
    pokemon_raw = json.loads(pokemon_path.read_text(encoding="utf-8"))
    moves = tuple(Move(**m) for m in moves_raw)
    pokemon = tuple(Pokemon(**p) for p in pokemon_raw)
    return MasterData(moves=moves, pokemon=pokemon)
