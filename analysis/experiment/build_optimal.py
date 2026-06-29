"""Compute provably-optimal covering paths for every Mow-the-Lawn level.

Reads the canonical level definitions from the game repo
(`mow/src/data/levels.json`), reconstructs each level's grid, and solves the
minimum-length covering walk from the player's start corner with the exact
Concorde TSP-based solver (`src.concorde.concorde_solver`).

Coordinate conventions
----------------------
The game (`Game.svelte`) starts the player at {x:0, y:0} with x = horizontal
(left/right) and y = vertical (down = +y). Moves: u=y-1, d=y+1, l=x-1, r=x+1.
The solver indexes `grid[a][b]` with u/d changing the first index and l/r the
second (see `shared_types.MOVE_DELTAS`). Building the grid as `grid[y][x]`
therefore makes the solver's u/d/l/r line up exactly with the game's, and
`find_start` returns (0,0).

Outputs (in this folder):
  - optimal_paths.csv   one row per level: optimal move count, path geometry,
                        stored value, and validation flags.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # analysis/experiment -> mow-data repo root
sys.path.insert(0, str(REPO))

from src.concorde import concorde_solver  # noqa: E402
from src.shared_types import MOVE_DELTAS  # noqa: E402

LEVELS_JSON = REPO.parent / "mow" / "src" / "data" / "levels.json"
OUT_OPTIMAL = HERE / "optimal_paths.csv"

LEVEL_ORDER = ["tutorial", "round1", "round2", "bonus1", "bonus2", "bonus3"]


def build_grid(size: int, obstacles: list[dict]) -> list[list[int]]:
    """grid[y][x] == 1 if mowable, 0 if obstacle (matches solver move axes)."""
    grid = [[1] * size for _ in range(size)]
    for o in obstacles:
        grid[o["y"]][o["x"]] = 0
    return grid


def moves_to_points(start_yx: tuple[int, int], moves: list[str]) -> list[dict]:
    """Expand solver (start, moves) into the game's {x,y} point sequence.

    `start` from the solver is (a, b) = (y, x); the game's start is (0,0).
    """
    y, x = start_yx
    pts = [{"x": x, "y": y}]
    for m in moves:
        dy, dx = MOVE_DELTAS[m]  # first index = y, second = x
        x += dx
        y += dy
        pts.append({"x": x, "y": y})
    return pts


def compute_optima() -> list[dict]:
    """Solve every level exactly with Concorde; return one validated row each.

    Each row carries the optimal move count, the full {x,y} path geometry, the
    value currently stored in the game's levels.json (for cross-checking), and
    validation flags (starts top-left, covers all mowable cells, legal moves).
    """
    levels = json.loads(LEVELS_JSON.read_text())
    rows = []
    for lvl in levels:
        lid = lvl["id"]
        size = lvl["size"]
        obstacles = lvl.get("obstacles", [])
        stored = lvl.get("optimal")
        open_cells = size * size - len(obstacles)

        grid = build_grid(size, obstacles)
        sol = concorde_solver(grid)
        moves = sol["moves"]
        start = sol["start"]  # (y, x)
        n_opt = len(moves)

        pts = moves_to_points(start, moves)
        covered = {(p["x"], p["y"]) for p in pts}
        starts_topleft = start == (0, 0)
        covers_all = len(covered) == open_cells
        # every step must be a unit orthogonal move onto a mowable cell
        obstacle_set = {(o["x"], o["y"]) for o in obstacles}
        legal = all(
            (0 <= p["x"] < size and 0 <= p["y"] < size and (p["x"], p["y"]) not in obstacle_set)
            for p in pts
        )

        rows.append(
            {
                "level": lid,
                "size": size,
                "open_cells": open_cells,
                "optimal_moves": n_opt,
                "stored_optimal": stored,
                "covers_all": covers_all,
                "starts_top_left": starts_topleft,
                "legal_path": legal,
                "path_json": json.dumps(pts, separators=(",", ":")),
                "moves": "".join(moves),
            }
        )
    return rows


def level_optima() -> dict[str, int]:
    """{level_id: optimal move count} — the true-optimal baseline for scoring."""
    return {r["level"]: r["optimal_moves"] for r in compute_optima()}


def main() -> None:
    print(f"Solving levels from {LEVELS_JSON}")
    rows = compute_optima()
    for r in rows:
        status = "ok" if (r["starts_top_left"] and r["covers_all"] and r["legal_path"]) else "CHECK"
        delta = "" if r["stored_optimal"] is None else r["optimal_moves"] - r["stored_optimal"]
        print(
            f"  {r['level']:<8} size={r['size']:<2} open={r['open_cells']:<3} "
            f"optimal={r['optimal_moves']:<4} stored={r['stored_optimal']} delta={delta} "
            f"covered={'all' if r['covers_all'] else 'PARTIAL'} [{status}]"
        )

    df = pd.DataFrame(rows)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df = df.sort_values("level")
    df.to_csv(OUT_OPTIMAL, index=False)
    print(f"\nWrote {OUT_OPTIMAL}  ({len(df)} levels)")


if __name__ == "__main__":
    main()
