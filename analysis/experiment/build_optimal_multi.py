"""Enumerate many distinct provably-optimal covering paths per Mow level.

For every canonical level (`mow/src/data/levels.json`) this collects up to
`--target` (default 100) *distinct* optimal covering walks, all of the exact
minimum length found by Concorde. Distinctness is by full move string.

Two independent sources of variation are exploited:

  Phase 1 - Concorde seed variation.
    The TSP-Path instance has (in general) many equally-optimal tours. Passing
    a different `-s <seed>` to Concorde makes it return different optimal tours.
    Each distinct tour is expanded (via the metric-closure BFS predecessor
    tree) into one move string.

  Phase 2 - alternate shortest-path reconstruction.
    A single optimal tour visits a fixed sequence of cities, but the shortest
    path *between* two consecutive cities is frequently not unique. We keep the
    tour's city sequence fixed and enumerate different combinations of shortest
    sub-paths, yielding more distinct optimal move strings of the same length.

Phase 1 runs first; when it stalls (many consecutive seeds add nothing new) we
fall back to Phase 2 to squeeze out the remaining distinct optima. If a level
has fewer than `--target` distinct optima we output all we found and record the
true count.

Coordinate conventions match `build_optimal.py`: grid is `grid[y][x]`, player
starts at (0, 0), moves u=y-1 d=y+1 l=x-1 r=x+1.

Output (this folder): `optimal_solutions_multi.csv`, columns:
  level, size, open_cells, optimal_moves, solution_index, source,
  covers_all, starts_top_left, legal_path, is_optimal, path_json, moves
plus a printed per-level summary of how many distinct optima were found.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # analysis/experiment -> mow-data repo root
sys.path.insert(0, str(REPO))

from concorde.concorde import Concorde  # noqa: E402
from concorde.problem import Problem  # noqa: E402

from src.concorde.backend import _default_concorde_binary  # noqa: E402
from src.concorde.metric_closure import MetricClosure, build_metric_closure  # noqa: E402
from src.concorde.tsp_path import (  # noqa: E402
    TSPPathInstance,
    build_tsp_path_instance,
    extract_path_from_tour,
)
from src.shared_types import Grid, MOVE_DELTAS, Move, Point  # noqa: E402
from src.solvers import find_start  # noqa: E402

from build_optimal import LEVELS_JSON, LEVEL_ORDER, build_grid, moves_to_points  # noqa: E402


# --------------------------------------------------------------------------- #
# Reduced TSP-Path instance shared by both phases.
# --------------------------------------------------------------------------- #

class LevelInstance:
    """Everything needed to solve/expand one level's optima."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self.start: Point | None = find_start(grid)
        self.mc: MetricClosure = build_metric_closure(grid)
        self.open_cells = sum(cell for row in grid for cell in row)

        if self.start is None:
            self.reachable_indices: list[int] = []
            self.reduced_dist = np.zeros((0, 0), dtype=np.int64)
            self.instance: TSPPathInstance | None = None
            return

        start_index = self.mc.index_by_point[self.start]
        reachable_points = self.mc.parents[start_index].keys()
        reachable_indices = sorted(
            self.mc.index_by_point[p] for p in reachable_points
        )
        self.reachable_indices = reachable_indices
        if len(reachable_indices) <= 1:
            self.reduced_dist = np.zeros((0, 0), dtype=np.int64)
            self.instance = None
            return

        self.reduced_dist = self.mc.dist[np.ix_(reachable_indices, reachable_indices)]
        reduced_start = reachable_indices.index(start_index)
        self.instance = build_tsp_path_instance(
            self.reduced_dist, start_index=reduced_start
        )

    def tour_to_node_sequence(self, tour: list[int]) -> list[int]:
        """Map an augmented-instance tour to original metric-closure node ids."""
        reduced_path = extract_path_from_tour(tour, self.instance)
        return [self.reachable_indices[ri] for ri in reduced_path]


def solve_tour_with_seed(matrix: np.ndarray, seed: int) -> list[int]:
    """Solve the symmetric TSP with a specific Concorde random seed."""
    problem = Problem.from_matrix(matrix.astype(np.int64))
    solver = Concorde()
    solution = solver.solve(
        problem,
        concorde_exe=str(_default_concorde_binary()),
        extra_args=["-s", str(seed)],
    )
    if not getattr(solution, "found_tour", True):
        raise RuntimeError("Concorde failed to find a tour")
    return [int(node) for node in solution.tour]


# --------------------------------------------------------------------------- #
# Phase 2: enumerate alternate shortest paths between grid cells.
# --------------------------------------------------------------------------- #

def all_shortest_move_seqs(
    grid: Grid, src: Point, dst: Point, *, cap: int
) -> list[list[Move]]:
    """Return up to `cap` distinct shortest move sequences from src to dst.

    BFS builds the full predecessor DAG (all parents that lie on *some*
    shortest path), then we enumerate move sequences by backtracking, bounded
    by `cap` to avoid combinatorial blow-up.
    """
    if src == dst:
        return [[]]
    rows, cols = len(grid), len(grid[0])
    dist: dict[Point, int] = {src: 0}
    preds: dict[Point, list[tuple[Point, Move]]] = defaultdict(list)
    queue: deque[Point] = deque([src])
    while queue:
        cur = queue.popleft()
        for move, (dr, dc) in MOVE_DELTAS.items():
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] != 1:
                continue
            nb = (nr, nc)
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                preds[nb].append((cur, move))
                queue.append(nb)
            elif dist[nb] == dist[cur] + 1:
                preds[nb].append((cur, move))

    if dst not in dist:
        return []

    results: list[list[Move]] = []

    def backtrack(node: Point, acc: list[Move]) -> None:
        if len(results) >= cap:
            return
        if node == src:
            results.append(list(reversed(acc)))
            return
        for parent, move in preds[node]:
            backtrack(parent, acc + [move])

    backtrack(dst, [])
    return results


def expand_alternates(
    inst: LevelInstance,
    node_sequence: list[int],
    *,
    seen: set[str],
    target: int,
    per_edge_cap: int,
) -> list[str]:
    """Given one optimal city sequence, yield distinct optimal move strings by
    swapping in alternate shortest sub-paths between consecutive cities.

    We fix the city order (which guarantees optimal total length) and take the
    cartesian product of alternate shortest paths per edge, capped so it stays
    tractable. Returns newly-found move strings (also added to `seen`).
    """
    mc = inst.mc
    per_edge_options: list[list[list[Move]]] = []
    for prev, nxt in zip(node_sequence, node_sequence[1:]):
        src = mc.nodes[prev]
        dst = mc.nodes[nxt]
        options = all_shortest_move_seqs(inst.grid, src, dst, cap=per_edge_cap)
        if not options:
            options = [mc.reconstruct_moves(prev, nxt)]
        per_edge_options.append(options)

    found: list[str] = []
    # itertools.product is lazy; we bail as soon as we hit the target.
    for combo in itertools.product(*per_edge_options):
        moves: list[Move] = []
        for seg in combo:
            moves.extend(seg)
        move_str = "".join(moves)
        if move_str not in seen:
            seen.add(move_str)
            found.append(move_str)
            if len(seen) >= target:
                break
    return found


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #

def validate(
    grid: Grid,
    size: int,
    start_yx: tuple[int, int],
    move_str: str,
    open_cells: int,
    optimal_moves: int,
) -> dict:
    moves = list(move_str)
    pts = moves_to_points(start_yx, moves)
    covered = {(p["x"], p["y"]) for p in pts}
    obstacle_set = {
        (x, y) for y in range(size) for x in range(size) if grid[y][x] == 0
    }
    legal = all(
        0 <= p["x"] < size
        and 0 <= p["y"] < size
        and (p["x"], p["y"]) not in obstacle_set
        for p in pts
    )
    return {
        "path_json": json.dumps(pts, separators=(",", ":")),
        "covers_all": len(covered) == open_cells,
        "starts_top_left": start_yx == (0, 0),
        "legal_path": legal,
        "is_optimal": len(moves) == optimal_moves,
    }


# --------------------------------------------------------------------------- #
# Per-level driver.
# --------------------------------------------------------------------------- #

def generate_for_level(
    level: dict,
    *,
    target: int,
    max_seeds: int,
    stall_limit: int,
    per_edge_cap: int,
    verbose: bool = True,
) -> list[dict]:
    lid = level["id"]
    size = level["size"]
    obstacles = level.get("obstacles", [])
    grid = build_grid(size, obstacles)
    open_cells = size * size - len(obstacles)

    inst = LevelInstance(grid)
    if inst.instance is None or inst.start is None:
        return []

    start_yx = inst.start  # (y, x); (0, 0) for these levels

    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []  # (move_str, source)

    # Keep the node sequence of every distinct tour we discover, for Phase 2.
    distinct_node_sequences: list[list[int]] = []
    node_seq_keys: set[tuple[int, ...]] = set()

    optimal_moves = -1

    # ---- Phase 1: seed variation ---------------------------------------- #
    consecutive_stall = 0
    for seed in range(max_seeds):
        if len(seen) >= target:
            break
        tour = solve_tour_with_seed(inst.instance.matrix, seed)
        node_seq = inst.tour_to_node_sequence(tour)
        moves: list[Move] = []
        for prev, nxt in zip(node_seq, node_seq[1:]):
            moves.extend(inst.mc.reconstruct_moves(prev, nxt))
        move_str = "".join(moves)

        if optimal_moves == -1:
            optimal_moves = len(moves)

        key = tuple(node_seq)
        if key not in node_seq_keys:
            node_seq_keys.add(key)
            distinct_node_sequences.append(node_seq)

        if move_str not in seen:
            seen.add(move_str)
            ordered.append((move_str, "seed"))
            consecutive_stall = 0
        else:
            consecutive_stall += 1
            if consecutive_stall >= stall_limit:
                if verbose:
                    print(
                        f"    [{lid}] phase-1 stalled after {seed + 1} seeds "
                        f"({len(seen)} distinct); switching to phase-2"
                    )
                break

    # ---- Phase 2: alternate shortest-path reconstruction ----------------- #
    if len(seen) < target and distinct_node_sequences:
        for node_seq in distinct_node_sequences:
            if len(seen) >= target:
                break
            new_strs = expand_alternates(
                inst,
                node_seq,
                seen=seen,
                target=target,
                per_edge_cap=per_edge_cap,
            )
            for s in new_strs:
                ordered.append((s, "alt-path"))

    # ---- Validate + assemble rows ---------------------------------------- #
    rows: list[dict] = []
    for idx, (move_str, source) in enumerate(ordered):
        v = validate(grid, size, start_yx, move_str, open_cells, optimal_moves)
        rows.append(
            {
                "level": lid,
                "size": size,
                "open_cells": open_cells,
                "optimal_moves": optimal_moves,
                "solution_index": idx,
                "source": source,
                "covers_all": v["covers_all"],
                "starts_top_left": v["starts_top_left"],
                "legal_path": v["legal_path"],
                "is_optimal": v["is_optimal"],
                "path_json": v["path_json"],
                "moves": move_str,
            }
        )

    if verbose:
        n_seed = sum(1 for _, s in ordered if s == "seed")
        n_alt = len(ordered) - n_seed
        print(
            f"  {lid:<8} size={size:<2} open={open_cells:<3} "
            f"optimal={optimal_moves:<4} distinct={len(rows):<4} "
            f"(seed={n_seed}, alt-path={n_alt})"
            f"{'' if len(rows) >= target else '  [< target]'}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate distinct optimal covering paths per level"
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=2000,
        help="upper bound on Concorde seeds tried in phase 1",
    )
    parser.add_argument(
        "--stall-limit",
        type=int,
        default=60,
        help="consecutive duplicate seeds before switching to phase 2",
    )
    parser.add_argument(
        "--per-edge-cap",
        type=int,
        default=6,
        help="max alternate shortest sub-paths enumerated per tour edge",
    )
    parser.add_argument(
        "--out",
        default=str(HERE / "optimal_solutions_multi.csv"),
    )
    args = parser.parse_args()

    levels = json.loads(Path(LEVELS_JSON).read_text())
    print(f"Enumerating up to {args.target} distinct optima per level")
    print(f"Levels from {LEVELS_JSON}\n")

    all_rows: list[dict] = []
    for level in levels:
        rows = generate_for_level(
            level,
            target=args.target,
            max_seeds=args.max_seeds,
            stall_limit=args.stall_limit,
            per_edge_cap=args.per_edge_cap,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df = df.sort_values(["level", "solution_index"])
    df.to_csv(args.out, index=False)

    print(f"\nWrote {args.out}  ({len(df)} rows across {df['level'].nunique()} levels)")

    # Integrity summary.
    bad = df[~(df["covers_all"] & df["starts_top_left"] & df["legal_path"] & df["is_optimal"])]
    if len(bad):
        print(f"WARNING: {len(bad)} rows failed validation:")
        print(bad[["level", "solution_index", "covers_all", "legal_path", "is_optimal"]].to_string(index=False))
    else:
        print("All solutions validated (cover all, legal, start top-left, optimal length).")

    print("\nDistinct optima found per level:")
    counts = df.groupby("level", observed=True).size()
    for lid in LEVEL_ORDER:
        if lid in counts.index:
            n = int(counts[lid])
            flag = "" if n >= args.target else f"  (only {n} exist / found)"
            print(f"  {lid:<8} {n}{flag}")


if __name__ == "__main__":
    main()
