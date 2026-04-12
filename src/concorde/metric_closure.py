from collections import deque
from dataclasses import dataclass

import numpy as np

from src.shared_types import Grid, MOVE_DELTAS, Move, Point


@dataclass(frozen=True, slots=True)
class MetricClosure:
    """All-pairs shortest-path structure over the open cells of a grid.

    `nodes[i]` is the `(row, col)` of node `i`.
    `dist[i][j]` is the number of grid moves in a shortest path from `nodes[i]`
    to `nodes[j]`, or a very large sentinel if `j` is unreachable from `i`.
    `parents[i]` is a dict `{j: (prev_j, move)}` storing a predecessor tree
    rooted at `i`, which lets us reconstruct the full move sequence for any
    `(i, j)` pair via `reconstruct_moves`.
    """

    nodes: tuple[Point, ...]
    index_by_point: dict[Point, int]
    dist: np.ndarray
    parents: tuple[dict[Point, tuple[Point | None, Move | None]], ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def reconstruct_moves(self, i: int, j: int) -> list[Move]:
        if i == j:
            return []
        source = self.nodes[i]
        target = self.nodes[j]
        parent_tree = self.parents[i]
        if target not in parent_tree:
            raise ValueError(
                f"node {target} is unreachable from {source} in metric closure"
            )
        moves: list[Move] = []
        cursor: Point | None = target
        while cursor is not None and cursor != source:
            prev, move = parent_tree[cursor]
            if move is None:
                break
            moves.append(move)
            cursor = prev
        moves.reverse()
        return moves


def build_metric_closure(grid: Grid) -> MetricClosure:
    """Build the metric closure over the open cells of `grid`.

    Runs one BFS per open cell to collect all-pairs distances and predecessor
    trees for move reconstruction. Cost is `O(V * (V + E))` where `V` is the
    number of open cells and `E` is the number of open-cell adjacencies; both
    are bounded by `n^2` for an `n x n` grid, which is fine for our target
    sizes (`n <= ~15`).
    """
    if not grid or not grid[0]:
        return MetricClosure(
            nodes=(),
            index_by_point={},
            dist=np.zeros((0, 0), dtype=np.int64),
            parents=(),
        )

    rows = len(grid)
    cols = len(grid[0])

    nodes: list[Point] = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1
    ]
    n = len(nodes)
    if n == 0:
        return MetricClosure(
            nodes=(),
            index_by_point={},
            dist=np.zeros((0, 0), dtype=np.int64),
            parents=(),
        )

    index_by_point = {point: idx for idx, point in enumerate(nodes)}

    # Use a large sentinel for "unreachable" that is still finite so downstream
    # integer solvers can handle it without overflow concerns.
    unreachable = np.int64(n * (rows * cols + 1))
    dist = np.full((n, n), unreachable, dtype=np.int64)
    np.fill_diagonal(dist, 0)

    ordered_moves: list[Move] = ["u", "d", "l", "r"]
    parents: list[dict[Point, tuple[Point | None, Move | None]]] = []

    for source_idx, source in enumerate(nodes):
        parent_tree: dict[Point, tuple[Point | None, Move | None]] = {
            source: (None, None)
        }
        queue: deque[tuple[Point, int]] = deque([(source, 0)])
        while queue:
            (cr, cc), d = queue.popleft()
            target_idx = index_by_point[(cr, cc)]
            dist[source_idx, target_idx] = d
            for move in ordered_moves:
                dr, dc = MOVE_DELTAS[move]
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr][nc] != 1:
                    continue
                neighbor: Point = (nr, nc)
                if neighbor in parent_tree:
                    continue
                parent_tree[neighbor] = ((cr, cc), move)
                queue.append((neighbor, d + 1))
        parents.append(parent_tree)

    return MetricClosure(
        nodes=tuple(nodes),
        index_by_point=index_by_point,
        dist=dist,
        parents=tuple(parents),
    )
