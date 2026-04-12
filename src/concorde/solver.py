import numpy as np

from src.concorde.backend import solve_symmetric_tsp
from src.concorde.metric_closure import build_metric_closure
from src.concorde.tsp_path import build_tsp_path_instance, extract_path_from_tour
from src.shared_types import Grid, Move, Path
from src.solvers import find_start


def concorde_solver(grid: Grid) -> Path:
    """Return a provably optimal covering walk for `grid`.

    Reduces the grid coverage problem to fixed-start symmetric TSP on the
    metric closure (via the dummy `v*` trick), solves it with Concorde, and
    expands the resulting TSP-Path back into grid moves.
    """
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    mc = build_metric_closure(grid)
    start_index = mc.index_by_point[start]

    # Only consider the connected component containing `start`. The metric
    # closure uses a large sentinel for unreachable pairs; feeding those to
    # Concorde would produce nonsense tours that teleport between components.
    # The BFS parent tree rooted at `start` contains exactly the nodes in
    # the connected component of `start`, which is what we need to keep.
    reachable_points = mc.parents[start_index].keys()
    reachable_indices = [mc.index_by_point[point] for point in reachable_points]
    reachable_indices.sort()

    if len(reachable_indices) <= 1:
        return {"start": start, "moves": []}

    reduced_dist = mc.dist[np.ix_(reachable_indices, reachable_indices)]
    reduced_start = reachable_indices.index(start_index)

    instance = build_tsp_path_instance(reduced_dist, start_index=reduced_start)
    tour = solve_symmetric_tsp(instance.matrix)
    reduced_path = extract_path_from_tour(tour, instance)

    node_sequence = [reachable_indices[ri] for ri in reduced_path]

    moves: list[Move] = []
    for prev, nxt in zip(node_sequence, node_sequence[1:]):
        moves.extend(mc.reconstruct_moves(prev, nxt))

    return {"start": start, "moves": moves}
