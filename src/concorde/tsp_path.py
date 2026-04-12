from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TSPPathInstance:
    """Augmented symmetric TSP whose optimal tour encodes a fixed-start
    Hamiltonian s-path over the original cities.

    The augmentation adds one dummy node `v_star` to a copy of the metric
    closure. Edge costs are set so that:

    - `D(v_star, start) = 0`
    - `D(v_star, other) = big_M`

    where `big_M` is strictly larger than any possible s-path length. Any
    optimal TSP tour over the augmented graph then spends exactly one zero
    edge `v_star -> start` and one `big_M` edge `other -> v_star`. Removing
    `v_star` from the tour yields the optimal Hamiltonian path over the
    original cities starting at `start`.
    """

    matrix: np.ndarray
    start_index: int
    v_star_index: int
    big_m: int

    @property
    def size(self) -> int:
        return self.matrix.shape[0]


def build_tsp_path_instance(dist: np.ndarray, start_index: int) -> TSPPathInstance:
    """Build a TSPPathInstance from a symmetric, non-negative distance matrix.

    `dist[i][j]` must be `dist[j][i]` and `dist[i][i]` must be 0. `start_index`
    is the index of the fixed start city in the original matrix.
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square matrix")
    n = dist.shape[0]
    if n == 0:
        raise ValueError("dist must have at least one node")
    if not (0 <= start_index < n):
        raise ValueError(f"start_index {start_index} out of range for n={n}")
    if not np.array_equal(dist, dist.T):
        raise ValueError("dist must be symmetric")

    # big_M must strictly exceed the longest possible Hamiltonian s-path over
    # the original cities. An upper bound is `n * max_edge + 1`. We also add
    # some slack so that double-M tours can never beat 0 + M tours numerically.
    max_edge = int(dist.max()) if n > 1 else 0
    big_m = int(max_edge) * int(n) + 1

    augmented = np.zeros((n + 1, n + 1), dtype=np.int64)
    augmented[:n, :n] = dist
    v_star = n
    for i in range(n):
        augmented[v_star, i] = big_m
        augmented[i, v_star] = big_m
    augmented[v_star, start_index] = 0
    augmented[start_index, v_star] = 0

    return TSPPathInstance(
        matrix=augmented,
        start_index=start_index,
        v_star_index=v_star,
        big_m=big_m,
    )


def extract_path_from_tour(
    tour: list[int], instance: TSPPathInstance
) -> list[int]:
    """Given an optimal TSP tour over the augmented instance, return the
    Hamiltonian s-path over the ORIGINAL cities (indices into the non-augmented
    matrix) starting at `instance.start_index`.

    The input tour is a cyclic permutation of `0..size-1`. We rotate so that
    `v_star` is at position 0, then slice off `v_star` and optionally reverse
    so the first real city is `start_index`.
    """
    if not tour:
        raise ValueError("tour is empty")
    size = instance.size
    if sorted(tour) != list(range(size)):
        raise ValueError("tour must be a permutation of 0..size-1")

    v_star = instance.v_star_index
    start = instance.start_index

    v_star_pos = tour.index(v_star)
    rotated = tour[v_star_pos:] + tour[:v_star_pos]
    assert rotated[0] == v_star
    path = rotated[1:]

    # Concorde may return the tour in either orientation. We want the city
    # adjacent to v_star via the zero edge to be at position 0.
    if path and path[0] != start:
        if path[-1] == start:
            path = list(reversed(path))
        else:
            raise ValueError(
                "neither end of the extracted path is the fixed start; "
                "this should not happen if big_m was chosen correctly"
            )
    return path
