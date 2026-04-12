import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
from concorde.concorde import Concorde
from concorde.problem import Problem


BIN_DIR = Path(__file__).parent / "bin"


@lru_cache(maxsize=1)
def _default_concorde_binary() -> Path:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        name = "concorde-macos-arm64" if machine in ("arm64", "aarch64") else "concorde-macos-x86_64"
    elif system == "Linux":
        name = "concorde-linux"
    else:
        raise RuntimeError(
            f"no vendored Concorde binary for {system} {machine}; "
            f"install Concorde manually and pass `concorde_exe` explicitly"
        )
    binary = BIN_DIR / name
    if not binary.exists():
        raise RuntimeError(f"vendored Concorde binary not found at {binary}")
    return binary


def solve_symmetric_tsp(
    matrix: np.ndarray, concorde_exe: str | Path | None = None
) -> list[int]:
    """Solve a symmetric integer TSP with Concorde and return the optimal tour.

    The matrix must be square, symmetric, integer, non-negative, and have a
    zero diagonal. The returned tour is a list of node indices of length
    `matrix.shape[0]` in optimal cyclic order (not closed, i.e. the return
    edge is implicit).
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    if n < 2:
        return list(range(n))
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("matrix must be symmetric")
    if (matrix < 0).any():
        raise ValueError("matrix must be non-negative")
    if not np.array_equal(np.diag(matrix), np.zeros(n, dtype=matrix.dtype)):
        raise ValueError("matrix must have a zero diagonal")

    problem = Problem.from_matrix(matrix.astype(np.int64))
    solver = Concorde()
    binary = str(concorde_exe) if concorde_exe is not None else str(
        _default_concorde_binary()
    )
    solution = solver.solve(problem, concorde_exe=binary)
    if not getattr(solution, "found_tour", True):
        raise RuntimeError("Concorde failed to find a tour")
    return [int(node) for node in solution.tour]
