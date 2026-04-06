import argparse
import json
import random
from collections import deque

try:
    from src.shared_types import Grid, Point, MOVE_DELTAS
except ModuleNotFoundError:  # Support direct script execution: uv run src/grid.py
    from shared_types import Grid, Point, MOVE_DELTAS


def can_remove(grid: Grid, x: int, y: int) -> bool:
    n = len(grid)
    if n == 0 or not (0 <= x < n and 0 <= y < n):
        return False
    if grid[x][y] == 0:
        return False

    neighbors: list[Point] = []
    for dx, dy in MOVE_DELTAS.values():
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
            neighbors.append((nx, ny))

    if len(neighbors) <= 1:
        return True

    start = neighbors[0]
    visited = {start}
    queue = deque([start])

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < n and 0 <= ny < n):
                continue
            if (nx, ny) == (x, y):
                continue
            if grid[nx][ny] == 0 or (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))

    return all(neighbor in visited for neighbor in neighbors)


def create_random_grid(
    n: int,
    s: int,
    *,
    removed_fraction_range: tuple[float, float] = (0.18, 0.42),
) -> Grid:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    min_removed_fraction, max_removed_fraction = removed_fraction_range
    if not (0.0 <= min_removed_fraction <= max_removed_fraction < 1.0):
        raise ValueError("removed_fraction_range must satisfy 0.0 <= min <= max < 1.0")

    rng = random.Random(s)
    grid = [[1 for _ in range(n)] for _ in range(n)]

    max_removed = n * n * rng.uniform(min_removed_fraction, max_removed_fraction)
    removed = 0
    cluster_count = rng.randint(max(1, n // 3), max(2, n // 2))

    for _ in range(cluster_count):
        if removed >= max_removed:
            break

        x = rng.randrange(n)
        y = rng.randrange(n)
        cluster_size = rng.randint(max(2, n // 2), max(3, n))

        for _ in range(cluster_size):
            if can_remove(grid, x, y):
                grid[x][y] = 0
                removed += 1
                if removed >= max_removed:
                    break

            dx, dy = MOVE_DELTAS[rng.choice(list(MOVE_DELTAS.keys()))]
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                x, y = nx, ny

    return grid


def obstacle_coordinates(grid: Grid) -> list[dict[str, int]]:
    return [
        {"x": x, "y": y}
        for x, row in enumerate(grid)
        for y, cell in enumerate(row)
        if cell == 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and print a random grid")
    parser.add_argument("size", type=int, nargs="?", default=10, help="Grid size")
    parser.add_argument("seed", type=int, nargs="?", default=0, help="Random seed")
    parser.add_argument(
        "--removed-min",
        type=float,
        default=0.18,
        help="Minimum removed-cell fraction",
    )
    parser.add_argument(
        "--removed-max",
        type=float,
        default=0.42,
        help="Maximum removed-cell fraction",
    )
    args = parser.parse_args()

    grid = create_random_grid(
        args.size,
        args.seed,
        removed_fraction_range=(args.removed_min, args.removed_max),
    )
    payload = {
        "size": args.size,
        "seed": args.seed,
        "grid": grid,
        "obstacles": obstacle_coordinates(grid),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
