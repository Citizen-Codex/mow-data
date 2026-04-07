import random
from collections import deque
from heapq import heappop, heappush
from typing import cast

from src.shared_types import Grid, Move, MoveStrategy, Path, Point, MOVE_DELTAS


def _path_moves_for_strategy(
    grid: Grid,
    start: Point,
    target: Point,
    move_strategy: MoveStrategy,
    visit_counts: dict[Point, int],
) -> list[Move] | None:
    if move_strategy == "shortest":
        return shortest_path_moves(grid, start, target)
    if move_strategy == "least_overlap":
        return least_overlap_moves(grid, start, target, visit_counts)

    raise ValueError(f"Unknown move strategy: {move_strategy}")


def find_start(grid: Grid) -> Point | None:
    if not grid or not grid[0]:
        return None

    rows = len(grid)
    cols = len(grid[0])
    for col_index in range(cols):
        for row_index in range(rows):
            if grid[row_index][col_index] == 1:
                return row_index, col_index
    return None


def shortest_path_moves(grid: Grid, start: Point, target: Point) -> list[Move] | None:
    if start == target:
        return []

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    queue = deque([start])
    parents: dict[Point, tuple[Point, Move]] = {}
    visited = {start}

    ordered_moves: list[Move] = ["u", "d", "l", "r"]

    while queue:
        cx, cy = queue.popleft()
        for move_name in ordered_moves:
            dx, dy = MOVE_DELTAS[move_name]
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 0 or (nx, ny) in visited:
                continue

            parents[(nx, ny)] = ((cx, cy), move_name)
            if (nx, ny) == target:
                path: list[Move] = []
                node = (nx, ny)
                while node != start:
                    previous, step = parents[node]
                    path.append(step)
                    node = previous
                path.reverse()
                return path

            visited.add((nx, ny))
            queue.append((nx, ny))

    return None


def least_overlap_moves(
    grid: Grid,
    start: Point,
    target: Point,
    visit_counts: dict[Point, int],
) -> list[Move] | None:
    if start == target:
        return []

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    ordered_moves: list[Move] = ["u", "d", "l", "r"]
    queue: list[tuple[int, int, Point]] = [(0, 0, start)]
    best_costs: dict[Point, tuple[int, int]] = {start: (0, 0)}
    parents: dict[Point, tuple[Point, Move]] = {}

    while queue:
        overlap_cost, step_cost, (cx, cy) = heappop(queue)
        if (overlap_cost, step_cost) != best_costs[(cx, cy)]:
            continue

        if (cx, cy) == target:
            path: list[Move] = []
            node = target
            while node != start:
                previous, step = parents[node]
                path.append(step)
                node = previous
            path.reverse()
            return path

        for move_name in ordered_moves:
            dx, dy = MOVE_DELTAS[move_name]
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] == 0:
                continue

            overlap_increment = 1 if visit_counts.get((nx, ny), 0) > 0 else 0
            next_cost = (overlap_cost + overlap_increment, step_cost + 1)
            previous_best = best_costs.get((nx, ny))
            if previous_best is not None and previous_best <= next_cost:
                continue

            best_costs[(nx, ny)] = next_cost
            parents[(nx, ny)] = ((cx, cy), move_name)
            heappush(queue, (next_cost[0], next_cost[1], (nx, ny)))

    return None


def snake_solver(
    grid: Grid,
    move_strategy: MoveStrategy = "least_overlap",
    *,
    start_down: bool | None = None,
) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    rows = len(grid)
    cols = len(grid[0])
    start_row = start[0]
    start_col = start[1]
    if start_down is None:
        start_down = random.choice((True, False))

    targets: list[Point] = []
    if start_down:
        for col in range(start_col, cols):
            row_range = (
                range(rows) if (col - start_col) % 2 == 0 else range(rows - 1, -1, -1)
            )
            for row in row_range:
                if grid[row][col] == 1:
                    targets.append((row, col))
    else:
        for row in range(start_row, rows):
            col_range = (
                range(cols) if (row - start_row) % 2 == 0 else range(cols - 1, -1, -1)
            )
            for col in col_range:
                if grid[row][col] == 1:
                    targets.append((row, col))

    remaining_targets = dict.fromkeys(targets)
    remaining_targets.pop(start, None)

    current = start
    moves: list[Move] = []
    visit_counts: dict[Point, int] = {start: 1}

    while remaining_targets:
        target = next(iter(remaining_targets))

        path_moves = _path_moves_for_strategy(
            grid, current, target, move_strategy, visit_counts
        )
        if path_moves is None:
            remaining_targets.pop(target, None)
            continue

        for move in path_moves:
            dx, dy = MOVE_DELTAS[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)
            visit_counts[current] = visit_counts.get(current, 0) + 1
            remaining_targets.pop(current, None)

    return {"start": start, "moves": moves}


def spiral_solver(
    grid: Grid,
    move_strategy: MoveStrategy = "least_overlap",
    *,
    start_down: bool | None = None,
) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    rows = len(grid)
    cols = len(grid[0])

    all_targets: set[Point] = {
        (row, col) for row in range(rows) for col in range(cols) if grid[row][col] == 1
    }
    if start_down is None:
        start_down = random.choice((True, False))

    targets: list[Point] = [start]
    remaining_targets = set(all_targets)
    remaining_targets.discard(start)

    while remaining_targets:
        min_row = min(row for row, _ in remaining_targets)
        max_row = max(row for row, _ in remaining_targets)
        min_col = min(col for _, col in remaining_targets)
        max_col = max(col for _, col in remaining_targets)

        if start_down:
            down_targets = sorted(
                (point for point in remaining_targets if point[1] == min_col),
                key=lambda point: point[0],
            )
            for point in down_targets:
                targets.append(point)
                remaining_targets.remove(point)

            right_targets = sorted(
                (point for point in remaining_targets if point[0] == max_row),
                key=lambda point: point[1],
            )
            for point in right_targets:
                targets.append(point)
                remaining_targets.remove(point)

            up_targets = sorted(
                (point for point in remaining_targets if point[1] == max_col),
                key=lambda point: point[0],
                reverse=True,
            )
            for point in up_targets:
                targets.append(point)
                remaining_targets.remove(point)

            left_targets = sorted(
                (point for point in remaining_targets if point[0] == min_row),
                key=lambda point: point[1],
                reverse=True,
            )
            for point in left_targets:
                targets.append(point)
                remaining_targets.remove(point)
        else:
            right_targets = sorted(
                (point for point in remaining_targets if point[0] == min_row),
                key=lambda point: point[1],
            )
            for point in right_targets:
                targets.append(point)
                remaining_targets.remove(point)

            down_targets = sorted(
                (point for point in remaining_targets if point[1] == max_col),
                key=lambda point: point[0],
            )
            for point in down_targets:
                targets.append(point)
                remaining_targets.remove(point)

            left_targets = sorted(
                (point for point in remaining_targets if point[0] == max_row),
                key=lambda point: point[1],
                reverse=True,
            )
            for point in left_targets:
                targets.append(point)
                remaining_targets.remove(point)

            up_targets = sorted(
                (point for point in remaining_targets if point[1] == min_col),
                key=lambda point: point[0],
                reverse=True,
            )
            for point in up_targets:
                targets.append(point)
                remaining_targets.remove(point)

    remaining_targets = dict.fromkeys(targets)
    remaining_targets.pop(start, None)

    current = start
    moves: list[Move] = []
    visit_counts: dict[Point, int] = {start: 1}

    while remaining_targets:
        target = next(iter(remaining_targets))

        path_moves = _path_moves_for_strategy(
            grid, current, target, move_strategy, visit_counts
        )
        if path_moves is None:
            remaining_targets.pop(target, None)
            continue

        for move in path_moves:
            dx, dy = MOVE_DELTAS[move]
            current = (current[0] + dx, current[1] + dy)
            moves.append(move)
            visit_counts[current] = visit_counts.get(current, 0) + 1
            remaining_targets.pop(current, None)

    return {"start": start, "moves": moves}


def random_walk_solver(grid: Grid, rng: random.Random | None = None) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    if rng is None:
        rng = random.Random()

    rows = len(grid)
    cols = len(grid[0])

    all_mowable = sum(row.count(1) for row in grid)
    visited: set[Point] = {start}
    current = start
    moves: list[Move] = []
    ordered_moves: list[Move] = ["u", "d", "l", "r"]

    while len(visited) < all_mowable:
        neighbors: list[tuple[Point, Move]] = []
        for move_name in ordered_moves:
            dx, dy = MOVE_DELTAS[move_name]
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 1:
                neighbors.append(((nx, ny), move_name))

        if not neighbors:
            break

        unvisited = [(p, m) for p, m in neighbors if p not in visited]
        chosen_point: Point
        chosen_move: Move
        if unvisited:
            chosen_point, chosen_move = cast(tuple[Point, Move], rng.choice(unvisited))
        else:
            chosen_point, chosen_move = cast(tuple[Point, Move], rng.choice(neighbors))

        moves.append(chosen_move)
        current = chosen_point
        visited.add(current)

    return {"start": start, "moves": moves}


from src.memetic_solver import memetic_ga_solver
