import random
from collections import deque
from dataclasses import dataclass
from typing import cast

from src.shared_types import Grid, Move, MoveStrategy, Path, Point, MOVE_DELTAS


@dataclass(frozen=True, slots=True)
class _BranchFeature:
    move: Move
    next_point: Point
    corridor_length: int
    bridge_size: int
    is_bridge: bool
    leads_to_dead_end: bool
    terminal_boundary_score: int
    terminal_density: int


@dataclass(frozen=True, slots=True)
class _MemeticLayout:
    open_points: tuple[Point, ...]
    open_point_set: frozenset[Point]
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]]
    degrees: dict[Point, int]
    decision_points: tuple[Point, ...]
    boundary_scores: dict[Point, int]
    local_densities: dict[Point, int]
    max_density: int
    branch_features: dict[tuple[Point, Move], _BranchFeature]


@dataclass(slots=True)
class _MemeticGenome:
    junction_orders: dict[Point, tuple[Move, ...]]
    local_weights: tuple[float, ...]
    target_weights: tuple[float, ...]


@dataclass(slots=True)
class _MemeticCandidate:
    genome: _MemeticGenome
    path: Path
    score: float
    move_count: int
    overlap_count: int
    covered_cells: int


@dataclass(frozen=True, slots=True)
class _MemeticTemplate:
    name: str
    move_order: tuple[Move, ...]
    local_weights: tuple[float, ...]
    target_weights: tuple[float, ...]


MEMETIC_TEMPLATES = (
    _MemeticTemplate(
        name="leaf_sweep_down",
        move_order=("d", "r", "u", "l"),
        local_weights=(1.8, 1.9, 0.7, 1.4, 0.9, 0.8, 0.7, 1.2),
        target_weights=(2.0, 1.5, 0.8, 1.0, -0.2, 0.7),
    ),
    _MemeticTemplate(
        name="leaf_sweep_right",
        move_order=("r", "d", "l", "u"),
        local_weights=(1.8, 1.9, 0.7, 1.4, 0.9, 0.8, 0.7, 1.2),
        target_weights=(2.0, 1.5, 0.8, 1.0, -0.2, 0.7),
    ),
    _MemeticTemplate(
        name="boundary_sweep",
        move_order=("r", "d", "u", "l"),
        local_weights=(1.6, 0.8, 0.2, 0.7, 0.4, 1.3, 1.4, 0.5),
        target_weights=(2.3, 0.2, 1.2, 0.4, 0.9, 0.6),
    ),
    _MemeticTemplate(
        name="corridor_first",
        move_order=("d", "r", "l", "u"),
        local_weights=(1.7, 1.0, 0.4, 0.8, 1.4, 0.6, 1.0, 0.9),
        target_weights=(2.1, 0.8, 0.6, 0.8, 0.3, 0.8),
    ),
    _MemeticTemplate(
        name="chamber_first",
        move_order=("r", "d", "l", "u"),
        local_weights=(1.5, 0.5, -0.2, 0.2, 0.6, 0.1, 0.8, -1.0),
        target_weights=(1.8, 0.3, 0.5, -0.8, 1.5, 0.5),
    ),
)


def _step_point(point: Point, move: Move) -> Point:
    d_row, d_col = MOVE_DELTAS[move]
    return point[0] + d_row, point[1] + d_col


def _iter_open_points(grid: Grid) -> tuple[Point, ...]:
    return tuple(
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, cell in enumerate(row)
        if cell == 1
    )


def _build_neighbor_map(
    open_points: tuple[Point, ...],
) -> dict[Point, tuple[tuple[Move, Point], ...]]:
    open_point_set = frozenset(open_points)
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]] = {}
    for point in open_points:
        row, col = point
        neighbors_by_point[point] = tuple(
            (move, (row + d_row, col + d_col))
            for move, (d_row, d_col) in MOVE_DELTAS.items()
            if (row + d_row, col + d_col) in open_point_set
        )
    return neighbors_by_point


def _boundary_score(point: Point, grid: Grid) -> int:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    row, col = point
    score = 0
    for d_row, d_col in MOVE_DELTAS.values():
        next_row = row + d_row
        next_col = col + d_col
        if (
            not (0 <= next_row < rows and 0 <= next_col < cols)
            or grid[next_row][next_col] == 0
        ):
            score += 1
    return score


def _local_density(
    point: Point, open_point_set: frozenset[Point], *, radius: int = 2
) -> int:
    row, col = point
    count = 0
    for d_row in range(-radius, radius + 1):
        remaining = radius - abs(d_row)
        for d_col in range(-remaining, remaining + 1):
            if (row + d_row, col + d_col) in open_point_set:
                count += 1
    return count


def _find_bridge_edges(
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]],
) -> set[frozenset[Point]]:
    discovery: dict[Point, int] = {}
    low_link: dict[Point, int] = {}
    bridges: set[frozenset[Point]] = set()
    time = 0

    def dfs(point: Point, parent: Point | None) -> None:
        nonlocal time
        discovery[point] = time
        low_link[point] = time
        time += 1

        for _, neighbor in neighbors_by_point[point]:
            if neighbor == parent:
                continue
            if neighbor not in discovery:
                dfs(neighbor, point)
                low_link[point] = min(low_link[point], low_link[neighbor])
                if low_link[neighbor] > discovery[point]:
                    bridges.add(frozenset((point, neighbor)))
                continue
            low_link[point] = min(low_link[point], discovery[neighbor])

    for point in neighbors_by_point:
        if point not in discovery:
            dfs(point, None)

    return bridges


def _bridge_component_size(
    start: Point,
    blocked_edge: frozenset[Point],
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]],
) -> int:
    visited = {start}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for _, neighbor in neighbors_by_point[point]:
            if frozenset((point, neighbor)) == blocked_edge or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return len(visited)


def _directed_bridge_sizes(
    open_points: tuple[Point, ...],
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]],
) -> dict[tuple[Point, Point], int]:
    bridges = _find_bridge_edges(neighbors_by_point)
    directed_sizes: dict[tuple[Point, Point], int] = {}
    total_points = len(open_points)
    for bridge in bridges:
        left, right = tuple(bridge)
        right_size = _bridge_component_size(right, bridge, neighbors_by_point)
        directed_sizes[(left, right)] = right_size
        directed_sizes[(right, left)] = total_points - right_size
    return directed_sizes


def _build_branch_features(
    point: Point,
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]],
    degrees: dict[Point, int],
    boundary_scores: dict[Point, int],
    local_densities: dict[Point, int],
    directed_bridge_sizes: dict[tuple[Point, Point], int],
) -> dict[Move, _BranchFeature]:
    branch_features: dict[Move, _BranchFeature] = {}
    for move, next_point in neighbors_by_point[point]:
        previous = point
        current = next_point
        corridor_length = 1
        while degrees[current] == 2:
            onward = [
                candidate
                for _, candidate in neighbors_by_point[current]
                if candidate != previous
            ]
            if not onward:
                break
            previous, current = current, onward[0]
            corridor_length += 1

        bridge_size = directed_bridge_sizes.get((point, next_point), 0)
        branch_features[move] = _BranchFeature(
            move=move,
            next_point=next_point,
            corridor_length=corridor_length,
            bridge_size=bridge_size,
            is_bridge=bridge_size > 0,
            leads_to_dead_end=degrees[current] == 1,
            terminal_boundary_score=boundary_scores[current],
            terminal_density=local_densities[current],
        )
    return branch_features


def _build_memetic_layout(grid: Grid, start: Point | None) -> _MemeticLayout:
    open_points = _iter_open_points(grid)
    open_point_set = frozenset(open_points)
    neighbors_by_point = _build_neighbor_map(open_points)
    degrees = {point: len(neighbors) for point, neighbors in neighbors_by_point.items()}
    decision_points = tuple(
        point for point in open_points if point == start or degrees[point] != 2
    )
    boundary_scores = {point: _boundary_score(point, grid) for point in open_points}
    local_densities = {
        point: _local_density(point, open_point_set) for point in open_points
    }
    max_density = max(local_densities.values(), default=1)
    directed_bridge_sizes = _directed_bridge_sizes(open_points, neighbors_by_point)
    branch_features: dict[tuple[Point, Move], _BranchFeature] = {}
    for point in open_points:
        for move, feature in _build_branch_features(
            point,
            neighbors_by_point,
            degrees,
            boundary_scores,
            local_densities,
            directed_bridge_sizes,
        ).items():
            branch_features[(point, move)] = feature

    return _MemeticLayout(
        open_points=open_points,
        open_point_set=open_point_set,
        neighbors_by_point=neighbors_by_point,
        degrees=degrees,
        decision_points=decision_points,
        boundary_scores=boundary_scores,
        local_densities=local_densities,
        max_density=max_density,
        branch_features=branch_features,
    )


def _path_signature(path: Path) -> str:
    start = path["start"]
    prefix = "none" if start is None else f"{start[0]}:{start[1]}"
    return prefix + "|" + "".join(path["moves"])


def _evaluate_path(grid: Grid, path: Path) -> tuple[float, int, int, int]:
    open_points = _iter_open_points(grid)
    open_point_set = frozenset(open_points)
    open_count = len(open_points)
    start = path["start"]
    if open_count == 0:
        return 0.0, 0, 0, 0
    if start is None or start not in open_point_set:
        penalty = open_count * (open_count + 4)
        return float(penalty), len(path["moves"]), penalty, 0

    current = start
    visited = {start}
    visit_counts: dict[Point, int] = {start: 1}
    overlaps = 0

    for move in path["moves"]:
        current = _step_point(current, move)
        if current not in open_point_set:
            penalty = open_count * (open_count + 4)
            return (
                float(len(path["moves"]) + penalty),
                len(path["moves"]),
                penalty,
                len(visited),
            )
        visit_counts[current] = visit_counts.get(current, 0) + 1
        if visit_counts[current] > 1:
            overlaps += 1
        visited.add(current)

    missing = open_count - len(visited)
    score = float(len(path["moves"])) + overlaps * 0.05
    if missing > 0:
        score += missing * (open_count + 4)
    return score, len(path["moves"]), overlaps, len(visited)


def _distance_map(
    start: Point,
    neighbors_by_point: dict[Point, tuple[tuple[Move, Point], ...]],
) -> dict[Point, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        next_distance = distances[point] + 1
        for _, neighbor in neighbors_by_point[point]:
            if neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _frontier_points(layout: _MemeticLayout, visited: set[Point]) -> list[Point]:
    frontier: list[Point] = []
    for point in layout.open_points:
        if point in visited:
            continue
        if any(neighbor in visited for _, neighbor in layout.neighbors_by_point[point]):
            frontier.append(point)
    return frontier


def _static_branch_score(
    feature: _BranchFeature,
    layout: _MemeticLayout,
    template: _MemeticTemplate,
) -> float:
    bridge_ratio = (
        feature.bridge_size / max(1, len(layout.open_points))
        if feature.is_bridge
        else 0.0
    )
    small_bridge_bonus = (1.0 - bridge_ratio) if feature.is_bridge else 0.0
    corridor_ratio = feature.corridor_length / max(1, len(layout.open_points))
    boundary_ratio = feature.terminal_boundary_score / 4.0
    sparse_ratio = 1.0 - (feature.terminal_density / max(1, layout.max_density))
    feature_vector = (
        float(feature.leads_to_dead_end),
        float(feature.is_bridge),
        small_bridge_bonus,
        corridor_ratio,
        boundary_ratio,
        sparse_ratio,
    )
    weights = template.local_weights[1:7]
    return sum(
        weight * value for weight, value in zip(weights, feature_vector, strict=True)
    )


def _template_order_for_point(
    point: Point,
    layout: _MemeticLayout,
    template: _MemeticTemplate,
) -> tuple[Move, ...]:
    move_order_index = {move: index for index, move in enumerate(template.move_order)}
    moves = [move for move, _ in layout.neighbors_by_point[point]]
    moves.sort(
        key=lambda move: (
            -_static_branch_score(
                layout.branch_features[(point, move)], layout, template
            ),
            move_order_index[move],
        )
    )
    return tuple(moves)


def _jitter_weights(
    weights: tuple[float, ...],
    rng: random.Random,
    *,
    scale: float,
) -> tuple[float, ...]:
    return tuple(
        max(-3.0, min(3.0, weight + rng.uniform(-scale, scale))) for weight in weights
    )


def _genome_from_template(
    layout: _MemeticLayout,
    template: _MemeticTemplate,
    rng: random.Random,
    *,
    jitter_scale: float = 0.18,
) -> _MemeticGenome:
    junction_orders = {
        point: _template_order_for_point(point, layout, template)
        for point in layout.decision_points
    }
    return _MemeticGenome(
        junction_orders=junction_orders,
        local_weights=_jitter_weights(template.local_weights, rng, scale=jitter_scale),
        target_weights=_jitter_weights(
            template.target_weights, rng, scale=jitter_scale
        ),
    )


def _genome_from_path_seed(
    path: Path,
    layout: _MemeticLayout,
    template: _MemeticTemplate,
    rng: random.Random,
) -> _MemeticGenome:
    genome = _genome_from_template(layout, template, rng, jitter_scale=0.1)
    first_seen_moves: dict[Point, list[Move]] = {}
    current = path["start"]
    if current is None:
        return genome

    for move in path["moves"]:
        if (
            current in genome.junction_orders
            and move in genome.junction_orders[current]
        ):
            first_seen_moves.setdefault(current, [])
            if move not in first_seen_moves[current]:
                first_seen_moves[current].append(move)
        current = _step_point(current, move)

    for point, base_order in genome.junction_orders.items():
        used_moves = first_seen_moves.get(point, [])
        merged = [move for move in used_moves if move in base_order]
        merged.extend(move for move in base_order if move not in merged)
        genome.junction_orders[point] = tuple(merged)
    return genome


def _candidate_sort_key(candidate: _MemeticCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.score,
        candidate.move_count,
        candidate.overlap_count,
        -candidate.covered_cells,
    )


def _select_neighbor_move(
    current: Point,
    previous_move: Move | None,
    visited: set[Point],
    layout: _MemeticLayout,
    genome: _MemeticGenome,
) -> tuple[Move, Point] | None:
    available = [
        (move, neighbor)
        for move, neighbor in layout.neighbors_by_point[current]
        if neighbor not in visited
    ]
    if not available:
        return None
    if len(available) == 1:
        return available[0]

    move_ranks = {
        move: index
        for index, move in enumerate(genome.junction_orders.get(current, ()))
    }
    rank_denominator = max(1, len(available) - 1)
    best_choice: tuple[float, Move, Point] | None = None
    total_open = max(1, len(layout.open_points))
    for move, neighbor in available:
        feature = layout.branch_features[(current, move)]
        bridge_ratio = feature.bridge_size / total_open if feature.is_bridge else 0.0
        small_bridge_bonus = (1.0 - bridge_ratio) if feature.is_bridge else 0.0
        corridor_ratio = feature.corridor_length / total_open
        boundary_ratio = feature.terminal_boundary_score / 4.0
        straight_bonus = 1.0 if previous_move == move else 0.0
        sparse_ratio = 1.0 - (feature.terminal_density / max(1, layout.max_density))
        rank = move_ranks.get(move, len(available) - 1)
        order_score = 1.0 - (rank / rank_denominator)
        feature_vector = (
            order_score,
            float(feature.leads_to_dead_end),
            float(feature.is_bridge),
            small_bridge_bonus,
            corridor_ratio,
            boundary_ratio,
            straight_bonus,
            sparse_ratio,
        )
        score = sum(
            weight * value
            for weight, value in zip(genome.local_weights, feature_vector, strict=True)
        )
        choice = (score, move, neighbor)
        if best_choice is None or choice > best_choice:
            best_choice = choice
    if best_choice is None:
        return None
    return best_choice[1], best_choice[2]


def _select_frontier_target(
    current: Point,
    visited: set[Point],
    layout: _MemeticLayout,
    genome: _MemeticGenome,
) -> Point | None:
    frontier = _frontier_points(layout, visited)
    if not frontier:
        return None

    distances = _distance_map(current, layout.neighbors_by_point)
    total_open = max(1, len(layout.open_points))
    best_choice: tuple[float, Point] | None = None
    for point in frontier:
        distance = distances.get(point)
        if distance is None:
            continue
        degree = max(1, layout.degrees[point])
        closeness = 1.0 - (distance / total_open)
        dead_end_bonus = 1.0 if layout.degrees[point] == 1 else 0.0
        boundary_ratio = layout.boundary_scores[point] / 4.0
        sparse_ratio = 1.0 - (
            layout.local_densities[point] / max(1, layout.max_density)
        )
        forward_ratio = (
            sum(
                1
                for _, neighbor in layout.neighbors_by_point[point]
                if neighbor not in visited
            )
            / degree
        )
        visited_contact_ratio = (
            sum(
                1
                for _, neighbor in layout.neighbors_by_point[point]
                if neighbor in visited
            )
            / degree
        )
        feature_vector = (
            closeness,
            dead_end_bonus,
            boundary_ratio,
            sparse_ratio,
            forward_ratio,
            visited_contact_ratio,
        )
        score = sum(
            weight * value
            for weight, value in zip(genome.target_weights, feature_vector, strict=True)
        )
        choice = (score, point)
        if best_choice is None or choice > best_choice:
            best_choice = choice
    return None if best_choice is None else best_choice[1]


def _decode_memetic_path(
    grid: Grid,
    layout: _MemeticLayout,
    genome: _MemeticGenome,
    start: Point | None,
    *,
    least_overlap_path,
    shortest_path,
) -> Path:
    if start is None:
        return {"start": None, "moves": []}

    current = start
    previous_move: Move | None = None
    visited = {start}
    visit_counts: dict[Point, int] = {start: 1}
    moves: list[Move] = []
    target_coverage = len(layout.open_points)

    while len(visited) < target_coverage:
        next_step = _select_neighbor_move(
            current, previous_move, visited, layout, genome
        )
        if next_step is not None:
            move, next_point = next_step
            current = next_point
            previous_move = move
            moves.append(move)
            visit_counts[current] = visit_counts.get(current, 0) + 1
            visited.add(current)
            continue

        target = _select_frontier_target(current, visited, layout, genome)
        if target is None:
            break
        connector = least_overlap_path(grid, current, target, visit_counts)
        if connector is None:
            connector = shortest_path(grid, current, target)
        if connector is None:
            break

        for move in connector:
            current = _step_point(current, move)
            previous_move = move
            moves.append(move)
            visit_counts[current] = visit_counts.get(current, 0) + 1
            visited.add(current)

    return {"start": start, "moves": moves}


def _evaluate_genome(
    grid: Grid,
    layout: _MemeticLayout,
    genome: _MemeticGenome,
    start: Point | None,
    *,
    least_overlap_path,
    shortest_path,
) -> _MemeticCandidate:
    path = _decode_memetic_path(
        grid,
        layout,
        genome,
        start,
        least_overlap_path=least_overlap_path,
        shortest_path=shortest_path,
    )
    score, move_count, overlap_count, covered_cells = _evaluate_path(grid, path)
    return _MemeticCandidate(
        genome=genome,
        path=path,
        score=score,
        move_count=move_count,
        overlap_count=overlap_count,
        covered_cells=covered_cells,
    )


def _crossover_orders(
    left: tuple[Move, ...],
    right: tuple[Move, ...],
    rng: random.Random,
) -> tuple[Move, ...]:
    if len(left) <= 1:
        return left
    prefix_length = rng.randrange(0, len(left))
    merged: list[Move] = []
    for move in (*left[:prefix_length], *right, *left):
        if move not in merged:
            merged.append(move)
    return tuple(merged)


def _crossover_genomes(
    left: _MemeticGenome,
    right: _MemeticGenome,
    layout: _MemeticLayout,
    rng: random.Random,
) -> _MemeticGenome:
    junction_orders = {
        point: _crossover_orders(
            left.junction_orders[point], right.junction_orders[point], rng
        )
        for point in layout.decision_points
    }
    local_weights = tuple(
        ((left_value + right_value) / 2.0)
        if rng.random() < 0.5
        else rng.choice((left_value, right_value))
        for left_value, right_value in zip(
            left.local_weights, right.local_weights, strict=True
        )
    )
    target_weights = tuple(
        ((left_value + right_value) / 2.0)
        if rng.random() < 0.5
        else rng.choice((left_value, right_value))
        for left_value, right_value in zip(
            left.target_weights, right.target_weights, strict=True
        )
    )
    return _MemeticGenome(
        junction_orders=junction_orders,
        local_weights=local_weights,
        target_weights=target_weights,
    )


def _mutate_genome(
    genome: _MemeticGenome,
    layout: _MemeticLayout,
    rng: random.Random,
    *,
    order_mutations: int,
    weight_scale: float,
) -> _MemeticGenome:
    junction_orders = dict(genome.junction_orders)
    mutable_points = [
        point for point in layout.decision_points if len(junction_orders[point]) > 1
    ]
    for _ in range(order_mutations):
        if not mutable_points:
            break
        point = rng.choice(mutable_points)
        order = list(junction_orders[point])
        left_index, right_index = sorted(rng.sample(range(len(order)), 2))
        if rng.random() < 0.5:
            order[left_index], order[right_index] = (
                order[right_index],
                order[left_index],
            )
        else:
            move = order.pop(right_index)
            order.insert(left_index, move)
        junction_orders[point] = tuple(order)

    local_weights = list(genome.local_weights)
    for index in range(len(local_weights)):
        if rng.random() < 0.4:
            local_weights[index] = max(
                -3.0,
                min(
                    3.0, local_weights[index] + rng.uniform(-weight_scale, weight_scale)
                ),
            )

    target_weights = list(genome.target_weights)
    for index in range(len(target_weights)):
        if rng.random() < 0.4:
            target_weights[index] = max(
                -3.0,
                min(
                    3.0,
                    target_weights[index] + rng.uniform(-weight_scale, weight_scale),
                ),
            )

    return _MemeticGenome(
        junction_orders=junction_orders,
        local_weights=tuple(local_weights),
        target_weights=tuple(target_weights),
    )


def _tournament_select(
    population: list[_MemeticCandidate],
    rng: random.Random,
    *,
    size: int = 3,
) -> _MemeticCandidate:
    tournament_size = min(size, len(population))
    sampled = rng.sample(population, tournament_size)
    return min(sampled, key=_candidate_sort_key)


def _local_improve_candidate(
    candidate: _MemeticCandidate,
    grid: Grid,
    layout: _MemeticLayout,
    start: Point | None,
    rng: random.Random,
    *,
    steps: int,
    least_overlap_path,
    shortest_path,
) -> _MemeticCandidate:
    best = candidate
    for _ in range(steps):
        mutated = _mutate_genome(
            best.genome, layout, rng, order_mutations=1, weight_scale=0.22
        )
        trial = _evaluate_genome(
            grid,
            layout,
            mutated,
            start,
            least_overlap_path=least_overlap_path,
            shortest_path=shortest_path,
        )
        if _candidate_sort_key(trial) < _candidate_sort_key(best):
            best = trial
    return best


def memetic_ga_solver(grid: Grid, rng: random.Random | None = None) -> Path:
    if not grid or not grid[0]:
        return {"start": None, "moves": []}

    from src.solvers import (
        find_start,
        least_overlap_moves,
        shortest_path_moves,
        snake_solver,
        spiral_solver,
    )

    start = find_start(grid)
    if start is None:
        return {"start": None, "moves": []}

    if rng is None:
        rng = random.Random()

    layout = _build_memetic_layout(grid, start)
    open_count = len(layout.open_points)
    if open_count <= 1:
        return {"start": start, "moves": []}

    heuristic_paths: list[Path] = []
    for move_strategy in ("least_overlap", "shortest"):
        heuristic_paths.append(
            snake_solver(
                grid, move_strategy=cast(MoveStrategy, move_strategy), start_down=False
            )
        )
        heuristic_paths.append(
            snake_solver(
                grid, move_strategy=cast(MoveStrategy, move_strategy), start_down=True
            )
        )
        heuristic_paths.append(
            spiral_solver(
                grid, move_strategy=cast(MoveStrategy, move_strategy), start_down=False
            )
        )
        heuristic_paths.append(
            spiral_solver(
                grid, move_strategy=cast(MoveStrategy, move_strategy), start_down=True
            )
        )

    unique_paths: dict[str, Path] = {}
    for path in heuristic_paths:
        unique_paths.setdefault(_path_signature(path), path)
    baseline_paths = list(unique_paths.values())
    baseline_scores = [(*_evaluate_path(grid, path), path) for path in baseline_paths]
    baseline_scores.sort(key=lambda row: (row[0], row[1], row[2], -row[3]))
    best_path = baseline_scores[0][4]
    best_score = baseline_scores[0][0]

    population_size = 10 * max(12, min(20, 10 + (open_count // 40)))
    generation_count = 10 * max(6, min(12, 5 + (open_count // 50)))
    elite_count = 2 * max(2, population_size // 4)
    local_search_steps = 5 * (3 if open_count <= 120 else 2 if open_count <= 220 else 1)
    stagnation_limit = 5 * (4 if open_count <= 180 else 3)

    seed_genomes: list[_MemeticGenome] = []
    for template in MEMETIC_TEMPLATES:
        seed_genomes.append(_genome_from_template(layout, template, rng))
    for _, _, _, _, path in baseline_scores[: min(4, len(baseline_scores))]:
        for template in MEMETIC_TEMPLATES[:2]:
            seed_genomes.append(_genome_from_path_seed(path, layout, template, rng))

    population: list[_MemeticCandidate] = []
    for genome in seed_genomes[:population_size]:
        population.append(
            _evaluate_genome(
                grid,
                layout,
                genome,
                start,
                least_overlap_path=least_overlap_moves,
                shortest_path=shortest_path_moves,
            )
        )

    while len(population) < population_size:
        template = rng.choice(MEMETIC_TEMPLATES)
        genome = _genome_from_template(layout, template, rng, jitter_scale=0.35)
        genome = _mutate_genome(
            genome, layout, rng, order_mutations=2, weight_scale=0.45
        )
        population.append(
            _evaluate_genome(
                grid,
                layout,
                genome,
                start,
                least_overlap_path=least_overlap_moves,
                shortest_path=shortest_path_moves,
            )
        )

    population.sort(key=_candidate_sort_key)
    stagnant_generations = 0

    for _ in range(generation_count):
        current_best = population[0]
        if _candidate_sort_key(current_best) < (
            best_score,
            len(best_path["moves"]),
            open_count,
            0,
        ):
            best_path = current_best.path
            best_score = current_best.score
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        if stagnant_generations >= stagnation_limit:
            break

        next_population = population[:elite_count]
        while len(next_population) < population_size:
            left = _tournament_select(population, rng)
            right = _tournament_select(population, rng)
            child_genome = _crossover_genomes(left.genome, right.genome, layout, rng)
            child_genome = _mutate_genome(
                child_genome,
                layout,
                rng,
                order_mutations=1 if open_count <= 140 else 2,
                weight_scale=0.35,
            )
            child = _evaluate_genome(
                grid,
                layout,
                child_genome,
                start,
                least_overlap_path=least_overlap_moves,
                shortest_path=shortest_path_moves,
            )
            child = _local_improve_candidate(
                child,
                grid,
                layout,
                start,
                rng,
                steps=local_search_steps,
                least_overlap_path=least_overlap_moves,
                shortest_path=shortest_path_moves,
            )
            next_population.append(child)

        population = sorted(next_population, key=_candidate_sort_key)

    best_candidate = min(population, key=_candidate_sort_key)
    best_baseline_score, _, _, _ = _evaluate_path(grid, best_path)
    if _candidate_sort_key(best_candidate) < (
        best_baseline_score,
        len(best_path["moves"]),
        0,
        0,
    ):
        return best_candidate.path
    return best_path
