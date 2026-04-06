import argparse
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path as FilePath
import sys
import time
from typing import Iterator

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[1]))

from src.grid import create_random_grid
from src.shared_types import Grid, MOVE_DELTAS, Move, Path, Point
from src.solvers import find_start, snake_solver, spiral_solver
from src.visualize import (
    open_live_grid_path_tk,
    path_stats,
    show_grid_path_tk,
    show_grid_tk,
)


TERMINAL_EXACT_THRESHOLD = 6
PROGRESS_LOG_STATE_INTERVAL = 2048
SCRIPT_PROGRESS_INTERVAL_SECONDS = 1.0


type ProgressReporter = Callable[[str], None]


@dataclass(slots=True)
class SearchProgress:
    phase: str
    message: str
    elapsed_seconds: float
    states_explored: int
    states_per_second: float
    best_length: int
    max_visited_cells: int
    node_count: int
    pruned_branches: int
    best_path: Path


type ProgressCallback = Callable[[SearchProgress], None]


def _flush_print(message: str) -> None:
    print(message, flush=True)


def _iter_bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class _BranchAndBoundSearch:
    def __init__(
        self,
        grid: Grid,
        *,
        progress_reporter: ProgressReporter | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 5.0,
    ):
        if progress_interval_seconds <= 0:
            raise ValueError("progress_interval_seconds must be positive")

        self.grid = grid
        self.progress_reporter = progress_reporter
        self.progress_callback = progress_callback
        self.progress_interval_seconds = progress_interval_seconds
        self.started_at = 0.0
        self.next_progress_at = 0.0
        self.states_explored = 0
        self.bound_prunes = 0
        self.state_prunes = 0
        self.max_visited_cells = 0
        self.points: list[Point] = []
        self.point_colors: list[int] = []
        self.point_to_index: dict[Point, int] = {}
        self.neighbors: list[list[tuple[int, Move]]] = []
        self.adjacency_masks: list[int] = []
        self.degrees: list[int] = []
        self.color_masks = [0, 0]
        self.move_order = {
            move: index for index, move in enumerate(("u", "d", "l", "r"))
        }

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    index = len(self.points)
                    point = (row, col)
                    color = (row + col) & 1
                    self.points.append(point)
                    self.point_colors.append(color)
                    self.color_masks[color] |= 1 << index
                    self.point_to_index[point] = index

        self.node_count = len(self.points)
        self.all_visited_mask = (1 << self.node_count) - 1

        for point in self.points:
            row, col = point
            current_neighbors: list[tuple[int, Move]] = []
            neighbor_mask = 0
            for move_name in ("u", "d", "l", "r"):
                d_row, d_col = MOVE_DELTAS[move_name]
                next_point = (row + d_row, col + d_col)
                next_index = self.point_to_index.get(next_point)
                if next_index is None:
                    continue
                current_neighbors.append((next_index, move_name))
                neighbor_mask |= 1 << next_index
            self.neighbors.append(current_neighbors)
            self.adjacency_masks.append(neighbor_mask)
            self.degrees.append(len(current_neighbors))

        self.start = find_start(grid)
        if self.node_count == 0 or self.start is None:
            self.start_index = -1
            self.distances: list[list[int]] = []
            self.node_to_bridge_component: list[int] = []
            self.bridge_tree: list[list[int]] = []
            self.best_length = 0
            self.best_moves: list[Move] = []
            self.best_cost_by_state: dict[tuple[int, int], int] = {}
            self.lower_bound_cache: dict[tuple[int, int], int] = {}
            self.components_cache: dict[int, tuple[int, ...]] = {}
            self.node_to_mask_distance_cache: dict[tuple[int, int], int] = {}
            self.mask_distance_cache: dict[tuple[int, int], int] = {}
            self.component_pair_cost_cache: dict[
                tuple[int, ...], tuple[tuple[int, ...], ...]
            ] = {}
            self.component_path_start_cost_cache: dict[
                tuple[int, ...], tuple[int, ...]
            ] = {}
            self.exact_tail_cache: dict[tuple[int, int], tuple[Move, ...]] = {}
            self.bridge_extra_cache: dict[tuple[int, int], int] = {}
            return

        self.start_index = self.point_to_index[self.start]
        self._validate_connected_open_cells()
        self.distances = self._compute_all_pairs_distances()
        self.node_to_bridge_component, self.bridge_tree = self._build_bridge_tree()
        self.best_length = float("inf")
        self.best_moves: list[Move] = []
        self.best_cost_by_state: dict[tuple[int, int], int] = {}
        self.lower_bound_cache: dict[tuple[int, int], int] = {}
        self.components_cache: dict[int, tuple[int, ...]] = {}
        self.node_to_mask_distance_cache: dict[tuple[int, int], int] = {}
        self.mask_distance_cache: dict[tuple[int, int], int] = {}
        self.component_pair_cost_cache: dict[
            tuple[int, ...], tuple[tuple[int, ...], ...]
        ] = {}
        self.component_path_start_cost_cache: dict[
            tuple[int, ...], tuple[int, ...]
        ] = {}
        self.exact_tail_cache: dict[tuple[int, int], tuple[Move, ...]] = {}
        self.bridge_extra_cache: dict[tuple[int, int], int] = {}

    def solve(self) -> Path:
        if self.node_count == 0 or self.start is None:
            return {"start": None, "moves": []}

        self.started_at = time.perf_counter()
        self.next_progress_at = self.started_at + self.progress_interval_seconds

        incumbent = self._initial_upper_bound()
        self.best_length = len(incumbent["moves"])
        self.best_moves = list(incumbent["moves"])

        start_mask = 1 << self.start_index
        self.max_visited_cells = 1
        self._emit_progress(
            "started",
            "Optimal solver started: "
            f"{self.node_count} open cells, initial upper bound {self.best_length} moves, "
            f"reporting every {self.progress_interval_seconds:g}s.",
        )
        start_lower_bound = self._lower_bound(self.start_index, start_mask)
        self._search(
            self.start_index,
            start_mask,
            [],
            start_lower_bound,
        )
        elapsed = time.perf_counter() - self.started_at
        self.max_visited_cells = self.node_count
        self._emit_progress(
            "finished",
            "Optimal solver finished: "
            f"elapsed {self._format_elapsed(elapsed)}, "
            f"explored {self.states_explored:,} states, "
            f"pruned {self.bound_prunes + self.state_prunes:,} branches, "
            f"optimal length {self.best_length} moves.",
            now=time.perf_counter(),
        )
        return {"start": self.start, "moves": list(self.best_moves)}

    def _current_best_path(self) -> Path:
        return {"start": self.start, "moves": list(self.best_moves)}

    def _emit_progress(
        self,
        phase: str,
        message: str,
        *,
        now: float | None = None,
    ) -> None:
        if self.progress_reporter is None and self.progress_callback is None:
            return

        if self.progress_reporter is not None:
            self.progress_reporter(message)

        if self.progress_callback is None:
            return

        current_time = time.perf_counter() if now is None else now
        elapsed = max(0.0, current_time - self.started_at)
        states_per_second = self.states_explored / elapsed if elapsed > 0 else 0.0
        self.progress_callback(
            SearchProgress(
                phase=phase,
                message=message,
                elapsed_seconds=elapsed,
                states_explored=self.states_explored,
                states_per_second=states_per_second,
                best_length=self.best_length,
                max_visited_cells=self.max_visited_cells,
                node_count=self.node_count,
                pruned_branches=self.bound_prunes + self.state_prunes,
                best_path=self._current_best_path(),
            )
        )

    def _format_elapsed(self, elapsed: float) -> str:
        if elapsed < 60:
            return f"{elapsed:.1f}s"

        minutes, seconds = divmod(elapsed, 60.0)
        if minutes < 60:
            return f"{int(minutes)}m {seconds:04.1f}s"

        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes):02d}m {seconds:04.1f}s"

    def _maybe_report_progress(self, visited_mask: int) -> None:
        if self.progress_reporter is None and self.progress_callback is None:
            return

        self.states_explored += 1
        self.max_visited_cells = max(self.max_visited_cells, visited_mask.bit_count())
        if self.states_explored % PROGRESS_LOG_STATE_INTERVAL != 0:
            return

        now = time.perf_counter()
        if now < self.next_progress_at:
            return

        elapsed = now - self.started_at
        states_per_second = self.states_explored / elapsed if elapsed > 0 else 0.0
        self._emit_progress(
            "progress",
            "Optimal solver progress: "
            f"elapsed {self._format_elapsed(elapsed)}, "
            f"explored {self.states_explored:,} states ({states_per_second:,.0f}/s), "
            f"best {self.best_length} moves, "
            f"max coverage {self.max_visited_cells}/{self.node_count} cells, "
            f"pruned {self.bound_prunes + self.state_prunes:,} branches.",
            now=now,
        )
        self.next_progress_at = now + self.progress_interval_seconds

    def _validate_connected_open_cells(self) -> None:
        reachable = {self.start_index}
        queue = deque([self.start_index])

        while queue:
            node = queue.popleft()
            for neighbor, _ in self.neighbors[node]:
                if neighbor in reachable:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)

        if len(reachable) != self.node_count:
            raise ValueError("optimal_solver requires all open cells to be connected")

    def _compute_all_pairs_distances(self) -> list[list[int]]:
        distances = [
            [self.node_count + 1 for _ in range(self.node_count)]
            for _ in range(self.node_count)
        ]

        for start_index in range(self.node_count):
            distances[start_index][start_index] = 0
            queue = deque([start_index])
            while queue:
                node = queue.popleft()
                next_distance = distances[start_index][node] + 1
                for neighbor, _ in self.neighbors[node]:
                    if next_distance >= distances[start_index][neighbor]:
                        continue
                    distances[start_index][neighbor] = next_distance
                    queue.append(neighbor)

        return distances

    def _build_bridge_tree(self) -> tuple[list[int], list[list[int]]]:
        if self.node_count == 1:
            return [0], [[]]

        discovery = [-1 for _ in range(self.node_count)]
        low_link = [0 for _ in range(self.node_count)]
        bridges: set[tuple[int, int]] = set()
        time = 0

        def dfs(node: int, parent: int) -> None:
            nonlocal time
            discovery[node] = time
            low_link[node] = time
            time += 1

            for neighbor, _ in self.neighbors[node]:
                if neighbor == parent:
                    continue
                if discovery[neighbor] == -1:
                    dfs(neighbor, node)
                    low_link[node] = min(low_link[node], low_link[neighbor])
                    if low_link[neighbor] > discovery[node]:
                        bridge = (
                            (node, neighbor) if node < neighbor else (neighbor, node)
                        )
                        bridges.add(bridge)
                    continue
                low_link[node] = min(low_link[node], discovery[neighbor])

        dfs(self.start_index, -1)

        node_to_component = [-1 for _ in range(self.node_count)]
        component_count = 0

        for node in range(self.node_count):
            if node_to_component[node] != -1:
                continue

            queue = deque([node])
            node_to_component[node] = component_count

            while queue:
                current = queue.popleft()
                for neighbor, _ in self.neighbors[current]:
                    bridge = (
                        (current, neighbor)
                        if current < neighbor
                        else (neighbor, current)
                    )
                    if bridge in bridges or node_to_component[neighbor] != -1:
                        continue
                    node_to_component[neighbor] = component_count
                    queue.append(neighbor)

            component_count += 1

        bridge_tree = [[] for _ in range(component_count)]
        for left, right in bridges:
            left_component = node_to_component[left]
            right_component = node_to_component[right]
            bridge_tree[left_component].append(right_component)
            bridge_tree[right_component].append(left_component)

        return node_to_component, bridge_tree

    def _initial_upper_bound(self) -> Path:
        candidates = [self._greedy_upper_bound()]
        for start_down in (False, True):
            candidates.append(
                snake_solver(
                    self.grid,
                    move_strategy="shortest",
                    start_down=start_down,
                )
            )
            candidates.append(
                spiral_solver(
                    self.grid,
                    move_strategy="shortest",
                    start_down=start_down,
                )
            )

        valid_candidates = [
            path for path in candidates if self._path_covers_all_open_cells(path)
        ]
        if not valid_candidates:
            raise RuntimeError("failed to build an initial complete path")

        return min(valid_candidates, key=lambda path: len(path["moves"]))

    def _greedy_upper_bound(self) -> Path:
        current = self.start_index
        visited_mask = 1 << current
        moves: list[Move] = []

        while visited_mask != self.all_visited_mask:
            unvisited_neighbors = [
                (neighbor, move)
                for neighbor, move in self.neighbors[current]
                if not (visited_mask & (1 << neighbor))
            ]
            if unvisited_neighbors:
                unvisited_neighbors.sort(
                    key=lambda item: (
                        (self.adjacency_masks[item[0]] & ~visited_mask).bit_count(),
                        self.degrees[item[0]],
                        self.move_order[item[1]],
                    )
                )
                next_index, move = unvisited_neighbors[0]
                current = next_index
                visited_mask |= 1 << next_index
                moves.append(move)
                continue

            target = self._nearest_unvisited_index(current, visited_mask)
            connector = self._shortest_path_moves_between_indices(current, target)
            if connector is None:
                break

            for move in connector:
                current = self._step_index(current, move)
                visited_mask |= 1 << current
                moves.append(move)

        return {"start": self.start, "moves": moves}

    def _path_covers_all_open_cells(self, path: Path) -> bool:
        if path["start"] is None:
            return self.node_count == 0

        start_index = self.point_to_index.get(path["start"])
        if start_index is None:
            return False

        current = start_index
        visited_mask = 1 << current
        for move in path["moves"]:
            next_index = self._next_index(current, move)
            if next_index is None:
                return False
            current = next_index
            visited_mask |= 1 << current

        return visited_mask == self.all_visited_mask

    def _search(
        self,
        current: int,
        visited_mask: int,
        path: list[Move],
        lower_bound: int,
    ) -> None:
        steps = len(path)
        self._maybe_report_progress(visited_mask)
        if steps + lower_bound >= self.best_length:
            self.bound_prunes += 1
            return

        state = (current, visited_mask)
        previous_best = self.best_cost_by_state.get(state)
        if previous_best is not None and previous_best <= steps:
            self.state_prunes += 1
            return
        self.best_cost_by_state[state] = steps

        if visited_mask == self.all_visited_mask:
            self.best_length = steps
            self.best_moves = list(path)
            return

        unvisited_mask = self.all_visited_mask ^ visited_mask
        if unvisited_mask.bit_count() <= TERMINAL_EXACT_THRESHOLD:
            tail_moves = self._exact_tail_moves(current, unvisited_mask)
            total_steps = steps + len(tail_moves)
            if total_steps < self.best_length:
                self.best_length = total_steps
                self.best_moves = [*path, *tail_moves]
            return

        for next_index, next_mask, move, child_lower_bound in self._ordered_neighbors(
            current, visited_mask, steps
        ):
            path.append(move)
            self._search(next_index, next_mask, path, child_lower_bound)
            path.pop()

    def _ordered_neighbors(
        self, current: int, visited_mask: int, steps: int
    ) -> list[tuple[int, int, Move, int]]:
        ordered: list[tuple[tuple[int, int, int, int, int], int, int, Move, int]] = []

        for next_index, move in self.neighbors[current]:
            is_visited = 1 if visited_mask & (1 << next_index) else 0
            next_mask = visited_mask | (1 << next_index)
            remaining_after_move = self.all_visited_mask ^ next_mask
            child_lower_bound = (
                0
                if next_mask == self.all_visited_mask
                else self._lower_bound(next_index, next_mask)
            )
            optimistic_cost = steps + 1 + child_lower_bound
            if optimistic_cost >= self.best_length:
                continue

            branching = (
                self.adjacency_masks[next_index] & remaining_after_move
            ).bit_count()
            ordered.append(
                (
                    (
                        optimistic_cost,
                        is_visited,
                        branching,
                        self.degrees[next_index],
                        self.move_order[move],
                    ),
                    next_index,
                    next_mask,
                    move,
                    child_lower_bound,
                )
            )

        ordered.sort(key=lambda item: item[0])
        return [
            (next_index, next_mask, move, child_lower_bound)
            for _, next_index, next_mask, move, child_lower_bound in ordered
        ]

    def _lower_bound(self, current: int, visited_mask: int) -> int:
        unvisited_mask = self.all_visited_mask ^ visited_mask
        if unvisited_mask == 0:
            return 0

        cache_key = (current, unvisited_mask)
        cached = self.lower_bound_cache.get(cache_key)
        if cached is not None:
            return cached

        components = self._get_unvisited_components(unvisited_mask)
        internal_cost = sum(component.bit_count() - 1 for component in components)
        if len(components) <= 10:
            component_path_cost = self._component_path_lower_bound(current, components)
        else:
            component_path_cost = self._component_mst_lower_bound(current, components)

        component_bound = internal_cost + component_path_cost
        bridge_bound = self._bridge_lower_bound(current, unvisited_mask)
        bipartite_bound = self._bipartite_lower_bound(current, unvisited_mask)
        lower_bound = max(component_bound, bridge_bound, bipartite_bound)
        self.lower_bound_cache[cache_key] = lower_bound
        return lower_bound

    def _bipartite_lower_bound(self, current: int, unvisited_mask: int) -> int:
        current_color = self.point_colors[current]
        color_zero_targets = (unvisited_mask & self.color_masks[0]).bit_count()
        color_one_targets = (unvisited_mask & self.color_masks[1]).bit_count()
        if current_color == 0:
            return max(color_zero_targets * 2, color_one_targets * 2 - 1)
        return max(color_one_targets * 2, color_zero_targets * 2 - 1)

    def _get_unvisited_components(self, unvisited_mask: int) -> tuple[int, ...]:
        cached = self.components_cache.get(unvisited_mask)
        if cached is not None:
            return cached

        components: list[int] = []
        remaining = unvisited_mask

        while remaining:
            bit = remaining & -remaining
            queue = deque([bit.bit_length() - 1])
            remaining ^= bit
            component_mask = bit

            while queue:
                node = queue.popleft()
                frontier = self.adjacency_masks[node] & remaining
                while frontier:
                    next_bit = frontier & -frontier
                    frontier ^= next_bit
                    remaining ^= next_bit
                    component_mask |= next_bit
                    queue.append(next_bit.bit_length() - 1)

            components.append(component_mask)

        component_tuple = tuple(components)
        self.components_cache[unvisited_mask] = component_tuple
        return component_tuple

    def _get_component_pair_costs(
        self, components: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...]:
        cached = self.component_pair_cost_cache.get(components)
        if cached is not None:
            return cached

        count = len(components)
        pair_costs = [[0 for _ in range(count)] for _ in range(count)]
        for left in range(count):
            for right in range(left + 1, count):
                distance = self._min_distance_between_masks(
                    components[left], components[right]
                )
                pair_costs[left][right] = distance
                pair_costs[right][left] = distance

        pair_costs_tuple = tuple(tuple(row) for row in pair_costs)
        self.component_pair_cost_cache[components] = pair_costs_tuple
        return pair_costs_tuple

    def _get_component_path_start_costs(
        self, components: tuple[int, ...]
    ) -> tuple[int, ...]:
        cached = self.component_path_start_cost_cache.get(components)
        if cached is not None:
            return cached

        count = len(components)
        if count <= 1:
            start_costs = (0,)
            self.component_path_start_cost_cache[components] = start_costs
            return start_costs

        pair_costs = self._get_component_pair_costs(components)
        full_mask = (1 << count) - 1
        inf = self.node_count * max(1, count)
        tail_costs = [[inf for _ in range(count)] for _ in range(full_mask + 1)]
        for index in range(count):
            tail_costs[full_mask][index] = 0

        for mask in range(full_mask - 1, 0, -1):
            for last in range(count):
                if not (mask & (1 << last)):
                    continue
                remaining = full_mask ^ mask
                best = inf
                while remaining:
                    bit = remaining & -remaining
                    remaining ^= bit
                    next_index = bit.bit_length() - 1
                    best = min(
                        best,
                        pair_costs[last][next_index]
                        + tail_costs[mask | bit][next_index],
                    )
                if best < inf:
                    tail_costs[mask][last] = best

        start_costs = tuple(tail_costs[1 << index][index] for index in range(count))
        self.component_path_start_cost_cache[components] = start_costs
        return start_costs

    def _exact_tail_moves(self, current: int, unvisited_mask: int) -> tuple[Move, ...]:
        cache_key = (current, unvisited_mask)
        cached = self.exact_tail_cache.get(cache_key)
        if cached is not None:
            return cached

        targets = list(_iter_bits(unvisited_mask))
        if not targets:
            return ()

        target_index_by_node = [-1 for _ in range(self.node_count)]
        for target_index, node in enumerate(targets):
            target_index_by_node[node] = target_index

        full_target_mask = (1 << len(targets)) - 1
        queue = deque([(current, 0)])
        parents: dict[tuple[int, int], tuple[tuple[int, int], Move]] = {}
        seen = {(current, 0)}

        while queue:
            node, covered_mask = queue.popleft()
            if covered_mask == full_target_mask:
                moves: list[Move] = []
                state = (node, covered_mask)
                while state != (current, 0):
                    previous_state, move = parents[state]
                    moves.append(move)
                    state = previous_state
                moves.reverse()
                exact_moves = tuple(moves)
                self.exact_tail_cache[cache_key] = exact_moves
                self.lower_bound_cache[cache_key] = len(exact_moves)
                return exact_moves

            for next_index, move in self.neighbors[node]:
                next_mask = covered_mask
                target_index = target_index_by_node[next_index]
                if target_index >= 0:
                    next_mask |= 1 << target_index

                next_state = (next_index, next_mask)
                if next_state in seen:
                    continue
                seen.add(next_state)
                parents[next_state] = ((node, covered_mask), move)
                queue.append(next_state)

        raise RuntimeError("failed to solve exact tail for a connected grid state")

    def _component_path_lower_bound(
        self, current: int, components: tuple[int, ...]
    ) -> int:
        if not components:
            return 0
        if len(components) == 1:
            return self._min_distance_from_node_to_mask(current, components[0])

        start_path_costs = self._get_component_path_start_costs(components)
        return min(
            self._min_distance_from_node_to_mask(current, component)
            + start_path_costs[index]
            for index, component in enumerate(components)
        )

    def _component_mst_lower_bound(
        self, current: int, components: tuple[int, ...]
    ) -> int:
        best_costs = [
            self._min_distance_from_node_to_mask(current, component)
            for component in components
        ]
        pair_costs = self._get_component_pair_costs(components)
        in_tree = [False for _ in components]
        total = 0

        for _ in components:
            next_index = min(
                index for index, included in enumerate(in_tree) if not included
            )
            next_cost = best_costs[next_index]
            for index, included in enumerate(in_tree):
                if included or best_costs[index] >= next_cost:
                    continue
                next_index = index
                next_cost = best_costs[index]

            total += next_cost
            in_tree[next_index] = True

            for other_index, included in enumerate(in_tree):
                if included:
                    continue
                distance = pair_costs[next_index][other_index]
                if distance < best_costs[other_index]:
                    best_costs[other_index] = distance

        return total

    def _bridge_lower_bound(self, current: int, unvisited_mask: int) -> int:
        root_component = self.node_to_bridge_component[current]
        marked_components_mask = 1 << root_component
        for index in _iter_bits(unvisited_mask):
            marked_components_mask |= 1 << self.node_to_bridge_component[index]

        cache_key = (root_component, marked_components_mask)
        extra_bridge_steps = self.bridge_extra_cache.get(cache_key)
        if extra_bridge_steps is None:
            extra_bridge_steps = self._bridge_extra_steps(
                root_component, marked_components_mask
            )
            self.bridge_extra_cache[cache_key] = extra_bridge_steps

        return unvisited_mask.bit_count() + extra_bridge_steps

    def _bridge_extra_steps(
        self, root_component: int, marked_components_mask: int
    ) -> int:
        def dfs(node: int, parent: int, depth: int) -> tuple[bool, int, int]:
            has_marked = bool(marked_components_mask & (1 << node))
            included_edges = 0
            farthest_depth = depth if has_marked else -1

            for neighbor in self.bridge_tree[node]:
                if neighbor == parent:
                    continue
                child_has_marked, child_edges, child_farthest_depth = dfs(
                    neighbor, node, depth + 1
                )
                if not child_has_marked:
                    continue
                included_edges += child_edges + 1
                has_marked = True
                farthest_depth = max(farthest_depth, child_farthest_depth)

            return has_marked, included_edges, farthest_depth

        _, included_edges, farthest_depth = dfs(root_component, -1, 0)
        if farthest_depth < 0:
            return 0
        return max(0, included_edges - farthest_depth)

    def _min_distance_from_node_to_mask(self, node: int, mask: int) -> int:
        if mask == 0:
            return 0

        cache_key = (node, mask)
        cached = self.node_to_mask_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        best = self.node_count + 1
        for index in _iter_bits(mask):
            best = min(best, self.distances[node][index])

        self.node_to_mask_distance_cache[cache_key] = best
        return best

    def _min_distance_between_masks(self, left_mask: int, right_mask: int) -> int:
        cache_key = (
            (left_mask, right_mask)
            if left_mask <= right_mask
            else (right_mask, left_mask)
        )
        cached = self.mask_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        best = self.node_count + 1
        for left_index in _iter_bits(left_mask):
            row = self.distances[left_index]
            for right_index in _iter_bits(right_mask):
                best = min(best, row[right_index])
                if best == 1:
                    self.mask_distance_cache[cache_key] = best
                    return best

        self.mask_distance_cache[cache_key] = best
        return best

    def _nearest_unvisited_index(self, current: int, visited_mask: int) -> int:
        unvisited_mask = self.all_visited_mask ^ visited_mask
        candidates = list(_iter_bits(unvisited_mask))
        return min(
            candidates,
            key=lambda index: (
                self.distances[current][index],
                self.degrees[index],
                self.points[index],
            ),
        )

    def _shortest_path_moves_between_indices(
        self, start_index: int, target_index: int
    ) -> list[Move] | None:
        if start_index == target_index:
            return []

        queue = deque([start_index])
        parents: dict[int, tuple[int, Move]] = {}
        visited = {start_index}

        while queue:
            current = queue.popleft()
            for next_index, move in self.neighbors[current]:
                if next_index in visited:
                    continue
                parents[next_index] = (current, move)
                if next_index == target_index:
                    path: list[Move] = []
                    node = next_index
                    while node != start_index:
                        previous, step = parents[node]
                        path.append(step)
                        node = previous
                    path.reverse()
                    return path
                visited.add(next_index)
                queue.append(next_index)

        return None

    def _next_index(self, current: int, move: Move) -> int | None:
        current_point = self.points[current]
        d_row, d_col = MOVE_DELTAS[move]
        return self.point_to_index.get(
            (current_point[0] + d_row, current_point[1] + d_col)
        )

    def _step_index(self, current: int, move: Move) -> int:
        next_index = self._next_index(current, move)
        if next_index is None:
            raise RuntimeError(
                "encountered an invalid move while reconstructing a path"
            )
        return next_index


def optimal_solver(
    grid: Grid,
    *,
    progress_reporter: ProgressReporter | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval_seconds: float = 5.0,
) -> Path:
    return _BranchAndBoundSearch(
        grid,
        progress_reporter=progress_reporter,
        progress_callback=progress_callback,
        progress_interval_seconds=progress_interval_seconds,
    ).solve()


def main(
    n: int = 12,
    seed: int = 7,
    *,
    show_grid: bool = True,
    show_path: bool = True,
) -> None:
    demo_grid = create_random_grid(n, seed)
    live_window = None

    print(f"Simulated {n}x{n} grid (seed={seed})")

    if show_path:
        print("Launching live optimal path visualizer...")
        live_window = open_live_grid_path_tk(
            demo_grid,
            {"start": find_start(demo_grid), "moves": []},
            title="Optimal Solver Path Visualizer",
            status_line="Waiting for initial incumbent...",
        )
        if live_window is None:
            print(
                "Live optimal path visualizer could not start (likely no display environment)."
            )
    elif show_grid:
        print("Launching empty-grid visualizer...")
        launched_grid = show_grid_tk(demo_grid, title="Optimal Solver Grid Visualizer")
        if not launched_grid:
            print("Grid visualizer could not start (likely no display environment).")

    def update_live_window(progress: SearchProgress) -> None:
        nonlocal live_window
        if live_window is None:
            return
        if not live_window.update_path(
            progress.best_path, status_line=progress.message
        ):
            live_window = None

    optimal_path = optimal_solver(
        demo_grid,
        progress_reporter=_flush_print,
        progress_callback=update_live_window if live_window is not None else None,
        progress_interval_seconds=SCRIPT_PROGRESS_INTERVAL_SECONDS,
    )
    print(path_stats(optimal_path))

    if live_window is not None:
        live_window.update_path(
            optimal_path,
            status_line="Optimal solver finished. Final incumbent shown.",
        )
        live_window.run()
    elif show_path:
        print("Launching optimal path visualizer...")
        launched_path = show_grid_path_tk(
            demo_grid,
            optimal_path,
            title="Optimal Solver Path Visualizer",
        )
        if not launched_path:
            print(
                "Optimal path visualizer could not start (likely no display environment)."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the exact optimal solver on a generated grid"
    )
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=12,
        help="Grid size (size x size, default: 12)",
    )
    parser.add_argument(
        "seed",
        nargs="?",
        type=int,
        default=7,
        help="Seed (default: 7)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the base grid visualizer",
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Skip the optimal path visualizer",
    )
    args = parser.parse_args()

    main(
        args.n,
        args.seed,
        show_grid=not args.no_grid,
        show_path=not args.no_path,
    )
