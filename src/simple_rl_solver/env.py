from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.grid import create_random_grid
from src.simple_rl_solver.config import ACTION_ORDER, EnvConfig, validate_env_config
from src.shared_types import Grid, MOVE_DELTAS, Move, Path, Point
from src.solvers import find_start


@dataclass(slots=True)
class PooledGrid:
    grid_seed: int
    grid: Grid
    start: Point
    open_cell_count: int
    obs_template: np.ndarray
    action_masks: np.ndarray


class SimpleLawnMowingEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, *, record_path: bool = True):
        super().__init__()
        validate_env_config(config)
        self.config = config
        self.record_path = record_path
        self._board_size = config.size
        self._cell_count = config.size * config.size
        self._remaining_steps_index = self._cell_count * 3
        self._stall_steps_index = self._remaining_steps_index + 1
        self._last_action_start = self._stall_steps_index + 1
        self._obs_size = self._last_action_start + len(ACTION_ORDER)
        self.action_space = spaces.Discrete(len(ACTION_ORDER))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._obs_size,),
            dtype=np.float32,
        )

        self.grid: Grid = []
        self.grid_seed = config.base_seed
        self.start: Point | None = None
        self.position: Point | None = None
        self.previous_position: Point | None = None
        self.last_action_index: int | None = None
        self.steps_since_last_new_cell = 0
        self.obs_buffer = np.zeros((self._obs_size,), dtype=np.float32)
        self.open_mask = self.obs_buffer[: self._cell_count].reshape(
            (self._board_size, self._board_size)
        )
        self.visited_mask = self.obs_buffer[
            self._cell_count : self._cell_count * 2
        ].reshape((self._board_size, self._board_size))
        self.agent_mask = self.obs_buffer[
            self._cell_count * 2 : self._cell_count * 3
        ].reshape((self._board_size, self._board_size))
        self.open_cell_count = 0
        self.covered_cell_count = 0
        self.steps_taken = 0
        self.max_steps = 0
        self.moves: list[Move] = []
        self.agent_flat_index: int | None = None
        self.empty_action_mask = np.zeros(len(ACTION_ORDER), dtype=bool)
        self.all_actions_mask = np.ones(len(ACTION_ORDER), dtype=bool)
        self.current_action_masks = np.zeros(
            (self._board_size, self._board_size, len(ACTION_ORDER)),
            dtype=bool,
        )
        self.grid_pool: list[PooledGrid] = self._build_grid_pool()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        provided_grid = options.get("grid")
        if provided_grid is not None:
            runtime_grid = self._build_grid_data(
                self._normalize_grid(provided_grid),
                int(options.get("grid_seed", self.config.base_seed)),
            )
        else:
            requested_seed = int(
                options.get(
                    "grid_seed",
                    seed if seed is not None else self.np_random.integers(0, 2**31 - 1),
                )
            )
            runtime_grid = self._sample_grid(requested_seed)

        self.grid_seed = runtime_grid.grid_seed
        self.grid = runtime_grid.grid
        self.start = runtime_grid.start
        self.open_cell_count = runtime_grid.open_cell_count
        self.current_action_masks = runtime_grid.action_masks

        if self.start is None:
            raise ValueError("Grid must contain at least one open cell")

        self.position = self.start
        self.previous_position = None
        self.last_action_index = None
        np.copyto(self.obs_buffer, runtime_grid.obs_template)
        self.covered_cell_count = 1
        self.steps_taken = 0
        self.steps_since_last_new_cell = 0
        self.max_steps = max(
            1, math.ceil(self.open_cell_count * self.config.max_steps_factor)
        )
        self.moves = []

        row, col = self.start
        self.visited_mask[row, col] = 1.0
        self.agent_flat_index = self._flat_index(row, col)
        self.obs_buffer[self._cell_count * 2 + self.agent_flat_index] = 1.0

        return self._get_obs(), self._get_info(include_always=True)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.position is None:
            raise RuntimeError("Environment must be reset before stepping")

        reward = 0.0
        terminated = False
        truncated = False
        stall_terminated = False

        move = ACTION_ORDER[action]
        prior_position = self.position
        row, col = self.position
        d_row, d_col = MOVE_DELTAS[move]
        next_row = row + d_row
        next_col = col + d_col

        if self._is_valid_cell(next_row, next_col):
            next_position = (next_row, next_col)
            reverse_edge = (
                self.previous_position is not None
                and next_position == self.previous_position
            )
            if self.agent_flat_index is not None:
                self.obs_buffer[self._cell_count * 2 + self.agent_flat_index] = 0.0
            self.position = next_position
            self.agent_flat_index = self._flat_index(next_row, next_col)
            self.obs_buffer[self._cell_count * 2 + self.agent_flat_index] = 1.0
            self.previous_position = prior_position
            self.last_action_index = action
            if self.record_path:
                self.moves.append(move)

            if self.visited_mask[next_row, next_col] == 0.0:
                self.visited_mask[next_row, next_col] = 1.0
                self.covered_cell_count += 1
                reward += self.config.reward_new_cell
                self.steps_since_last_new_cell = 0
            else:
                reward += self.config.reward_revisit
                if reverse_edge:
                    reward += self.config.reward_reverse_edge
                self.steps_since_last_new_cell += 1

        self.steps_taken += 1
        if self.covered_cell_count >= self.open_cell_count:
            reward += self.config.reward_complete
            terminated = True
        elif self.steps_since_last_new_cell >= self.config.stall_grace_steps > 0:
            reward += self._get_stall_termination_penalty()
            terminated = True
            stall_terminated = True
        elif self.steps_taken >= self.max_steps:
            truncated = True

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(
                episode_end=terminated or truncated,
                stall_terminated=stall_terminated,
            ),
        )

    def get_path(self) -> Path:
        return {"start": self.start, "moves": list(self.moves)}

    def action_masks(self) -> np.ndarray:
        if self.position is None:
            return self.empty_action_mask
        if self.covered_cell_count >= self.open_cell_count and self.open_cell_count > 0:
            return self.all_actions_mask

        row, col = self.position
        mask = self.current_action_masks[row, col]
        if mask.any():
            return mask
        return self.all_actions_mask

    def _get_obs(self) -> np.ndarray:
        remaining_steps = max(0, self.max_steps - self.steps_taken)
        self.obs_buffer[self._remaining_steps_index] = remaining_steps / max(
            1, self.max_steps
        )
        self.obs_buffer[self._stall_steps_index] = self.steps_since_last_new_cell / max(
            1, self.max_steps
        )
        self.obs_buffer[self._last_action_start : self._obs_size] = 0.0
        if self.last_action_index is not None:
            self.obs_buffer[self._last_action_start + self.last_action_index] = 1.0
        return self.obs_buffer

    def _get_info(
        self,
        *,
        episode_end: bool = False,
        include_always: bool = False,
        stall_terminated: bool = False,
    ) -> dict[str, Any]:
        if not episode_end and not include_always:
            return {}
        return {
            "grid_seed": self.grid_seed,
            "covered_cells": self.covered_cell_count,
            "open_cells": self.open_cell_count,
            "coverage_ratio": self.covered_cell_count / max(1, self.open_cell_count),
            "steps_taken": self.steps_taken,
            "steps_remaining": max(0, self.max_steps - self.steps_taken),
            "steps_since_last_new_cell": self.steps_since_last_new_cell,
            "stall_terminated": stall_terminated,
        }

    def _get_stall_termination_penalty(self) -> float:
        uncovered_ratio = 1.0 - (self.covered_cell_count / max(1, self.open_cell_count))
        return uncovered_ratio * self.config.reward_stall_terminate_scale

    def _generate_grid(self, seed: int) -> tuple[int, Grid, Point]:
        for seed_offset in range(32):
            grid = create_random_grid(
                self.config.size,
                seed + seed_offset,
                removed_fraction_range=(
                    self.config.removed_fraction_min,
                    self.config.removed_fraction_max,
                ),
            )
            start = find_start(grid)
            if start is not None:
                return seed + seed_offset, grid, start
        raise RuntimeError("Failed to generate a grid with at least one open cell")

    def _build_grid_pool(self) -> list[PooledGrid]:
        if self.config.grid_pool_size <= 0:
            return []

        grid_pool: list[PooledGrid] = []
        seed = self.config.base_seed
        while len(grid_pool) < self.config.grid_pool_size:
            grid_seed, grid, start = self._generate_grid(seed)
            grid_pool.append(self._build_grid_data(grid, grid_seed, start))
            seed = grid_seed + 1
        return grid_pool

    def _sample_grid(self, requested_seed: int) -> PooledGrid:
        if not self.grid_pool:
            grid_seed, grid, start = self._generate_grid(requested_seed)
            return self._build_grid_data(grid, grid_seed, start)

        pool_index = requested_seed % len(self.grid_pool)
        return self.grid_pool[pool_index]

    def _build_grid_data(
        self,
        grid: Grid,
        grid_seed: int,
        start: Point | None = None,
    ) -> PooledGrid:
        resolved_start = start if start is not None else find_start(grid)
        if resolved_start is None:
            raise RuntimeError("Grid must contain at least one open cell")

        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        open_cell_count = sum(cell == 1 for row in grid for cell in row)
        obs_template = np.zeros((self._obs_size,), dtype=np.float32)
        open_mask = obs_template[: self._cell_count].reshape(
            (self._board_size, self._board_size)
        )
        open_mask[:rows, :cols] = np.asarray(grid, dtype=np.float32)

        action_masks = np.zeros(
            (self._board_size, self._board_size, len(ACTION_ORDER)),
            dtype=bool,
        )
        for row_index in range(rows):
            for col_index in range(cols):
                if grid[row_index][col_index] != 1:
                    continue
                for action_index, move in enumerate(ACTION_ORDER):
                    d_row, d_col = MOVE_DELTAS[move]
                    next_row = row_index + d_row
                    next_col = col_index + d_col
                    action_masks[row_index, col_index, action_index] = (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and grid[next_row][next_col] == 1
                    )

        return PooledGrid(
            grid_seed=grid_seed,
            grid=grid,
            start=resolved_start,
            open_cell_count=open_cell_count,
            obs_template=obs_template,
            action_masks=action_masks,
        )

    def _normalize_grid(self, grid: Grid) -> Grid:
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")

        size = len(grid)
        if size > self.config.size:
            raise ValueError(f"Grid size {size} exceeds model size {self.config.size}")
        if any(len(row) != size for row in grid):
            raise ValueError("Grid must be square")
        normalized = [list(row) for row in grid]
        if find_start(normalized) is None:
            raise ValueError("Grid must contain at least one open cell")
        return normalized

    def _is_valid_cell(self, row: int, col: int) -> bool:
        rows = len(self.grid)
        cols = len(self.grid[0]) if rows else 0
        return 0 <= row < rows and 0 <= col < cols and self.grid[row][col] == 1

    def _flat_index(self, row: int, col: int) -> int:
        return row * self._board_size + col
