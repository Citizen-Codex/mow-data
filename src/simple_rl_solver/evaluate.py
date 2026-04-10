from __future__ import annotations

from src.grid import create_random_grid
from src.rl_solver.metrics import RolloutResult, SummaryStats, summarize_rollout_results
from src.simple_rl_solver.config import make_env_config
from src.simple_rl_solver.model import load_simple_maskable_ppo
from src.simple_rl_solver.rollout import rollout_loaded_model_on_grid


def evaluate_model(
    model_path: str,
    *,
    size: int,
    seeds: range,
    deterministic: bool = True,
) -> list[RolloutResult]:
    results: list[RolloutResult] = []
    model, config = load_simple_maskable_ppo(model_path)
    env_config = make_env_config(config["env"])

    for seed in seeds:
        grid = create_random_grid(
            size,
            seed,
            removed_fraction_range=(
                env_config.removed_fraction_min,
                env_config.removed_fraction_max,
            ),
        )
        results.append(
            rollout_loaded_model_on_grid(
                model,
                env_config,
                grid,
                deterministic=deterministic,
            )
        )

    return results


def summarize_results(rows: list[RolloutResult]) -> SummaryStats:
    return summarize_rollout_results(rows)
