import argparse
from pathlib import Path as FilePath
import sys
import tempfile
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[2]))

from src.simple_rl_solver.config import EnvConfig, TrainConfig, resolve_train_config
from src.simple_rl_solver.env import SimpleLawnMowingEnv
from src.simple_rl_solver.model import train_simple_maskable_ppo


def run_env_microbenchmark(
    *,
    size: int,
    grid_pool_size: int,
    reset_count: int,
    mask_calls: int,
    step_count: int,
) -> dict[str, float | int]:
    env = SimpleLawnMowingEnv(
        EnvConfig(size=size, grid_pool_size=grid_pool_size),
        record_path=False,
    )
    rng = np.random.default_rng(0)

    try:
        started_at = time.perf_counter()
        for seed in range(reset_count):
            env.reset(seed=seed)
        reset_seconds = max(1e-9, time.perf_counter() - started_at)

        env.reset(seed=0)
        started_at = time.perf_counter()
        for _ in range(mask_calls):
            env.action_masks()
        mask_seconds = max(1e-9, time.perf_counter() - started_at)

        env.reset(seed=0)
        started_at = time.perf_counter()
        completed_steps = 0
        next_seed = 1
        while completed_steps < step_count:
            mask = env.action_masks()
            valid_actions = np.flatnonzero(mask)
            action = int(valid_actions[rng.integers(0, len(valid_actions))])
            _, _, terminated, truncated, _ = env.step(action)
            completed_steps += 1
            if terminated or truncated:
                env.reset(seed=next_seed)
                next_seed += 1
        step_seconds = max(1e-9, time.perf_counter() - started_at)
    finally:
        env.close()

    return {
        "size": size,
        "grid_pool_size": grid_pool_size,
        "reset_count": reset_count,
        "resets_per_sec": reset_count / reset_seconds,
        "mask_calls": mask_calls,
        "mask_calls_per_sec": mask_calls / mask_seconds,
        "step_count": step_count,
        "steps_per_sec": step_count / step_seconds,
    }


def run_train_benchmark(
    *,
    label: str,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> dict[str, float | int | str]:
    resolved_train_config = resolve_train_config(train_config)
    with tempfile.TemporaryDirectory(prefix="simple_rl_bench_") as temp_dir:
        model_path = FilePath(temp_dir) / f"{label}.zip"
        started_at = time.perf_counter()
        train_simple_maskable_ppo(
            env_config,
            resolved_train_config,
            model_path,
            log_interval_seconds=None,
        )
        elapsed_seconds = max(1e-9, time.perf_counter() - started_at)

    return {
        "label": label,
        "timesteps": resolved_train_config.total_timesteps,
        "n_envs": resolved_train_config.n_envs,
        "n_steps": resolved_train_config.n_steps,
        "n_epochs": resolved_train_config.n_epochs,
        "batch_size": resolved_train_config.batch_size,
        "ent_coef": resolved_train_config.ent_coef,
        "device": resolved_train_config.device,
        "elapsed_seconds": elapsed_seconds,
        "fps": resolved_train_config.total_timesteps / elapsed_seconds,
    }


def print_env_microbenchmark(result: dict[str, float | int]) -> None:
    print(
        "env_microbenchmark "
        f"size={result['size']} "
        f"grid_pool_size={result['grid_pool_size']} "
        f"resets_per_sec={result['resets_per_sec']:.1f} "
        f"mask_calls_per_sec={result['mask_calls_per_sec']:.1f} "
        f"steps_per_sec={result['steps_per_sec']:.1f}"
    )


def print_train_benchmark(result: dict[str, float | int | str]) -> None:
    print(
        "train_benchmark "
        f"label={result['label']} "
        f"timesteps={result['timesteps']} "
        f"n_envs={result['n_envs']} "
        f"n_steps={result['n_steps']} "
        f"n_epochs={result['n_epochs']} "
        f"batch_size={result['batch_size']} "
        f"ent_coef={result['ent_coef']} "
        f"device={result['device']} "
        f"elapsed={result['elapsed_seconds']:.2f}s "
        f"fps={result['fps']:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple RL throughput benchmarks")
    parser.add_argument(
        "--mode",
        choices=("all", "env", "train"),
        default="all",
        help="Benchmark mode",
    )
    parser.add_argument("--size", type=int, default=8, help="Grid size")
    parser.add_argument("--grid-pool-size", type=int, default=64, help="Grid pool size")
    parser.add_argument(
        "--timesteps", type=int, default=4096, help="Training timesteps per case"
    )
    args = parser.parse_args()

    if args.mode in {"all", "env"}:
        print_env_microbenchmark(
            run_env_microbenchmark(
                size=args.size,
                grid_pool_size=args.grid_pool_size,
                reset_count=200,
                mask_calls=200_000,
                step_count=20_000,
            )
        )

    if args.mode in {"all", "train"}:
        env_config = EnvConfig(size=args.size, grid_pool_size=args.grid_pool_size)
        cases = [
            (
                "baseline",
                TrainConfig(
                    total_timesteps=args.timesteps,
                    n_envs=8,
                    n_steps=128,
                    n_epochs=10,
                    batch_size=64,
                    ent_coef=0.01,
                    device="cpu",
                ),
            ),
            (
                "low_epochs_big_batch",
                TrainConfig(
                    total_timesteps=args.timesteps,
                    n_envs=8,
                    n_steps=128,
                    n_epochs=2,
                    batch_size=256,
                    ent_coef=0.01,
                    device="cpu",
                ),
            ),
            (
                "bigger_rollout",
                TrainConfig(
                    total_timesteps=args.timesteps,
                    n_envs=8,
                    n_steps=256,
                    n_epochs=2,
                    batch_size=256,
                    ent_coef=0.01,
                    device="cpu",
                ),
            ),
            (
                "fewer_envs",
                TrainConfig(
                    total_timesteps=args.timesteps,
                    n_envs=4,
                    n_steps=256,
                    n_epochs=2,
                    batch_size=256,
                    ent_coef=0.01,
                    device="cpu",
                ),
            ),
        ]
        for label, train_config in cases:
            print_train_benchmark(
                run_train_benchmark(
                    label=label,
                    env_config=env_config,
                    train_config=train_config,
                )
            )


if __name__ == "__main__":
    main()
