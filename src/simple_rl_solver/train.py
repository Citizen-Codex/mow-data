import argparse
from dataclasses import replace
from pathlib import Path as FilePath
import sys

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[2]))

from src.simple_rl_solver.config import EnvConfig, TrainConfig, resolve_train_config
from src.simple_rl_solver.model import train_simple_maskable_ppo


def _override_config(config, **overrides):
    filtered_overrides = {
        key: value for key, value in overrides.items() if value is not None
    }
    return replace(config, **filtered_overrides)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a minimal MaskablePPO solver for lawn grid"
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Grid size (size x size)"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Training timesteps"
    )
    parser.add_argument("--n-envs", type=int, help="Parallel training environments")
    parser.add_argument("--n-steps", type=int, help="Rollout steps per environment")
    parser.add_argument("--n-epochs", type=int, help="PPO epochs per rollout")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="MaskablePPO optimizer learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="MaskablePPO discount factor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Replay minibatch size",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        help="Entropy bonus coefficient",
    )
    parser.add_argument(
        "--removed-fraction-min",
        type=float,
        help="Minimum removed-cell fraction for generated grids",
    )
    parser.add_argument(
        "--removed-fraction-max",
        type=float,
        help="Maximum removed-cell fraction for generated grids",
    )
    parser.add_argument(
        "--max-steps-factor",
        type=float,
        help="Episode step budget as a multiple of open-cell count",
    )
    parser.add_argument(
        "--grid-pool-size",
        type=int,
        help="Cached generated grids per environment (0 disables pooling)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0, help="Base seed for generated grids"
    )
    parser.add_argument("--seed", type=int, default=0, help="Training RNG seed")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device (`auto`, `cpu`, `cuda`, `mps`)",
    )
    parser.add_argument(
        "--resume",
        help="Resume training from existing simple MaskablePPO checkpoint",
    )
    parser.add_argument(
        "--output",
        default="data/rl/simple_ppo.zip",
        help="Model output path",
    )
    args = parser.parse_args()

    env_config = _override_config(
        EnvConfig(size=args.size),
        removed_fraction_min=args.removed_fraction_min,
        removed_fraction_max=args.removed_fraction_max,
        max_steps_factor=args.max_steps_factor,
        grid_pool_size=args.grid_pool_size,
        base_seed=args.base_seed,
    )
    train_config = resolve_train_config(
        _override_config(
            TrainConfig(total_timesteps=args.timesteps),
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
        )
    )
    print(
        "Training config: "
        f"timesteps={train_config.total_timesteps} "
        f"size={env_config.size} "
        f"n_envs={train_config.n_envs} "
        f"n_steps={train_config.n_steps} "
        f"n_epochs={train_config.n_epochs} "
        f"removed_fraction_min={env_config.removed_fraction_min} "
        f"removed_fraction_max={env_config.removed_fraction_max} "
        f"max_steps_factor={env_config.max_steps_factor} "
        f"grid_pool_size={env_config.grid_pool_size} "
        f"learning_rate={train_config.learning_rate} "
        f"gamma={train_config.gamma} "
        f"batch_size={train_config.batch_size} "
        f"ent_coef={train_config.ent_coef} "
        f"device={train_config.device} "
        f"resume={args.resume or 'none'} "
        f"seed={train_config.seed}"
    )
    model_path = train_simple_maskable_ppo(
        env_config,
        train_config,
        args.output,
        resume_from=args.resume,
    )
    print(f"Saved simple RL model to {model_path}")


if __name__ == "__main__":
    main()
