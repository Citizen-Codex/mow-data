import argparse
from pathlib import Path as FilePath
import sys

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[2]))

from src.grid import create_random_grid
from src.simple_rl_solver.config import load_run_config, make_env_config
from src.simple_rl_solver.rollout import rollout_model_on_grid
from src.visualize import path_stats, show_grid_path_tk, show_grid_tk


def _print_result(label: str, result: dict) -> None:
    metrics = result["metrics"]
    info = result.get("info", {})
    print(
        f"{label}: completed={int(bool(metrics['completed']))} "
        f"coverage={float(metrics['coverage_ratio']):.3f} "
        f"moves={int(metrics['moves'])} "
        f"overlaps={int(metrics['overlaps'])} "
        f"reward={float(result['total_reward']):.2f} "
        f"steps_since_last_new_cell={int(info.get('steps_since_last_new_cell', 0))} "
        f"stall_terminated={int(bool(info.get('stall_terminated', False)))} "
        f"edge_reuses={int(result['edge_reuses'])} "
        f"loop_oscillations={int(result['loop_oscillations'])} "
        f"max_oscillation_streak={int(result['max_oscillation_streak'])} "
        f"max_cell_overlap_count={int(result['max_cell_overlap_count'])} "
        f"overlap_limit_hit={int(bool(result['overlap_limit_hit']))} "
        f"terminated={int(bool(result['terminated']))} "
        f"truncated={int(bool(result['truncated']))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize generated grid and simple MaskablePPO solution using saved density range"
    )
    parser.add_argument(
        "--model", required=True, help="Path to saved simple MaskablePPO model"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Grid size to visualize (defaults to model max size)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Grid seed to visualize")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic inference",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip base grid window and only show solution path windows",
    )
    args = parser.parse_args()

    config = load_run_config(args.model)
    env_config = make_env_config(config["env"])
    size = int(env_config.size if args.size is None else args.size)
    grid = create_random_grid(
        size,
        args.seed,
        removed_fraction_range=(
            env_config.removed_fraction_min,
            env_config.removed_fraction_max,
        ),
    )
    model_result = rollout_model_on_grid(
        args.model,
        grid,
        deterministic=not args.stochastic,
    )

    print(
        f"Visualizing seed={args.seed} size={size}x{size} "
        f"(model_max_size={config['env']['size']})"
    )
    _print_result("simple_rl", model_result)
    print(path_stats(model_result["path"]))

    if not args.no_grid:
        launched_grid = show_grid_tk(
            grid, title=f"SimpleLawnMowingEnv Grid (seed={args.seed})"
        )
        if not launched_grid:
            print("Grid visualizer could not start (likely no display environment).")

    launched_model = show_grid_path_tk(
        grid,
        model_result["path"],
        title=f"Simple MaskablePPO Path (seed={args.seed})",
    )
    if not launched_model:
        print("Model path visualizer could not start (likely no display environment).")


if __name__ == "__main__":
    main()
