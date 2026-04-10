import argparse
from pathlib import Path as FilePath
import sys

if __package__ in (None, ""):
    sys.path.append(str(FilePath(__file__).resolve().parents[2]))

from src.rl_solver.metrics import format_summary
from src.simple_rl_solver.evaluate import evaluate_model, summarize_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple MaskablePPO solver on generated grids using model's saved density range"
    )
    parser.add_argument(
        "--model", required=True, help="Path to saved simple MaskablePPO model"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Grid size used for evaluation (must not exceed model max size)",
    )
    parser.add_argument(
        "--seeds", type=int, default=50, help="Number of evaluation seeds"
    )
    parser.add_argument(
        "--start-seed", type=int, default=0, help="First evaluation seed"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic inference",
    )
    args = parser.parse_args()

    results = evaluate_model(
        args.model,
        size=args.size,
        seeds=range(args.start_seed, args.start_seed + args.seeds),
        deterministic=not args.stochastic,
    )
    summary = summarize_results(results)

    print(f"Evaluated {args.seeds} seeds at size {args.size}x{args.size}")
    print(format_summary(summary, prefix="simple_rl"))


if __name__ == "__main__":
    main()
