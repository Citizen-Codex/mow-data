import argparse
import random

from src.concorde import concorde_solver
from src.grid import create_random_grid
from src.solvers import random_walk_solver, snake_solver, spiral_solver
from src.optimal_solver import optimal_solver
from src.visualize import path_stats, show_grid_path_tk, show_grid_tk

SOLVERS = ["snake", "spiral", "random_walk", "optimal", "concorde"]


def main(n: int, seed: int, skip: set[str]):
    demo_grid = create_random_grid(n, seed)

    print(f"Simulated {n}x{n} grid (seed={seed})")
    print("Launching empty-grid visualizer...")

    launched_grid = show_grid_tk(demo_grid, title="Grid Visualizer")
    if not launched_grid:
        print("Grid visualizer could not start (likely no display environment).")

    if "snake" not in skip:
        snake_path = snake_solver(demo_grid)
        print(path_stats(snake_path))
        print("Launching snake path visualizer...")
        launched = show_grid_path_tk(demo_grid, snake_path, title="Snake Path Visualizer")
        if not launched:
            print("Snake path visualizer could not start (likely no display environment).")

    if "spiral" not in skip:
        spiral_path = spiral_solver(demo_grid)
        print(path_stats(spiral_path))
        print("Launching spiral path visualizer...")
        launched = show_grid_path_tk(demo_grid, spiral_path, title="Spiral Path Visualizer")
        if not launched:
            print("Spiral path visualizer could not start (likely no display environment).")

    if "random_walk" not in skip:
        random_walk_path = random_walk_solver(demo_grid, rng=random.Random(seed))
        print(path_stats(random_walk_path))
        print("Launching random walk path visualizer...")
        launched = show_grid_path_tk(
            demo_grid, random_walk_path, title="Random Walk Path Visualizer"
        )
        if not launched:
            print(
                "Random walk path visualizer could not start (likely no display environment)."
            )

    if "optimal" not in skip:
        optimal_path = optimal_solver(
            demo_grid,
            progress_reporter=lambda message: print(message, flush=True),
        )
        print(path_stats(optimal_path))
        print("Launching optimal path visualizer...")
        launched = show_grid_path_tk(
            demo_grid, optimal_path, title="Optimal Path Visualizer"
        )
        if not launched:
            print(
                "Optimal path visualizer could not start (likely no display environment)."
            )

    if "concorde" not in skip:
        concorde_path = concorde_solver(demo_grid)
        print(path_stats(concorde_path))
        print("Launching concorde path visualizer...")
        launched = show_grid_path_tk(
            demo_grid, concorde_path, title="Concorde Path Visualizer"
        )
        if not launched:
            print(
                "Concorde path visualizer could not start (likely no display environment)."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View and solve paths for a grid size and seed"
    )
    parser.add_argument("n", type=int, help="Grid size (size x size)")
    parser.add_argument("seed", type=int, help="Seed")
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help=f"Comma-separated solver names to skip ({', '.join(SOLVERS)})",
    )
    args = parser.parse_args()

    skip = set()
    if args.skip:
        for name in args.skip.split(","):
            name = name.strip()
            if name not in SOLVERS:
                parser.error(f"unknown solver '{name}'; choose from: {', '.join(SOLVERS)}")
            skip.add(name)

    main(args.n, args.seed, skip=skip)
