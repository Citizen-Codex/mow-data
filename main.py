import argparse
import random

from src.grid import create_random_grid
from src.solvers import random_walk_solver, snake_solver, spiral_solver
from src.optimal_solver import optimal_solver
from src.visualize import path_stats, show_grid_path_tk, show_grid_tk


def main(n: int, seed: int):
    demo_grid = create_random_grid(n, seed)

    print(f"Simulated {n}x{n} grid (seed={seed})")
    print("Launching empty-grid visualizer...")

    launched_grid = show_grid_tk(demo_grid, title="Grid Visualizer")
    if not launched_grid:
        print("Grid visualizer could not start (likely no display environment).")

    snake_path = snake_solver(demo_grid)
    print(path_stats(snake_path))
    print("Launching snake path visualizer...")

    launched_snake = show_grid_path_tk(
        demo_grid, snake_path, title="Snake Path Visualizer"
    )
    if not launched_snake:
        print("Snake path visualizer could not start (likely no display environment).")

    spiral_path = spiral_solver(demo_grid)
    print(path_stats(spiral_path))
    print("Launching spiral path visualizer...")

    launched_spiral = show_grid_path_tk(
        demo_grid, spiral_path, title="Spiral Path Visualizer"
    )
    if not launched_spiral:
        print("Spiral path visualizer could not start (likely no display environment).")

    random_walk_path = random_walk_solver(demo_grid, rng=random.Random(seed))
    print(path_stats(random_walk_path))
    print("Launching random walk path visualizer...")

    launched_random_walk = show_grid_path_tk(
        demo_grid, random_walk_path, title="Random Walk Path Visualizer"
    )
    if not launched_random_walk:
        print(
            "Random walk path visualizer could not start (likely no display environment)."
        )

    optimal_path = optimal_solver(
        demo_grid,
        progress_reporter=lambda message: print(message, flush=True),
    )
    print(path_stats(optimal_path))
    print("Launching optimal path visualizer...")

    launched_optimal = show_grid_path_tk(
        demo_grid, optimal_path, title="Optimal Path Visualizer"
    )
    if not launched_optimal:
        print(
            "Optimal path visualizer could not start (likely no display environment)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View and solve paths for a grid size and seed"
    )
    parser.add_argument("n", type=int, help="Grid size (size x size)")
    parser.add_argument("seed", type=int, help="Seed")
    args = parser.parse_args()

    main(args.n, args.seed)
