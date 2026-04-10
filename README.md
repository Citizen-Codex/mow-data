# Mow The Lawn

Mow The Lawn is a small Python project for generating connected random lawn grids, solving traversal paths with heuristic and exact strategies, and visualizing the result. It also includes a simulation pipeline to generate labelled path data for basic classification experiments.

## Project purpose

- Generate deterministic random grids from a size and seed.
- Compare heuristic traversal strategies, a memetic genetic solver, and an exact branch-and-bound solver.
- Visualize base grids and computed paths.
- Simulate many grids to build labelled training data (`snake_path`, `spiral_path`, `random_walk_path`).

## Codebase components

- `main.py`: Entrypoint for running a single grid demo and visualizing solver output.
- `simulate.py`: Entrypoint for generating labelled CSV data through the `simulate()` function.
- `classifier.py`: Train/evaluate/classify path strings as snake-like, spiral-like, or random-walk-like.
- `src/simple_rl_solver/train.py`: Train a minimal masked PPO baseline on fixed-size generated grids.
- `src/simple_rl_solver/eval.py`: Evaluate a saved minimal masked PPO model across generated seeds.
- `src/simple_rl_solver/visualize.py`: Visualize a saved minimal masked PPO rollout on one generated grid.
- `src/grid.py`: Random grid generation with connectivity-safe cell removals.
- `src/solvers.py`: Snake/spiral/random-walk solvers, pathing strategies (`shortest`, `least_overlap`), and the shared fixed-start/path helpers.
- `src/simple_rl_solver/`: Minimal RL baseline with a flat observation, simple reward landscape, and masked PPO.
- `src/memetic_solver/`: Structure-aware memetic genetic solver for fast near-optimal coverage paths.
- `src/optimal_solver.py`: Exact `optimal_solver()` branch-and-bound search for minimum-move full coverage.
- `src/visualize.py`: Tkinter visualization helpers and path statistics.
- `src/shared_types.py`: Shared types (`Grid`, `Path`, moves, and deltas).
- `data/labelled_paths.csv`: Example generated dataset.

## Quickstart

### 1) Setup

```bash
uv sync
```

### 2) Run the main demo

`main.py` calls `main(n, seed)` to:

- Create one deterministic random grid.
- Solve it with snake, spiral, and random walk strategies.
- Print path overlap stats.
- Attempt to launch Tkinter windows for grid/path visualization.

Run it:

```bash
uv run main.py 12 7
```

### 3) Run the simulation pipeline

`simulate.py` calls `simulate()` (CLI entrypoint), which uses `create_labelled_paths(count, size, output_path)` to:

- Generate many deterministic random grids.
- Solve each grid with snake, spiral, and random walk strategies.
- Save the move sequences as CSV rows.

Run it:

```bash
uv run simulate.py 100 12 --output data/labelled_paths.csv
```

Output CSV columns:

- `snake_path`
- `spiral_path`
- `random_walk_path`

Path strings use only: `u`, `d`, `l`, `r`.

## 4) Use the classifier

Evaluate model performance on a train/test split:

```bash
uv run classifier.py eval
```

Classify a single path string:

```bash
uv run classifier.py classify --path "dddddddddruuuuuuuuurdddddddd"
```

## 5) Train the simple RL baseline

The repository now includes a second RL path meant to stay easy to reason about.

- Fixed grid size only
- Connected random grids from `src/grid.py`
- Flat observation: open cells, visited cells, current position, remaining-steps ratio, no-progress ratio, last action
- Simple reward landscape: new-cell reward, revisit penalty, reverse-edge penalty, completion bonus, and stall termination penalty, with invalid moves masked out
- Invalid moves prevented with action masking
- Plain `sb3_contrib.MaskablePPO` with `MlpPolicy`
- Training speedups: in-place observations, cached grid pools, parallel envs

Train a small baseline:

```bash
uv run src/simple_rl_solver/train.py --size 6 --timesteps 50000 --output data/rl/simple_maskable_ppo_6x6.zip
```

Higher-throughput example:

```bash
uv run src/simple_rl_solver/train.py --size 6 --timesteps 50000 --n-envs 8 --n-steps 256 --n-epochs 2 --batch-size 256 --grid-pool-size 64 --device cpu --output data/rl/simple_maskable_ppo_6x6_fast.zip
```

Benchmark env and PPO throughput:

```bash
uv run src/simple_rl_solver/benchmark.py --mode all --size 8 --grid-pool-size 64 --timesteps 4096
```

Resume training from saved checkpoint with same max grid size:

```bash
uv run src/simple_rl_solver/train.py --size 6 --resume data/rl/simple_maskable_ppo_6x6.zip --timesteps 25000 --output data/rl/simple_maskable_ppo_6x6_resumed.zip
```

Resume notes:

- resume requires same grid size
- updated env settings like reward values or removed-cell range are applied to resumed training env
- saved PPO optimizer/algo hyperparameters remain on loaded checkpoint; new CLI learning knobs are not reapplied automatically

Evaluate it:

```bash
uv run src/simple_rl_solver/eval.py --model data/rl/simple_maskable_ppo_6x6.zip --size 6 --seeds 25
```

Visualize one rollout:

```bash
uv run src/simple_rl_solver/visualize.py --model data/rl/simple_maskable_ppo_6x6.zip --seed 7
```

## Notes

- Tkinter windows may not open in headless environments; CLI output still runs.
- Use the same seed to reproduce the same generated grid for a given size.
- Interactive exact solves now print periodic progress snapshots to the terminal while the branch-and-bound search is running.
- Running `src/optimal_solver.py` with path visualization enabled now redraws the Tk window with the current best-so-far exact path during the search.
- The browser UI also streams those exact-solver updates and redraws the best-so-far path on the grid while the search is in progress.
- The browser UI includes a `memetic_ga` solver option for faster near-optimal paths without running the full exact search.
- `optimal_solver()` is exact for the fixed start chosen by `find_start()`, but it is still exponential. It is most useful for smaller grids, solver benchmarking, and validating heuristic output.
- The older `src/rl_solver/` PPO stack is still present. The new `src/simple_rl_solver/` package is a separate baseline rather than a rewrite of that stack.
