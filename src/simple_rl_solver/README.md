# Simple RL Solver

This package is the intentionally small RL baseline for the lawn grid.

## Design

- Fixed grid size only
- Connected random grids from `src/grid.py`
- Flat observation vector
- No curriculum
- No behavior cloning
- Invalid moves prevented with action masking
- Plain `sb3_contrib.MaskablePPO` with `MlpPolicy`
- In-place observation buffer to cut per-step allocations
- Cached grid pool to cut reset cost
- Vectorized training via parallel environments

## Observation

The observation is a flat vector containing:

- open-cell mask
- visited-cell mask
- current-position mask
- remaining-steps ratio
- steps-since-progress ratio
- last-action one-hot

## Reward

- `+1.0` for visiting a new open cell
- `-0.3` for revisiting a cell
- `-0.75` for immediately reversing onto previous cell
- `+5.0` completion bonus
- uncovered-area penalty when no new cell is found for too many steps

## Entrypoints

- `src/simple_rl_solver/train.py`
- `src/simple_rl_solver/eval.py`
- `src/simple_rl_solver/visualize.py`

## Resume

Training can continue from saved checkpoint with `--resume`.

- resume requires same grid size
- updated reward or grid-generation settings may be applied while resuming
- saved PPO optimizer/algo hyperparameters stay on loaded checkpoint

## Speed Knobs

- `--n-envs`: parallel training env count
- `--n-steps`: rollout steps per env
- `--n-epochs`: PPO epochs per rollout
- `--grid-pool-size`: cached generated grids per env
- `--device`: training device

Current defaults favor throughput:

- `n_envs=8`
- `grid_pool_size=1024`

## Benchmarking

Run env and PPO throughput benchmarks:

```bash
uv run src/simple_rl_solver/benchmark.py --mode all --size 8 --grid-pool-size 64 --timesteps 4096
```
