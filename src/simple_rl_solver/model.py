from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path as FilePath
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from src.simple_rl_solver.config import (
    EnvConfig,
    TrainConfig,
    env_configs_resume_compatible,
    load_run_config,
    make_env_config,
    make_train_config,
    resolve_train_config,
    save_run_config,
    validate_env_config,
)
from src.simple_rl_solver.env import SimpleLawnMowingEnv
from src.shared_types import Grid, Path


class TimeIntervalLoggerCallback(BaseCallback):
    def __init__(self, *, interval_seconds: float = 5.0):
        super().__init__()
        self.interval_seconds = interval_seconds
        self.started_at = 0.0
        self.start_num_timesteps = 0
        self.next_log_at = 0.0
        self.recent_episode_rewards: deque[float] = deque(maxlen=50)
        self.recent_episode_coverages: deque[float] = deque(maxlen=50)
        self.recent_episode_lengths: deque[int] = deque(maxlen=50)

    def _on_training_start(self) -> None:
        self.started_at = time.monotonic()
        self.start_num_timesteps = self.num_timesteps
        self.next_log_at = self.started_at + self.interval_seconds
        print(
            "Learning params: "
            f"learning_rate={self.model.learning_rate} "
            f"gamma={self.model.gamma} "
            f"n_steps={self.model.n_steps} "
            f"batch_size={self.model.batch_size} "
            f"ent_coef={self.model.ent_coef} "
            f"n_epochs={self.model.n_epochs} "
            f"episode_window={self.recent_episode_rewards.maxlen} "
            f"n_envs={self.training_env.num_envs}"
        )

    def _on_step(self) -> bool:
        for done, info in zip(
            self.locals.get("dones", []),
            self.locals.get("infos", []),
            strict=False,
        ):
            if not done:
                continue

            episode = info.get("episode")
            if episode is not None:
                self.recent_episode_rewards.append(float(episode["r"]))
                self.recent_episode_lengths.append(int(episode["l"]))
            self.recent_episode_coverages.append(float(info.get("coverage_ratio", 0.0)))

        now = time.monotonic()
        if now < self.next_log_at:
            return True

        elapsed_seconds = max(0.0, now - self.started_at)
        session_timesteps = self.num_timesteps - self.start_num_timesteps
        fps = session_timesteps / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        avg_reward = (
            sum(self.recent_episode_rewards) / len(self.recent_episode_rewards)
            if self.recent_episode_rewards
            else 0.0
        )
        avg_coverage = (
            sum(self.recent_episode_coverages) / len(self.recent_episode_coverages)
            if self.recent_episode_coverages
            else 0.0
        )
        avg_ep_len = (
            sum(self.recent_episode_lengths) / len(self.recent_episode_lengths)
            if self.recent_episode_lengths
            else 0.0
        )
        print(
            "Training progress: "
            f"session_timesteps={session_timesteps} "
            f"total_timesteps={self.num_timesteps} "
            f"elapsed={elapsed_seconds:.1f}s "
            f"fps={fps:.0f} "
            f"n_updates={self.model._n_updates} "
            f"avg_reward={avg_reward:.2f} "
            f"avg_coverage={avg_coverage:.3f} "
            f"avg_ep_len={avg_ep_len:.1f}"
        )
        self.next_log_at = now + self.interval_seconds
        return True


def _mask_fn(env: SimpleLawnMowingEnv):
    return env.unwrapped.action_masks()


def make_masked_env(env_config: EnvConfig, *, seed_offset: int = 0) -> ActionMasker:
    validate_env_config(env_config)
    config = replace(env_config, base_seed=env_config.base_seed + seed_offset)
    env = SimpleLawnMowingEnv(config, record_path=False)
    monitored_env = Monitor(env)
    return ActionMasker(monitored_env, _mask_fn)


def build_vec_env(env_config: EnvConfig, train_config: TrainConfig) -> VecEnv:
    env_fns = [
        (lambda offset=offset: make_masked_env(env_config, seed_offset=offset))
        for offset in range(train_config.n_envs)
    ]
    return DummyVecEnv(env_fns)


def _changed_env_fields(saved: EnvConfig, current: EnvConfig) -> list[str]:
    changed: list[str] = []
    for field_name in saved.__dataclass_fields__:
        if getattr(saved, field_name) != getattr(current, field_name):
            changed.append(
                f"{field_name}={getattr(saved, field_name)}->{getattr(current, field_name)}"
            )
    return changed


def _changed_train_fields(saved: TrainConfig, current: TrainConfig) -> list[str]:
    changed: list[str] = []
    for field_name in saved.__dataclass_fields__:
        if getattr(saved, field_name) != getattr(current, field_name):
            changed.append(
                f"{field_name}={getattr(saved, field_name)}->{getattr(current, field_name)}"
            )
    return changed


def _apply_train_config_to_model(
    model: MaskablePPO, train_config: TrainConfig, env: VecEnv
) -> None:
    model.set_env(env)
    model.n_steps = train_config.n_steps
    model.n_envs = env.num_envs
    model.batch_size = train_config.batch_size
    model.n_epochs = train_config.n_epochs
    model.learning_rate = train_config.learning_rate
    model.lr_schedule = FloatSchedule(train_config.learning_rate)
    model.gamma = train_config.gamma
    model.ent_coef = train_config.ent_coef
    model.rollout_buffer = model.rollout_buffer_class(
        model.n_steps,
        model.observation_space,
        model.action_space,
        model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
        **model.rollout_buffer_kwargs,
    )
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = float(train_config.learning_rate)


def _rollout_env_with_model(
    model: MaskablePPO,
    env: SimpleLawnMowingEnv,
    obs,
    *,
    deterministic: bool = True,
) -> Path:
    while True:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=deterministic,
        )
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return env.get_path()


def solve_grid_with_loaded_model(
    model: MaskablePPO,
    env_config: EnvConfig,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> Path:
    if len(grid) > env_config.size:
        raise ValueError(
            f"Model supports grids up to size {env_config.size}, got {len(grid)}"
        )

    env = SimpleLawnMowingEnv(env_config)
    try:
        obs, _ = env.reset(options={"grid": grid})
        return _rollout_env_with_model(model, env, obs, deterministic=deterministic)
    finally:
        env.close()


def train_simple_maskable_ppo(
    env_config: EnvConfig,
    train_config: TrainConfig,
    model_path: str | FilePath,
    *,
    resume_from: str | FilePath | None = None,
    log_interval_seconds: float | None = 5.0,
) -> FilePath:
    validate_env_config(env_config)
    train_config = resolve_train_config(train_config)
    env = build_vec_env(env_config, train_config)
    reset_num_timesteps = True
    if resume_from is not None:
        resume_path = FilePath(resume_from)
        resume_config = load_run_config(resume_path)
        resume_env_config = make_env_config(resume_config["env"])
        resume_train_config = make_train_config(resume_config["train"])
        if not env_configs_resume_compatible(resume_env_config, env_config):
            raise ValueError("Resume checkpoint is only compatible with same grid size")

        changed_fields = _changed_env_fields(resume_env_config, env_config)
        print(
            "Resuming from checkpoint: "
            f"{resume_path} "
            f"saved_size={resume_env_config.size}"
        )
        if changed_fields:
            print(
                "Applying updated env config while resuming: "
                + ", ".join(changed_fields)
            )
        changed_train_fields = _changed_train_fields(resume_train_config, train_config)
        if changed_train_fields:
            print(
                "Applying updated train config while resuming: "
                + ", ".join(changed_train_fields)
            )

        try:
            model = MaskablePPO.load(
                resume_path.with_suffix(""), env=env, device=train_config.device
            )
        except Exception as error:
            raise ValueError(
                "Resume checkpoint is not a simple MaskablePPO model. Old DQN checkpoints cannot be resumed."
            ) from error
        _apply_train_config_to_model(model, train_config, env)
        reset_num_timesteps = False
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            n_steps=train_config.n_steps,
            n_epochs=train_config.n_epochs,
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            batch_size=train_config.batch_size,
            ent_coef=train_config.ent_coef,
            seed=train_config.seed,
            device=train_config.device,
            verbose=0,
        )

    print(
        "Training simple MaskablePPO: "
        f"size={env_config.size} "
        f"n_envs={train_config.n_envs} "
        f"n_steps={train_config.n_steps} "
        f"n_epochs={train_config.n_epochs} "
        f"resume={resume_from or 'none'} "
        f"reset_num_timesteps={int(reset_num_timesteps)} "
        f"removed_fraction=[{env_config.removed_fraction_min:.2f}, {env_config.removed_fraction_max:.2f}] "
        f"max_steps_factor={env_config.max_steps_factor} "
        f"grid_pool_size={env_config.grid_pool_size} "
        f"timesteps={train_config.total_timesteps}"
    )
    try:
        callback = None
        if log_interval_seconds is not None:
            callback = TimeIntervalLoggerCallback(interval_seconds=log_interval_seconds)
        model.learn(
            total_timesteps=train_config.total_timesteps,
            progress_bar=False,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        env.close()

    output_path = FilePath(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    save_run_config(
        output_path,
        env_config,
        train_config,
        trained_total_timesteps=model.num_timesteps,
    )
    return output_path


def load_simple_maskable_ppo(model_path: str | FilePath) -> tuple[MaskablePPO, dict]:
    resolved_model_path = FilePath(model_path)
    sb3_load_path: str | FilePath = resolved_model_path
    if resolved_model_path.suffix == ".zip":
        sb3_load_path = resolved_model_path.with_suffix("")

    try:
        model = MaskablePPO.load(sb3_load_path)
    except Exception as error:
        raise ValueError(
            "Checkpoint is not a simple MaskablePPO model. Old DQN checkpoints are incompatible with masked rollout."
        ) from error
    config = load_run_config(resolved_model_path)
    return model, config


def solve_grid_with_model(
    model_path: str | FilePath,
    grid: Grid,
    *,
    deterministic: bool = True,
) -> Path:
    model, config = load_simple_maskable_ppo(model_path)
    return solve_grid_with_loaded_model(
        model,
        make_env_config(config["env"]),
        grid,
        deterministic=deterministic,
    )
