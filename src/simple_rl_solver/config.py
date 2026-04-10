from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


ACTION_ORDER = ("u", "d", "l", "r")


@dataclass(slots=True)
class EnvConfig:
    size: int = 10
    removed_fraction_min: float = 0.18
    removed_fraction_max: float = 0.42
    reward_new_cell: float = 1.0
    reward_revisit: float = -0.3
    reward_reverse_edge: float = -0.75
    reward_complete: float = 5.0
    stall_grace_steps: int = 8
    reward_stall_terminate_scale: float = -10.0
    max_steps_factor: float = 2.0
    grid_pool_size: int = 1024
    base_seed: int = 0


@dataclass(slots=True)
class TrainConfig:
    total_timesteps: int = 100_000
    n_envs: int = 16
    n_steps: int = 256
    n_epochs: int = 2
    learning_rate: float = 1e-3
    gamma: float = 0.995
    batch_size: int = 64
    ent_coef: float = 0.02
    seed: int = 0
    device: str = "auto"


def _filter_config_payload(data: dict, config_type: type) -> dict:
    allowed = {field.name for field in fields(config_type)}
    return {key: value for key, value in data.items() if key in allowed}


def make_env_config(data: dict) -> EnvConfig:
    return EnvConfig(**_filter_config_payload(data, EnvConfig))


def make_train_config(data: dict) -> TrainConfig:
    return TrainConfig(**_filter_config_payload(data, TrainConfig))


def env_configs_resume_compatible(saved: EnvConfig, current: EnvConfig) -> bool:
    return saved.size == current.size


def validate_env_config(config: EnvConfig) -> None:
    if config.size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 <= config.removed_fraction_min <= config.removed_fraction_max < 1.0:
        raise ValueError("removed_fraction range must satisfy 0 <= min <= max < 1")
    if config.max_steps_factor <= 0.0:
        raise ValueError("max_steps_factor must be positive")
    if config.stall_grace_steps < 0:
        raise ValueError("stall_grace_steps must be non-negative")
    if config.grid_pool_size < 0:
        raise ValueError("grid_pool_size must be non-negative")


def validate_train_config(config: TrainConfig) -> None:
    if config.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < config.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.ent_coef < 0.0:
        raise ValueError("ent_coef must be non-negative")


def resolve_train_config(train_config: TrainConfig) -> TrainConfig:
    validate_train_config(train_config)
    rollout_batch_size = train_config.n_envs * train_config.n_steps
    batch_size = min(train_config.batch_size, rollout_batch_size)
    return TrainConfig(
        total_timesteps=train_config.total_timesteps,
        n_envs=train_config.n_envs,
        n_steps=train_config.n_steps,
        n_epochs=train_config.n_epochs,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        batch_size=batch_size,
        ent_coef=train_config.ent_coef,
        seed=train_config.seed,
        device=train_config.device,
    )


def save_run_config(
    model_path: str | Path,
    env_config: EnvConfig,
    train_config: TrainConfig,
    *,
    trained_total_timesteps: int | None = None,
) -> Path:
    output_path = Path(model_path)
    config_path = output_path.with_suffix(".config.json")
    payload = {
        "env": asdict(env_config),
        "train": asdict(train_config),
        "action_order": list(ACTION_ORDER),
    }
    if trained_total_timesteps is not None:
        payload["trained_total_timesteps"] = int(trained_total_timesteps)
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def load_run_config(model_path: str | Path) -> dict:
    output_path = Path(model_path)
    config_path = output_path.with_suffix(".config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))
