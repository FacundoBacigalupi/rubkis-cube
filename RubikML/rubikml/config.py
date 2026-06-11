from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    min_depth: int = 1
    max_depth: int = 3
    samples_per_epoch: int = 50_000
    val_samples: int = 5_000
    batch_size: int = 512
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    val_seed: int = 0
    model_type: str = "policy"  # "policy" | "policy_value" | "policy_value_v2"
    value_weight: float = 0.5
    # Fixed denominator for value targets across all training phases.
    # v = depth / value_norm — must not change between curriculum phases.
    value_norm: int = 26
    resume: str | None = None  # path to checkpoint to resume from
