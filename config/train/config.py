from dataclasses import dataclass


@dataclass
class TrainConfig:
    acc: int = 1
    batch_size: int = 4
    compile: bool = False
    early_stop: bool = True
    fast_dev_run: bool = False
    lr: float = 1e-4
    num_workers: int = 8
    optimizer: str = "Adam"
    resume_aligner_path: str | None = None
    scheduler: str = "linear_warmup_decay"
    gradient_clip_val: float = 0.2
    precision: str = "16-mixed"
    project: str = "voicebox"
    mlflow: bool = False
    weight_average: bool = False

    train_len: int = 10000
    val_len: int = 1000
    test_len: int = 1000
    max_epochs: int = 100
