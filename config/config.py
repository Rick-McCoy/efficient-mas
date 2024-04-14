from dataclasses import dataclass, field

from omegaconf import MISSING, DictConfig

from config.data.config import DataConfig
from config.model.config import AlignerBaseConfig
from config.train.config import TrainConfig

defaults = ["_self_", {"aligner": "large"}, {"data": "base"}, {"train": "base"}]


@dataclass
class Config:
    defaults: list = field(default_factory=lambda: defaults)

    aligner: AlignerBaseConfig = MISSING
    data: DataConfig = MISSING
    train: TrainConfig = MISSING

    voco: str = "mel"
    name: str = "default"


def dict_to_config(cfg: dict | DictConfig):
    aligner_config = AlignerBaseConfig(**cfg["aligner"])
    data_config = DataConfig(**cfg["data"])
    train_config = TrainConfig(**cfg["train"])
    return Config(
        aligner=aligner_config,
        data=data_config,
        train=train_config,
        voco=cfg["voco"],
        name=cfg["name"],
    )
