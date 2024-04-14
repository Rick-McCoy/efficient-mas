from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    paths: list[Path] = field(default_factory=list)
