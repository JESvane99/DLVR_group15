from dataclasses import dataclass


@dataclass
class RunConfig:
    name: str
    seed: int
    lr: float
    optimizer_momentum: float
    epochs: int
