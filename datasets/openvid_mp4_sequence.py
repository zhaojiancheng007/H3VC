from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSampleKey:
    index: int
    crop_seed: int
