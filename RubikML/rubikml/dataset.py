import random
import torch
from torch.utils.data import IterableDataset

from .cube import Cube
from .encoding import encode_one_hot
from .moves import inverse_move, move_to_index, random_scramble


class RubikScrambleDataset(IterableDataset):
    def __init__(
        self,
        min_depth: int,
        max_depth: int,
        samples_per_epoch: int,
        seed: int | None = None,
        value_norm: int = 26,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.value_norm = value_norm

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.samples_per_epoch):
            depth = rng.randint(self.min_depth, self.max_depth)
            scramble = random_scramble(depth, rng=rng)
            cube = Cube.solved()
            cube.apply_many(scramble)
            x = encode_one_hot(cube.as_stickers())
            y = move_to_index(inverse_move(scramble[-1]))
            # Fixed-scale value target so the head scale stays stable across
            # curriculum phases (value_norm should never change between runs).
            v = float(depth) / float(self.value_norm)
            yield (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(v, dtype=torch.float32),
            )
