import numpy as np
import pytest
from rubikml.cube import Cube
from rubikml.moves import MOVES, inverse_move, random_scramble


def _color_counts(cube: Cube) -> dict:
    stickers = cube.as_stickers()
    return {c: int(np.sum(stickers == c)) for c in range(6)}


def test_solved_state():
    cube = Cube.solved()
    assert cube.is_solved()


def test_color_counts_solved():
    counts = _color_counts(Cube.solved())
    for c in range(6):
        assert counts[c] == 9


@pytest.mark.parametrize("move", MOVES)
def test_move_and_inverse_returns_to_solved(move):
    cube = Cube.solved()
    cube.apply(move)
    cube.apply(inverse_move(move))
    assert cube.is_solved(), f"move {move} + inverse did not return to solved"


@pytest.mark.parametrize("move", MOVES)
def test_four_turns_identity(move):
    cube = Cube.solved()
    for _ in range(4):
        cube.apply(move)
    assert cube.is_solved(), f"4x {move} did not return to solved"


@pytest.mark.parametrize("move", MOVES)
def test_color_counts_stable_after_move(move):
    cube = Cube.solved()
    cube.apply(move)
    counts = _color_counts(cube)
    for c in range(6):
        assert counts[c] == 9, f"color {c} count changed after {move}"


def test_scramble_inverse_returns_to_solved():
    import random
    random.seed(42)
    for depth in [1, 3, 5, 10]:
        scramble = random_scramble(depth)
        cube = Cube.solved()
        cube.apply_many(scramble)
        inverse_seq = [inverse_move(m) for m in reversed(scramble)]
        cube.apply_many(inverse_seq)
        assert cube.is_solved(), f"scramble depth {depth} inverse failed"


def test_color_counts_after_scramble():
    import random
    random.seed(99)
    for _ in range(20):
        depth = random.randint(1, 15)
        scramble = random_scramble(depth)
        cube = Cube.solved()
        cube.apply_many(scramble)
        counts = _color_counts(cube)
        for c in range(6):
            assert counts[c] == 9


def test_clone_is_independent():
    cube = Cube.solved()
    clone = cube.clone()
    cube.apply("R")
    assert clone.is_solved()
