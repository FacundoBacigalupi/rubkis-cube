import random

MOVES = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

_INVERSE = {
    "R": "R'", "R'": "R",
    "L": "L'", "L'": "L",
    "U": "U'", "U'": "U",
    "D": "D'", "D'": "D",
    "F": "F'", "F'": "F",
    "B": "B'", "B'": "B",
}

_MOVE_TO_INDEX = {m: i for i, m in enumerate(MOVES)}


def inverse_move(move: str) -> str:
    return _INVERSE[move]


def move_to_index(move: str) -> int:
    return _MOVE_TO_INDEX[move]


def index_to_move(index: int) -> str:
    return MOVES[index]


# Group moves by rotation axis — matches Unity's RubikShuffler constraint.
_AXIS = {
    "R": "X", "R'": "X", "L": "X", "L'": "X",
    "U": "Y", "U'": "Y", "D": "Y", "D'": "Y",
    "F": "Z", "F'": "Z", "B": "Z", "B'": "Z",
}


def random_scramble(
    depth: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a random scramble that matches Unity's shuffler:
    each move must be on a different axis than the previous move."""
    _rng = rng if rng is not None else random
    scramble: list[str] = []
    for _ in range(depth):
        if scramble:
            last_axis = _AXIS[scramble[-1]]
            candidates = [m for m in MOVES if _AXIS[m] != last_axis]
        else:
            candidates = list(MOVES)
        scramble.append(_rng.choice(candidates))
    return scramble
