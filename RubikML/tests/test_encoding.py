import numpy as np
import pytest
from rubikml.cube import Cube
from rubikml.encoding import encode_one_hot, decode_one_hot
from rubikml.moves import random_scramble
import random


def test_encoded_shape():
    stickers = Cube.solved().as_stickers()
    enc = encode_one_hot(stickers)
    assert enc.shape == (324,)


def test_encoded_dtype():
    enc = encode_one_hot(Cube.solved().as_stickers())
    assert enc.dtype == np.float32


def test_each_group_sums_to_one():
    stickers = Cube.solved().as_stickers()
    enc = encode_one_hot(stickers)
    groups = enc.reshape(54, 6)
    sums = groups.sum(axis=1)
    np.testing.assert_allclose(sums, np.ones(54))


def test_decode_roundtrip():
    stickers = Cube.solved().as_stickers()
    enc = encode_one_hot(stickers)
    decoded = decode_one_hot(enc)
    np.testing.assert_array_equal(decoded, stickers)


def test_encoding_after_scramble():
    random.seed(7)
    for _ in range(10):
        cube = Cube.solved()
        cube.apply_many(random_scramble(random.randint(1, 10)))
        stickers = cube.as_stickers()
        enc = encode_one_hot(stickers)
        assert enc.shape == (324,)
        groups = enc.reshape(54, 6)
        np.testing.assert_allclose(groups.sum(axis=1), np.ones(54))
        np.testing.assert_array_equal(decode_one_hot(enc), stickers)


def test_encoding_is_deterministic():
    cube = Cube.solved()
    cube.apply("R")
    cube.apply("U")
    s = cube.as_stickers()
    assert np.array_equal(encode_one_hot(s), encode_one_hot(s))
