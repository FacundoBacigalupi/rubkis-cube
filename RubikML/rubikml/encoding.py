import numpy as np


def encode_one_hot(stickers: np.ndarray) -> np.ndarray:
    """
    Input:  shape (54,), values 0..5
    Output: shape (324,), dtype float32
    """
    out = np.zeros(54 * 6, dtype=np.float32)
    for i, color in enumerate(stickers):
        out[i * 6 + int(color)] = 1.0
    return out


def decode_one_hot(encoded: np.ndarray) -> np.ndarray:
    """shape (324,) -> shape (54,) int8"""
    return encoded.reshape(54, 6).argmax(axis=1).astype(np.int8)
