import numpy as np
from copy import copy

# Face indices in the state array (each face = 9 stickers)
U, D, R, L, F, B = 0, 1, 2, 3, 4, 5
_FACE_OFFSET = [U*9, D*9, R*9, L*9, F*9, B*9]

# Sticker index helpers
def _f(face, row, col):
    return face * 9 + row * 3 + col


class Cube:
    __slots__ = ("_s",)

    def __init__(self, state: np.ndarray):
        self._s = state

    @staticmethod
    def solved() -> "Cube":
        s = np.empty(54, dtype=np.int8)
        for color in range(6):
            s[color*9:(color+1)*9] = color
        return Cube(s)

    def clone(self) -> "Cube":
        return Cube(self._s.copy())

    def is_solved(self) -> bool:
        for color in range(6):
            if not np.all(self._s[color*9:(color+1)*9] == color):
                return False
        return True

    def as_stickers(self) -> np.ndarray:
        return self._s.copy()

    def apply(self, move: str) -> None:
        _MOVE_FN[move](self._s)

    def apply_many(self, moves: list) -> None:
        for m in moves:
            _MOVE_FN[m](self._s)


# --- low-level move helpers ---

def _cycle4(s, a, b, c, d):
    tmp = s[a]
    s[a] = s[d]
    s[d] = s[c]
    s[c] = s[b]
    s[b] = tmp


def _rotate_face_cw(s, face):
    o = face * 9
    tmp = s[o+0]
    s[o+0] = s[o+6]
    s[o+6] = s[o+8]
    s[o+8] = s[o+2]
    s[o+2] = tmp
    tmp = s[o+1]
    s[o+1] = s[o+3]
    s[o+3] = s[o+7]
    s[o+7] = s[o+5]
    s[o+5] = tmp


def _rotate_face_ccw(s, face):
    _rotate_face_cw(s, face)
    _rotate_face_cw(s, face)
    _rotate_face_cw(s, face)


# R move: right face clockwise
def _R(s):
    _rotate_face_cw(s, R)
    # U col2 -> B col0 (reversed) -> D col2 -> F col2
    for i in range(3):
        tmp = s[_f(U, i, 2)]
        s[_f(U, i, 2)] = s[_f(F, i, 2)]
        s[_f(F, i, 2)] = s[_f(D, i, 2)]
        s[_f(D, i, 2)] = s[_f(B, 2-i, 0)]
        s[_f(B, 2-i, 0)] = tmp


def _R_prime(s):
    _R(s); _R(s); _R(s)


# L move: left face clockwise
def _L(s):
    _rotate_face_cw(s, L)
    for i in range(3):
        tmp = s[_f(U, i, 0)]
        s[_f(U, i, 0)] = s[_f(B, 2-i, 2)]
        s[_f(B, 2-i, 2)] = s[_f(D, i, 0)]
        s[_f(D, i, 0)] = s[_f(F, i, 0)]
        s[_f(F, i, 0)] = tmp


def _L_prime(s):
    _L(s); _L(s); _L(s)


# U move: up face clockwise
def _U(s):
    _rotate_face_cw(s, U)
    for i in range(3):
        tmp = s[_f(F, 0, i)]
        s[_f(F, 0, i)] = s[_f(R, 0, i)]
        s[_f(R, 0, i)] = s[_f(B, 0, i)]
        s[_f(B, 0, i)] = s[_f(L, 0, i)]
        s[_f(L, 0, i)] = tmp


def _U_prime(s):
    _U(s); _U(s); _U(s)


# D move: down face clockwise
def _D(s):
    _rotate_face_cw(s, D)
    for i in range(3):
        tmp = s[_f(F, 2, i)]
        s[_f(F, 2, i)] = s[_f(L, 2, i)]
        s[_f(L, 2, i)] = s[_f(B, 2, i)]
        s[_f(B, 2, i)] = s[_f(R, 2, i)]
        s[_f(R, 2, i)] = tmp


def _D_prime(s):
    _D(s); _D(s); _D(s)


# F move: front face clockwise
def _F(s):
    _rotate_face_cw(s, F)
    for i in range(3):
        tmp = s[_f(U, 2, i)]
        s[_f(U, 2, i)] = s[_f(L, 2-i, 2)]
        s[_f(L, 2-i, 2)] = s[_f(D, 0, 2-i)]
        s[_f(D, 0, 2-i)] = s[_f(R, i, 0)]
        s[_f(R, i, 0)] = tmp


def _F_prime(s):
    _F(s); _F(s); _F(s)


# B move: back face clockwise
def _B(s):
    _rotate_face_cw(s, B)
    for i in range(3):
        tmp = s[_f(U, 0, i)]
        s[_f(U, 0, i)] = s[_f(R, i, 2)]
        s[_f(R, i, 2)] = s[_f(D, 2, 2-i)]
        s[_f(D, 2, 2-i)] = s[_f(L, 2-i, 0)]
        s[_f(L, 2-i, 0)] = tmp


def _B_prime(s):
    _B(s); _B(s); _B(s)


_MOVE_FN = {
    "R": _R, "R'": _R_prime,
    "L": _L, "L'": _L_prime,
    "U": _U, "U'": _U_prime,
    "D": _D, "D'": _D_prime,
    "F": _F, "F'": _F_prime,
    "B": _B, "B'": _B_prime,
}
