import numpy as np

from qiskit_cayley_codes.utils import (
    f2_nullspace_basis,
    f2_rank,
    hamming_weight,
    int_to_vector,
    vector_to_int,
)


def test_vector_int_roundtrip():
    n = 6
    for x in range(1 << n):
        v = int_to_vector(x, n)
        assert vector_to_int(v) == x


def test_hamming_weight():
    assert hamming_weight(0) == 0
    assert hamming_weight(0b1011) == 3
    assert hamming_weight((1 << 10) - 1) == 10


def test_f2_rank_identity():
    identity = np.eye(4, dtype=np.uint8)
    assert f2_rank(identity) == 4


def test_f2_rank_dependent_rows():
    m = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)  # row3 = row1 xor row2
    assert f2_rank(m) == 2


def test_f2_rank_zero_matrix():
    m = np.zeros((3, 3), dtype=np.uint8)
    assert f2_rank(m) == 0


def test_f2_nullspace_dimension_matches_rank_nullity():
    m = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    basis = f2_nullspace_basis(m)
    rank = f2_rank(m)
    assert basis.shape[0] == m.shape[1] - rank
    # every basis vector should actually be in the null space
    for vec in basis:
        result = (m.astype(int) @ vec.astype(int)) % 2
        assert not result.any()
