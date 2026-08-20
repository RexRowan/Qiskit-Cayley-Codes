"""Linear algebra utilities over F_2, and vector <-> integer encoding.

F_2^n vectors are represented two ways depending on context:
  - as numpy arrays of {0,1} of shape (n,), for readability and matrix work
  - as plain Python ints (bitmasks), for fast Cayley-graph vertex labels

Both representations use the convention that bit/coordinate 0 is the
least-significant bit.
"""

from __future__ import annotations

import numpy as np


def vector_to_int(v: np.ndarray) -> int:
    """Encode an F_2^n vector (array of 0/1) as an integer bitmask."""
    out = 0
    for i, bit in enumerate(v):
        if bit:
            out |= 1 << i
    return out


def int_to_vector(x: int, n: int) -> np.ndarray:
    """Decode an integer bitmask into an F_2^n vector of length n."""
    return np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)


def hamming_weight(x: int) -> int:
    """Hamming weight of an integer bitmask (number of set bits)."""
    return x.bit_count()


def f2_rank(matrix: np.ndarray) -> int:
    """Rank of a 0/1 matrix over F_2, via Gaussian elimination.

    Parameters
    ----------
    matrix : np.ndarray
        2D array of 0/1 entries (any int dtype).

    Returns
    -------
    int
        Rank of the matrix over the field with two elements.
    """
    m = matrix.astype(np.uint8).copy() % 2
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def f2_nullspace_basis(matrix: np.ndarray) -> np.ndarray:
    """Basis for the null space of a 0/1 matrix over F_2.

    Returns an array of shape (k, n_cols) whose rows form a basis for
    {x in F_2^n_cols : matrix @ x = 0 (mod 2)}.
    """
    m = matrix.astype(np.uint8).copy() % 2
    rows, cols = m.shape
    pivots = []
    r = 0
    for col in range(cols):
        pivot = None
        for row in range(r, rows):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        m[[r, pivot]] = m[[pivot, r]]
        for row in range(rows):
            if row != r and m[row, col]:
                m[row] ^= m[r]
        pivots.append(col)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivots]
    basis = []
    for free in free_cols:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[free] = 1
        for i, pcol in enumerate(pivots):
            vec[pcol] = m[i, free]
        basis.append(vec)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, cols), dtype=np.uint8)
