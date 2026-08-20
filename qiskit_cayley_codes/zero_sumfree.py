"""ℓ-zero-sumfree sets over F_2^n and their role in generator selection.

A subset S of an abelian group G is *zero-sumfree* if no nonempty
subset of S sums to the identity. The ℓ-bounded variant restricts
attention to subsets of size at most ℓ. These notions connect to the
Davenport constant D(G): the smallest d such that every sequence of d
elements of G has a nonempty zero-sum subsequence.

Here, that theory is used to choose generating sets S for the Cayley
graph construction in ``construction.py``: a zero-sumfree S has no
"short circuit" among its generators, which controls the girth of
Cay(F_2^n, S) and, in turn, quantities relevant to the minimum distance
of the resulting CSS code. See the project README for the precise
statement connecting this repo's original G_k(n) / zero-sumfree work
to the CDZ construction.

NOTE: this module implements the finite-group combinatorics generically.
It does not itself compute code distances -- that link is made in
``construction.py`` / the accompanying research notes.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np

from .utils import vector_to_int


def is_zero_sumfree(generators: Iterable[np.ndarray], ell: int | None = None) -> bool:
    """Check whether a generating set is (ℓ-bounded) zero-sumfree over F_2^n.

    Parameters
    ----------
    generators : Iterable[np.ndarray]
        Candidate generating set S, as length-n 0/1 arrays.
    ell : int, optional
        If given, only subsets of size <= ell are checked (ℓ-bounded
        zero-sumfree). If None, all nonempty subsets are checked.

    Returns
    -------
    bool
        True if no nonempty (bounded) subset of S sums to zero over F_2^n
        (i.e. XORs to the all-zero vector).

    Notes
    -----
    This is exponential in |S| in the worst case, as it must be for an
    exact zero-sumfree check. Use ``ell`` to bound the search when |S|
    is large and only short zero-sum relations matter (e.g. because
    they are the ones that affect girth / distance at the code sizes
    of interest).
    """
    gens = [vector_to_int(np.asarray(g) % 2) for g in generators]
    m = len(gens)
    max_size = m if ell is None else min(ell, m)

    for size in range(1, max_size + 1):
        for combo in combinations(gens, size):
            acc = 0
            for g in combo:
                acc ^= g
            if acc == 0:
                return False
    return True


def find_zero_sum_relations(
    generators: Iterable[np.ndarray], ell: int | None = None
) -> list[list[int]]:
    """Return all nonempty (bounded) subsets of ``generators`` that sum to zero.

    Each relation is returned as a list of indices into ``generators``.
    Useful for diagnosing why a candidate generating set failed
    ``is_zero_sumfree``, and for locating the short cycles they induce
    in the corresponding Cayley graph.
    """
    gens = [vector_to_int(np.asarray(g) % 2) for g in generators]
    m = len(gens)
    max_size = m if ell is None else min(ell, m)

    relations = []
    for size in range(1, max_size + 1):
        for combo in combinations(range(m), size):
            acc = 0
            for idx in combo:
                acc ^= gens[idx]
            if acc == 0:
                relations.append(list(combo))
    return relations


def davenport_constant_upper_bound(n: int) -> int:
    """Upper bound on the Davenport constant D(F_2^n).

    For an elementary abelian 2-group of rank n, D(F_2^n) = n + 1
    (this is exact, not just an upper bound, for elementary abelian
    2-groups -- named this way for API consistency with the broader
    G_k(n) research where only bounds are available in general).

    A generating set of size >= D(F_2^n) is guaranteed to contain a
    zero-sum subset, so effective generator search for the CDZ
    construction is confined to |S| < n + 1 when an exactly
    zero-sumfree set is required.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return n + 1
