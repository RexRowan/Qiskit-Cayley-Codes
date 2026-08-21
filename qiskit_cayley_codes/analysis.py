"""Analysis tools built on top of the CDZ construction:

- ``theorem16_lower_bound``: the paper's own general lower bound on
  quantum code distance (Theorem 16), for cases too large to brute
  force exactly.
- ``search_generator_sets``: a search over candidate generator sets,
  using the zero-sumfree tooling as a girth proxy, ranked by rate.
- ``compare_to_known_families``: a rough benchmark against toric and
  planar surface code parameters at comparable length.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import numpy as np

from .construction import construct_cdz_code
from .utils import f2_nullspace_basis, vector_to_int
from .zero_sumfree import is_zero_sumfree


def classical_min_distance_bruteforce(H: np.ndarray, max_kernel_dim: int = 20) -> int:
    """Exact minimum distance of the classical code with parity check H.

    Codewords are the nonzero elements of Ker(H) (mod 2). Cost is
    O(2^dim(Ker(H))), so only practical when that dimension is small
    -- which is the case for the C(W) codes used in
    ``theorem16_lower_bound`` below, since dim(Ker(M(W))) = w, the
    number of *extra* generators, not the (potentially large) ambient
    dimension m.

    Raises
    ------
    ValueError
        If Ker(H) is trivial (no nonzero codewords), or if
        dim(Ker(H)) exceeds ``max_kernel_dim``.
    """
    kernel_basis = f2_nullspace_basis(H)
    k_dim = kernel_basis.shape[0]
    if k_dim == 0:
        raise ValueError("Parity check matrix has trivial kernel: no nonzero codewords.")
    if k_dim > max_kernel_dim:
        raise ValueError(
            f"Ker(H) has dimension {k_dim}; brute-force search costs "
            f"O(2^{k_dim}) and is impractical beyond max_kernel_dim={max_kernel_dim}."
        )
    n_cols = H.shape[1]
    best = None
    for bits in range(1, 1 << k_dim):
        word = np.zeros(n_cols, dtype=np.uint8)
        for i in range(k_dim):
            if (bits >> i) & 1:
                word ^= kernel_basis[i]
        weight = int(word.sum())
        if best is None or weight < best:
            best = weight
    return best


def theorem16_lower_bound(m: int, w_generators: Iterable[np.ndarray]) -> dict:
    """Apply Couvreur-Delfosse-Zemor's Theorem 16 quantum distance bound.

    Setup (paper Sec. 4.4 / 5): generators are the standard basis of
    F_2^m together with w extra vectors W (each length-m, distinct,
    nonzero, not already in the standard basis). This defines the
    classical code C(W) with parity check matrix M(W) = [I_m | P(W)]
    (length n = m + w), where P(W)'s columns are the vectors of W.
    Writing d for the exact minimum distance of C(W), Theorem 16 gives:

        D >= d * n^2 / 640,   valid for d >= 9.

    This bound only requires computing d for the *small* code C(W)
    (length n = m+w, dimension w), not brute-forcing the 2^m-qubit
    quantum code directly -- that's the point of the theorem.

    Parameters
    ----------
    m : int
        Ambient dimension for the standard-basis part of the generators.
    w_generators : Iterable[np.ndarray]
        The extra generators W, each a length-m 0/1 array.

    Returns
    -------
    dict with keys 'm', 'w', 'n', 'd', 'bound'.

    Raises
    ------
    ValueError
        If w == 0, or if the resulting classical distance d < 9 (outside
        the theorem's stated range).
    """
    W = [np.asarray(v, dtype=np.uint8) % 2 for v in w_generators]
    w = len(W)
    if w == 0:
        raise ValueError("Theorem 16 requires w > 0 (at least one extra generator).")

    n = m + w
    identity = np.eye(m, dtype=np.uint8)
    P_W = np.stack(W, axis=1)  # m x w
    M_W = np.hstack([identity, P_W])  # m x n, the parity check matrix of C(W)

    d = classical_min_distance_bruteforce(M_W)
    if d < 9:
        raise ValueError(
            f"Theorem 16 requires the classical code's minimum distance d >= 9; "
            f"got d={d}. The bound does not apply for this (m, W)."
        )
    bound = d * n**2 / 640
    return {"m": m, "w": w, "n": n, "d": d, "bound": bound}


def search_generator_sets(
    m: int,
    candidate_pool: Iterable[np.ndarray],
    w: int,
    max_candidates: int | None = None,
    require_zero_sumfree: bool = True,
) -> list:
    """Search combinations of w extra generators for good CDZ codes.

    For each size-w combination W drawn from ``candidate_pool``, checks
    whether S = standard_basis(m) + W is a valid generating set (even
    total size -- see ``construction.py``) and, optionally, zero-sumfree
    (used here as a girth proxy: a zero-sumfree S has no short additive
    relation among its generators, which controls short cycles in the
    Cayley graph). Builds each valid candidate's code via
    ``construct_cdz_code`` and records (N, K, rate).

    This does NOT compute quantum distance -- at the scale where a
    search over many candidates is useful, N = 2^m is generally too
    large to brute force. Combine promising candidates with
    ``theorem16_lower_bound`` separately to get a distance guarantee.

    Returns
    -------
    list of dicts (one per valid, buildable candidate), each with keys
    'generators' (the extra generators, as ints), 'N', 'K', 'rate' =
    K/N, sorted by descending rate.
    """
    pool = [np.asarray(v, dtype=np.uint8) % 2 for v in candidate_pool]
    standard_basis = [np.eye(m, dtype=np.uint8)[i] for i in range(m)]

    results = []
    considered = 0
    for combo in combinations(range(len(pool)), w):
        if max_candidates is not None and considered >= max_candidates:
            break
        considered += 1

        extra = [pool[i] for i in combo]
        candidate_set = standard_basis + extra
        if len(candidate_set) % 2 != 0:
            continue
        if require_zero_sumfree and not is_zero_sumfree(candidate_set):
            continue

        try:
            code = construct_cdz_code(m, candidate_set)
        except ValueError:
            continue

        N, K = code.parameters()
        results.append(
            {
                "generators": [vector_to_int(g) for g in extra],
                "N": N,
                "K": K,
                "rate": (K / N) if N else 0.0,
            }
        )

    results.sort(key=lambda r: r["rate"], reverse=True)
    return results


def toric_code_params(L: int) -> dict:
    """Toric code on an L x L lattice: parameters [[2L^2, 2, L]]."""
    return {"name": "toric", "N": 2 * L**2, "K": 2, "D": L}


def surface_code_params(d: int) -> dict:
    """Planar surface code of distance d: parameters [[d^2 + (d-1)^2, 1, d]]."""
    return {"name": "surface", "N": d**2 + (d - 1) ** 2, "K": 1, "D": d}


def compare_to_known_families(N: int, D: int | None = None, search_range: int = 500) -> dict:
    """Compare a CDZ code's length (and optionally known distance) against
    toric and planar surface codes of the closest comparable length.

    This is a rough benchmarking aid, not a rigorous equivalence: toric
    and surface codes are topological constructions with very different
    structure (constant stabilizer weight, geometric locality) from the
    CDZ family (logarithmic stabilizer weight, no locality constraint),
    so a fair comparison also has to weigh those structural differences,
    not just (N, K, D).

    Returns
    -------
    dict with keys 'cdz', 'toric', 'surface'.
    """
    best_toric_L = min(range(1, search_range), key=lambda L: abs(toric_code_params(L)["N"] - N))
    best_surface_d = min(range(2, search_range), key=lambda d: abs(surface_code_params(d)["N"] - N))
    return {
        "cdz": {"N": N, "D": D},
        "toric": toric_code_params(best_toric_L),
        "surface": surface_code_params(best_surface_d),
    }
