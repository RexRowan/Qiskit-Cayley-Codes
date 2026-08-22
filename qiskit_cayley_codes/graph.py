"""Cayley graph construction over F_2^n.

Cay(F_2^n, S) has vertex set F_2^n and an edge {x, x+s} for every x in
F_2^n and every s in S (S implicitly symmetric since s = -s over F_2).
Qubits in the CSS construction (see ``construction.py``) sit on the
VERTICES of this graph (one per element of F_2^n, so N = 2^n) -- this
graph itself is only the adjacency structure the check matrix is built
from, not a qubit-per-edge layout.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import networkx as nx
import numpy as np

from .utils import int_to_vector, vector_to_int


def cayley_graph(n: int, generators: Iterable[np.ndarray]) -> nx.Graph:
    """Build Cay(F_2^n, S) as a networkx.Graph.

    Parameters
    ----------
    n : int
        Ambient dimension; vertices are F_2^n, labeled by int in [0, 2**n).
    generators : Iterable[np.ndarray]
        The generating set S, as an iterable of length-n 0/1 arrays.
        Duplicate or zero generators are dropped (a zero generator would
        produce self-loops, which are not meaningful here) -- a
        UserWarning is issued naming exactly how many were dropped and
        why, since |S| is meaningful downstream (e.g. it must be even
        for ``construct_cdz_code``) and a silent size change could be
        mistaken for the caller's original count.

    Returns
    -------
    networkx.Graph
        Vertices are ints 0..2**n - 1. Each vertex carries a ``vector``
        attribute with its F_2^n representation. Each edge carries a
        ``generator`` attribute recording which s in S produced it.
    """
    gens = []
    seen = set()
    n_zero = 0
    n_duplicate = 0
    for g in generators:
        gi = vector_to_int(np.asarray(g) % 2)
        if gi == 0:
            n_zero += 1
            continue
        if gi in seen:
            n_duplicate += 1
            continue
        seen.add(gi)
        gens.append(gi)

    if n_zero or n_duplicate:
        warnings.warn(
            f"cayley_graph: dropped {n_zero} zero generator(s) and "
            f"{n_duplicate} duplicate generator(s); using |S| = {len(gens)} "
            "of the originally supplied generators. This changes the "
            "parity of |S|, which construct_cdz_code requires to be even.",
            UserWarning,
            stacklevel=2,
        )

    graph = nx.Graph()
    n_vertices = 1 << n
    for x in range(n_vertices):
        graph.add_node(x, vector=int_to_vector(x, n))

    for x in range(n_vertices):
        for g in gens:
            y = x ^ g
            if not graph.has_edge(x, y):
                graph.add_edge(x, y, generator=g)

    graph.graph["n"] = n
    graph.graph["generators"] = gens
    return graph
