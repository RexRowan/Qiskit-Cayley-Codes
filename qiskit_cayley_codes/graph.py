"""Cayley graph construction over F_2^n.

Cay(F_2^n, S) has vertex set F_2^n and an edge {x, x+s} for every x in
F_2^n and every s in S (S implicitly symmetric since s = -s over F_2).
This is the graph that CDZ place qubits on (one qubit per edge) as the
starting point for the CSS construction in ``construction.py``.
"""

from __future__ import annotations

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
        Duplicate or zero generators are ignored (a zero generator would
        produce self-loops, which are not meaningful here).

    Returns
    -------
    networkx.Graph
        Vertices are ints 0..2**n - 1. Each vertex carries a ``vector``
        attribute with its F_2^n representation. Each edge carries a
        ``generator`` attribute recording which s in S produced it.
    """
    gens = []
    seen = set()
    for g in generators:
        gi = vector_to_int(np.asarray(g) % 2)
        if gi == 0 or gi in seen:
            continue
        seen.add(gi)
        gens.append(gi)

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
