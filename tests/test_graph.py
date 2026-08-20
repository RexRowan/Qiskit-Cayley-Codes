import numpy as np

from qiskit_cayley_codes.graph import cayley_graph


def test_hypercube_case():
    """S = standard basis gives the n-cube graph Q_n: 2^n vertices, n-regular."""
    n = 3
    gens = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]
    g = cayley_graph(n, gens)
    assert g.number_of_nodes() == 2**n
    assert all(deg == n for _, deg in g.degree())


def test_single_generator_gives_perfect_matching():
    n = 3
    gens = [np.array([1, 0, 0], dtype=np.uint8)]
    g = cayley_graph(n, gens)
    assert g.number_of_edges() == (2**n) // 2
    assert all(deg == 1 for _, deg in g.degree())


def test_zero_generator_ignored():
    n = 2
    gens = [np.array([0, 0], dtype=np.uint8), np.array([1, 0], dtype=np.uint8)]
    g = cayley_graph(n, gens)
    assert g.graph["generators"] == [1]  # only the nonzero generator survives


def test_duplicate_generators_deduplicated():
    n = 2
    gens = [np.array([1, 0], dtype=np.uint8), np.array([1, 0], dtype=np.uint8)]
    g = cayley_graph(n, gens)
    assert len(g.graph["generators"]) == 1


def test_vertex_vector_attribute_consistent():
    n = 4
    gens = [np.eye(n, dtype=np.uint8)[0]]
    g = cayley_graph(n, gens)
    for node, data in g.nodes(data=True):
        from qiskit_cayley_codes.utils import vector_to_int

        assert vector_to_int(data["vector"]) == node
