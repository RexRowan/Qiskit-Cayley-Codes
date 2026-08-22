"""
qiskit-cayley-codes
====================

CSS quantum code construction from Cayley graphs over F_2^n, following
Couvreur, Delfosse & Zémor, "A Construction of Quantum LDPC Codes from
Cayley Graphs" (ISIT 2011 / IEEE Trans. Inf. Theory 59(9), 2013), with
generator-set selection informed by original research on ℓ-zero-sumfree
sets and the Davenport constant over F_2^n.

Public API
----------
- ``cayley_graph``            : build Cay(F_2^n, S) as a networkx.Graph
- ``CDZCode``                 : CSS code container (Hx, Hz, [[N, K, D]])
- ``construct_cdz_code``      : build a CDZCode from F_2^n and generator set S
- ``is_zero_sumfree``         : check the ℓ-zero-sumfree property of a subset
- ``davenport_constant_upper_bound`` : combinatorial bound used for generator search
- ``theorem16_lower_bound``   : CDZ's own quantum distance bound (Theorem 16)
- ``search_generator_sets``   : search candidate generators, ranked by rate
- ``compare_to_known_families`` : benchmark against toric/surface code params
"""

from .analysis import (
    classical_min_distance_bruteforce,
    compare_to_known_families,
    search_generator_sets,
    surface_code_params,
    theorem16_lower_bound,
    toric_code_params,
)
from .construction import CDZCode, construct_cdz_code
from .graph import cayley_graph
from .utils import f2_rank, hamming_weight, int_to_vector, vector_to_int
from .zero_sumfree import davenport_constant_upper_bound, is_zero_sumfree

__version__ = "0.1.1"

__all__ = [
    "CDZCode",
    "cayley_graph",
    "classical_min_distance_bruteforce",
    "compare_to_known_families",
    "construct_cdz_code",
    "davenport_constant_upper_bound",
    "f2_rank",
    "hamming_weight",
    "int_to_vector",
    "is_zero_sumfree",
    "search_generator_sets",
    "surface_code_params",
    "theorem16_lower_bound",
    "toric_code_params",
    "vector_to_int",
]
