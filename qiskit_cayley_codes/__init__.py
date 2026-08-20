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
"""

from .construction import CDZCode, construct_cdz_code
from .graph import cayley_graph
from .utils import f2_rank, hamming_weight, int_to_vector, vector_to_int
from .zero_sumfree import davenport_constant_upper_bound, is_zero_sumfree

__version__ = "0.1.0"

__all__ = [
    "CDZCode",
    "cayley_graph",
    "construct_cdz_code",
    "davenport_constant_upper_bound",
    "f2_rank",
    "hamming_weight",
    "int_to_vector",
    "is_zero_sumfree",
    "vector_to_int",
]
