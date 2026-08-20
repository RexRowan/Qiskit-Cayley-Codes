"""CSS code construction from a Cayley graph, following CDZ (ISIT 2011).

This module intentionally separates two concerns:

1. ``CDZCode`` -- a plain data container for a CSS code (Hx, Hz, and
   the resulting [[N, K, D]] parameters), independent of how the
   stabilizers were derived. This part is complete and tested.

2. ``construct_cdz_code`` / ``_derive_stabilizers`` -- the actual
   graph -> stabilizer derivation from the CDZ paper. The container
   and graph-building code around it (``graph.py``) are generic and
   safe to rely on; the stabilizer derivation is the piece with real
   research risk if reimplemented from a paper summary rather than
   your already-validated replication.

   Port your validated derivation into ``_derive_stabilizers`` below.
   Until that's done, ``construct_cdz_code`` raises NotImplementedError
   so nothing silently produces an incorrect code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .graph import cayley_graph
from .utils import f2_rank


@dataclass
class CDZCode:
    """A CSS code specified by its X- and Z-type parity check matrices.

    Attributes
    ----------
    Hx, Hz : np.ndarray
        0/1 parity check matrices over F_2, shape (m_x, N) and (m_z, N)
        where N is the number of physical qubits. Must satisfy the CSS
        orthogonality condition Hx @ Hz.T == 0 (mod 2).
    n_ambient : int
        The n such that the code was built from Cay(F_2^n, S).
    generators : list[int]
        The generating set S used, as integer bitmasks.
    """

    Hx: np.ndarray
    Hz: np.ndarray
    n_ambient: int
    generators: list

    def __post_init__(self):
        self.Hx = np.asarray(self.Hx, dtype=np.uint8) % 2
        self.Hz = np.asarray(self.Hz, dtype=np.uint8) % 2
        if self.Hx.shape[1] != self.Hz.shape[1]:
            raise ValueError("Hx and Hz must have the same number of columns (qubits)")
        residual = (self.Hx.astype(int) @ self.Hz.astype(int).T) % 2
        if residual.any():
            raise ValueError(
                "CSS orthogonality violated: Hx @ Hz.T != 0 (mod 2). "
                "This is not a valid CSS code."
            )

    @property
    def n_qubits(self) -> int:
        return self.Hx.shape[1]

    @property
    def k_logical(self) -> int:
        """Number of logical qubits: N - rank(Hx) - rank(Hz)."""
        return self.n_qubits - f2_rank(self.Hx) - f2_rank(self.Hz)

    def parameters(self) -> tuple[int, int]:
        """Return (N, K). Distance D is not computed here (see README)."""
        return self.n_qubits, self.k_logical

    def __repr__(self) -> str:
        n, k = self.parameters()
        return f"CDZCode(N={n}, K={k}, n_ambient={self.n_ambient}, |S|={len(self.generators)})"


def _derive_stabilizers(graph: nx.Graph, n: int, generators: list) -> tuple[np.ndarray, np.ndarray]:
    """Derive (Hx, Hz) from a Cayley graph per the CDZ construction.

    TODO(Rex): port the validated derivation here. Qubits sit on the
    edges of ``graph`` (one per generator per vertex, deduplicated);
    ``graph.edges(data=True)`` gives each edge's ``generator`` attribute
    to recover which s in S it corresponds to.

    Raises
    ------
    NotImplementedError
        Always, until this is filled in.
    """
    raise NotImplementedError(
        "Stabilizer derivation not yet ported. See module docstring: "
        "wire in the validated CDZ replication here rather than the "
        "placeholder in this scaffold."
    )


def construct_cdz_code(n: int, generators: Iterable[np.ndarray]) -> CDZCode:
    """Build a CDZCode from Cay(F_2^n, S) via the CDZ (ISIT 2011) construction.

    Parameters
    ----------
    n : int
        Ambient F_2^n dimension.
    generators : Iterable[np.ndarray]
        Generating set S (length-n 0/1 arrays). Typically chosen to be
        zero-sumfree (see ``zero_sumfree.is_zero_sumfree``) to control
        the girth of the resulting Cayley graph.

    Returns
    -------
    CDZCode
    """
    gens = list(generators)
    graph = cayley_graph(n, gens)
    Hx, Hz = _derive_stabilizers(graph, n, graph.graph["generators"])
    return CDZCode(Hx=Hx, Hz=Hz, n_ambient=n, generators=graph.graph["generators"])
