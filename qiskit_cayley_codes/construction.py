"""CSS code construction from a Cayley graph, following CDZ (ISIT 2011).

The construction (Couvreur, Delfosse, Zemor, "A Construction of Quantum
LDPC Codes from Cayley Graphs", arXiv:1206.2656):

  Given S subset of F_2^n with |S| even, let M(S) be the 2^n x 2^n
  adjacency matrix of the Cayley graph Cay(F_2^n, S): qubits sit on the
  VERTICES of the graph (one per element of F_2^n, so N = 2^n), and row
  x of M(S) has a 1 in column y = x XOR s for every generator s in S --
  i.e. row x is the indicator of x's neighborhood.

  M(S) is symmetric (the graph is undirected), and -- this is the key
  fact, proved in the paper's Example 2 -- for ANY generating set with
  |S| even, M(S) . M(S)^T = 0 (mod 2). This is a general fact about the
  group algebra F_2[F_2^n]: every element g in F_2^n squares to the
  identity, so a sum of an even number of (distinct) group elements
  squares to zero. No further condition on S is needed beyond parity.

  Using Hx = Hz = M(S) then automatically satisfies the CSS
  orthogonality condition Hx @ Hz.T = 0, giving a valid CSS code with
  N = 2^n physical qubits and K = N - 2*rank(M(S)) logical qubits.

  Reference worked example (paper Theorem 18 / Prop 25): for n odd,
  S = (standard basis of F_2^n) union {all-ones vector} (|S| = n+1,
  even), the resulting code has parameters
      [[N = 2^n, K = 2^((n+1)/2), D = 2^((n-1)/2)]].
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .graph import cayley_graph
from .utils import f2_nullspace_basis, f2_rank


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

    def min_distance_bruteforce(self, max_kernel_dim: int = 16) -> int:
        """Exact minimum distance via brute-force enumeration of Ker(Hx).

        The code distance is the minimum weight of a nonzero vector in
        Ker(Hx) that is not in the row space of Hz (a "logical Z
        operator"). For this construction Hx == Hz, so by symmetry this
        also equals the corresponding X-distance, and hence the overall
        code distance.

        Cost is O(2^dim(Ker(Hx))), so this is only practical for small
        test codes (e.g. the n=3 repetition-code case from the paper,
        N=8). It is not a substitute for the paper's analytical distance
        bounds/formulas for real-sized codes -- see Theorem 16 and
        Theorem 18 in Couvreur-Delfosse-Zemor.

        Raises
        ------
        ValueError
            If dim(Ker(Hx)) exceeds ``max_kernel_dim`` (the search space
            is 2^that, so this guards against an effectively-infinite
            brute-force loop).
        """
        kernel_basis = f2_nullspace_basis(self.Hx)
        k_dim = kernel_basis.shape[0]
        if k_dim > max_kernel_dim:
            raise ValueError(
                f"Ker(Hx) has dimension {k_dim}; brute-force search costs "
                f"O(2^{k_dim}) and is impractical beyond "
                f"max_kernel_dim={max_kernel_dim}. Use this method for "
                "small test codes only."
            )
        if k_dim == 0:
            return 0

        hz_rank = f2_rank(self.Hz)
        best = None
        for bits in range(1, 1 << k_dim):
            combo = np.zeros(self.n_qubits, dtype=np.uint8)
            for i in range(k_dim):
                if (bits >> i) & 1:
                    combo ^= kernel_basis[i]
            augmented = np.vstack([self.Hz, combo])
            if f2_rank(augmented) == hz_rank:
                continue  # combo is in the row space of Hz: not a logical operator
            weight = int(combo.sum())
            if best is None or weight < best:
                best = weight
        return best if best is not None else 0

    def x_stabilizer_strings(self) -> list:
        """Hx rows as sparse qubit-indexed Pauli strings, e.g. 'X0X3X7'."""
        return _stabilizer_strings(self.Hx, "X")

    def z_stabilizer_strings(self) -> list:
        """Hz rows as sparse qubit-indexed Pauli strings, e.g. 'Z1Z2Z9'."""
        return _stabilizer_strings(self.Hz, "Z")

    def to_qiskit_qec(self):
        """Build a qiskit_qec StabSubSystemCode from this CSS code.

        Requires ``qiskit-qec``, which is not published on PyPI -- install
        it from source first:

            pip install "qiskit-qec @ git+https://github.com/qiskit-community/qiskit-qec.git"

        (This is why it's not listed as a project dependency or optional-
        dependency extra in pyproject.toml: PyPI does not accept published
        packages that depend on a direct GitHub URL, so it can't be wired
        up as `pip install qiskit-cayley-codes[qec]`. Install it yourself
        and this method will pick it up.)

        Returns
        -------
        qiskit_qec.codes.StabSubSystemCode
        """
        try:
            from qiskit_qec.codes import StabSubSystemCode
            from qiskit_qec.operators import PauliList
            from qiskit_qec.structures import GaugeGroup
        except ImportError as exc:
            raise ImportError(
                "to_qiskit_qec() requires qiskit-qec, which is not on PyPI. "
                "Install it from source: pip install "
                '"qiskit-qec @ git+https://github.com/qiskit-community/qiskit-qec.git"'
            ) from exc

        stabilizer_strings = self.x_stabilizer_strings() + self.z_stabilizer_strings()
        generators = PauliList(stabilizer_strings)
        gauge_group = GaugeGroup(generators)
        return StabSubSystemCode(gauge_group)


def _stabilizer_strings(matrix: np.ndarray, pauli: str) -> list:
    """Convert a 0/1 check matrix into sparse qubit-indexed Pauli strings.

    Each row becomes e.g. 'X0X3X7' (pauli at each column where the row is 1).
    A theoretically-possible but practically-invalid all-zero row (an empty
    check) raises, since it corresponds to no operator at all.
    """
    strings = []
    for row in matrix:
        support = np.nonzero(row)[0]
        if len(support) == 0:
            raise ValueError(
                "Encountered an all-zero row in a check matrix; this is not "
                "a valid stabilizer generator."
            )
        strings.append("".join(f"{pauli}{int(idx)}" for idx in support))
    return strings


def _derive_stabilizers(graph: nx.Graph, n: int, generators: list) -> tuple[np.ndarray, np.ndarray]:
    """Derive (Hx, Hz) from a Cayley graph per the CDZ construction.

    Hx = Hz = the adjacency matrix M(S) of ``graph`` (qubits on vertices,
    N = 2^n). Valid whenever |S| is even -- see module docstring for why
    no further condition on S is required.

    Raises
    ------
    ValueError
        If the (deduplicated, nonzero) generating set has odd size, so
        the resulting matrix would not be self-orthogonal.
    """
    if len(generators) % 2 != 0:
        raise ValueError(
            "CDZ construction over F_2^n requires an even number of "
            "(deduplicated, nonzero) generators -- Couvreur-Delfosse-Zemor, "
            f"Example 2 -- got |S| = {len(generators)}. With |S| odd the "
            "adjacency matrix is not self-orthogonal and no CSS code results."
        )
    order = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=order, dtype=np.uint8) % 2
    return adjacency, adjacency.copy()


def construct_cdz_code(n: int, generators: Iterable[np.ndarray]) -> CDZCode:
    """Build a CDZCode from Cay(F_2^n, S) via the CDZ (ISIT 2011) construction.

    Parameters
    ----------
    n : int
        Ambient F_2^n dimension.
    generators : Iterable[np.ndarray]
        Generating set S (length-n 0/1 arrays), with an even number of
        distinct nonzero elements (see module docstring). Typically
        chosen to be zero-sumfree (see ``zero_sumfree.is_zero_sumfree``)
        to control the girth of the resulting Cayley graph.

    Returns
    -------
    CDZCode

    Notes
    -----
    Builds a dense 2^n x 2^n matrix, so this is only practical for
    small n (comfortably n <= 12 or so; n=20 would already need a
    ~1M x 1M dense array). This mirrors the construction's own
    N = 2^n qubit count -- it is not a limitation specific to this
    implementation.
    """
    gens = list(generators)
    graph = cayley_graph(n, gens)
    Hx, Hz = _derive_stabilizers(graph, n, graph.graph["generators"])
    return CDZCode(Hx=Hx, Hz=Hz, n_ambient=n, generators=graph.graph["generators"])
