"""Tests for the CDZ CSS code construction.

The end-to-end tests validate against the paper's own worked example
(Couvreur-Delfosse-Zemor, Theorem 18 / Prop 25): for odd n, generators
= standard basis of F_2^n union {all-ones vector} gives
[[N=2^n, K=2^((n+1)/2), D=2^((n-1)/2)]]. n=3 is used here (N=8) since
brute-force distance verification is only tractable for small N.
"""

import numpy as np
import pytest

from qiskit_cayley_codes.construction import CDZCode, construct_cdz_code


def test_x_stabilizer_strings():
    Hx = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    Hz = np.zeros((0, 4), dtype=np.uint8)
    code = CDZCode(Hx=Hx, Hz=Hz, n_ambient=2, generators=[1, 2])
    assert code.x_stabilizer_strings() == ["X0X2", "X1X3"]


def test_z_stabilizer_strings():
    Hx = np.zeros((0, 3), dtype=np.uint8)
    Hz = np.array([[1, 1, 0]], dtype=np.uint8)
    code = CDZCode(Hx=Hx, Hz=Hz, n_ambient=2, generators=[1, 2])
    assert code.z_stabilizer_strings() == ["Z0Z1"]


def test_stabilizer_strings_reject_all_zero_row():
    Hx = np.array([[0, 0, 0]], dtype=np.uint8)
    Hz = np.zeros((0, 3), dtype=np.uint8)
    code = CDZCode(Hx=Hx, Hz=Hz, n_ambient=2, generators=[1])
    with pytest.raises(ValueError):
        code.x_stabilizer_strings()


def test_to_qiskit_qec_requires_qiskit_qec_or_builds_correctly():
    """If qiskit-qec isn't installed (it's not on PyPI), confirm we raise a
    clear, actionable ImportError rather than a bare traceback. If it *is*
    installed, confirm the conversion actually builds a working code."""
    Hx = np.array([[1, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
    Hz = np.array([[1, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
    code = CDZCode(Hx=Hx, Hz=Hz, n_ambient=3, generators=[1, 2, 4])

    try:
        import qiskit_qec  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="not on PyPI"):
            code.to_qiskit_qec()
        return

    qec_code = code.to_qiskit_qec()
    assert qec_code is not None


def test_cdzcode_valid_css_pair():
    # Steane-like toy example: Hx == Hz, self-orthogonal under the
    # CSS condition Hx @ Hz.T == 0 (mod 2).
    Hx = np.array([[1, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
    Hz = np.array([[1, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
    code = CDZCode(Hx=Hx, Hz=Hz, n_ambient=3, generators=[1, 2, 4])
    n, k = code.parameters()
    assert n == 7
    assert k == 7 - 1 - 1


def test_cdzcode_rejects_non_orthogonal_pair():
    Hx = np.array([[1, 0, 0]], dtype=np.uint8)
    Hz = np.array([[1, 0, 0]], dtype=np.uint8)  # Hx . Hz = 1, not orthogonal
    with pytest.raises(ValueError):
        CDZCode(Hx=Hx, Hz=Hz, n_ambient=2, generators=[1, 2])


def test_cdzcode_shape_mismatch_rejected():
    Hx = np.zeros((1, 4), dtype=np.uint8)
    Hz = np.zeros((1, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        CDZCode(Hx=Hx, Hz=Hz, n_ambient=2, generators=[1])


def test_repetition_code_case_end_to_end():
    """Reference case from CDZ (Theorem 18): classical code = [n,1,n]
    repetition code, i.e. generators = standard basis + all-ones vector,
    gives quantum parameters [[2^n, 2^((n+1)/2), 2^((n-1)/2)]] for odd n."""
    n = 3
    generators = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]
    generators.append(np.ones(n, dtype=np.uint8))
    code = construct_cdz_code(n, generators)

    N, K = code.parameters()
    assert N == 2**n
    assert K == 2 ** ((n + 1) // 2)

    # Paper's Remark 7: M(S'_3) has rank 2 for the n=3 case specifically.
    from qiskit_cayley_codes.utils import f2_rank

    assert f2_rank(code.Hx) == 2

    D = code.min_distance_bruteforce()
    assert D == 2 ** ((n - 1) // 2)


def test_odd_generator_count_rejected():
    """|S| must be even (Example 2 in the paper); e.g. dropping the
    all-ones vector from the repetition-code generating set leaves an
    odd-sized set and should be rejected, not silently produce a code."""
    n = 3
    generators = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]  # |S| = 3, odd
    with pytest.raises(ValueError, match="even number"):
        construct_cdz_code(n, generators)


def test_hypercube_case_is_self_dual_trivial_code():
    """Paper Prop 9: for n even, S = standard basis gives a self-dual
    (K=0) code -- included here as a second, independent construction
    check distinct from the repetition-code case above."""
    n = 4
    generators = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]  # |S| = 4, even
    code = construct_cdz_code(n, generators)
    N, K = code.parameters()
    assert N == 2**n
    assert K == 0
