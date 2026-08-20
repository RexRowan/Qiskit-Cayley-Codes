"""Tests for the CDZ CSS code construction.

Two tiers:
  1. ``CDZCode`` container tests -- fully working today, exercise the
     CSS orthogonality check and (N, K) computation directly against
     hand-built Hx/Hz pairs.
  2. ``construct_cdz_code`` end-to-end tests -- marked xfail until
     ``_derive_stabilizers`` is ported from the validated replication
     (see construction.py). Once ported, remove the xfail marks and
     assert against the known parameters, e.g. the paper's worked
     example for the [n, 1, n] repetition code case:
     [[2^n, 2^((n+1)/2), 2^((n-1)/2)]] for odd n.
"""

import numpy as np
import pytest

from qiskit_cayley_codes.construction import CDZCode, construct_cdz_code


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


@pytest.mark.xfail(reason="stabilizer derivation not yet ported, see construction.py", strict=True)
def test_repetition_code_case_end_to_end():
    """Reference case from CDZ: classical code = [n,1,n] repetition code
    gives quantum parameters [[2^n, 2^((n+1)/2), 2^((n-1)/2)]] for odd n."""
    n = 3
    generators = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]
    code = construct_cdz_code(n, generators)
    N, K = code.parameters()
    assert N == 2**n
    assert K == 2 ** ((n + 1) // 2)
