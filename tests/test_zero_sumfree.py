import numpy as np
import pytest

from qiskit_cayley_codes.zero_sumfree import (
    davenport_constant_upper_bound,
    find_zero_sum_relations,
    is_zero_sumfree,
)


def test_standard_basis_is_zero_sumfree():
    """The standard basis of F_2^n is zero-sumfree for any n: no nonempty
    subset of distinct basis vectors can XOR to zero."""
    n = 5
    gens = [np.eye(n, dtype=np.uint8)[i] for i in range(n)]
    assert is_zero_sumfree(gens)


def test_pair_summing_to_zero_detected():
    v = np.array([1, 1, 0], dtype=np.uint8)
    gens = [v, v]  # v XOR v = 0
    assert not is_zero_sumfree(gens)
    relations = find_zero_sum_relations(gens)
    assert [0, 1] in relations


def test_three_way_relation_detected():
    a = np.array([1, 0, 0], dtype=np.uint8)
    b = np.array([0, 1, 0], dtype=np.uint8)
    c = np.array([1, 1, 0], dtype=np.uint8)  # a xor b xor c = 0
    assert not is_zero_sumfree([a, b, c])
    relations = find_zero_sum_relations([a, b, c])
    assert [0, 1, 2] in relations


def test_ell_bound_hides_longer_relations():
    a = np.array([1, 0, 0], dtype=np.uint8)
    b = np.array([0, 1, 0], dtype=np.uint8)
    c = np.array([1, 1, 0], dtype=np.uint8)
    # the only zero-sum relation has size 3; bounding ell=2 should miss it
    assert is_zero_sumfree([a, b, c], ell=2)
    assert not is_zero_sumfree([a, b, c], ell=3)


def test_davenport_constant_f2n():
    assert davenport_constant_upper_bound(0) == 1
    assert davenport_constant_upper_bound(4) == 5
    with pytest.raises(ValueError):
        davenport_constant_upper_bound(-1)
