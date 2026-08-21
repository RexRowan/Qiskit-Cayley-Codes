import numpy as np
import pytest

from qiskit_cayley_codes.analysis import (
    classical_min_distance_bruteforce,
    compare_to_known_families,
    search_generator_sets,
    surface_code_params,
    theorem16_lower_bound,
    toric_code_params,
)


def test_classical_min_distance_repetition_code():
    # [3,1,3] repetition code: parity check [[1,1,0],[0,1,1]], codewords
    # are {000, 111}, min distance 3.
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert classical_min_distance_bruteforce(H) == 3


def test_classical_min_distance_trivial_kernel_rejected():
    H = np.eye(3, dtype=np.uint8)  # full rank, trivial kernel
    with pytest.raises(ValueError):
        classical_min_distance_bruteforce(H)


def test_theorem16_lower_bound_requires_w_positive():
    with pytest.raises(ValueError, match="w > 0"):
        theorem16_lower_bound(5, [])


def test_theorem16_lower_bound_rejects_small_distance():
    # w=1 with a single extra generator: C(W) is [m+1, 1, d] and d is
    # small (bounded by the weight of the single extra column plus 1),
    # certainly under 9 for small m.
    m = 4
    W = [np.array([1, 0, 0, 0], dtype=np.uint8)]
    with pytest.raises(ValueError, match="d >= 9"):
        theorem16_lower_bound(m, W)


def test_theorem16_lower_bound_finds_valid_case():
    """Search for an (m, W) with classical distance >= 9. Kernel
    dimension for C(W) is w (small and fixed), so this is cheap
    regardless of m. w is kept small and m large so the code's rate
    w/(m+w) is low enough that random W plausibly clears d >= 9 (random
    linear codes typically approach the Gilbert-Varshamov distance
    bound, which needs a low rate to guarantee a large relative
    distance)."""
    m = 30
    w = 3
    rng = np.random.default_rng(42)
    for _ in range(200):
        W = [rng.integers(0, 2, size=m).astype(np.uint8) for _ in range(w)]
        try:
            result = theorem16_lower_bound(m, W)
        except ValueError:
            continue
        assert result["m"] == m
        assert result["w"] == w
        assert result["n"] == m + w
        assert result["d"] >= 9
        assert result["bound"] == pytest.approx(result["d"] * result["n"] ** 2 / 640)
        return
    pytest.fail("Could not find a qualifying random W in 200 tries (unexpected)")


def test_search_generator_sets_returns_valid_sorted_results():
    m = 4
    pool = [
        np.array([1, 1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 1, 0], dtype=np.uint8),
        np.array([1, 1, 1, 1], dtype=np.uint8),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    ]
    # standard_basis(4) has 4 elements (even); w must also be even so the
    # total generator count stays even (required for a valid CSS code).
    results = search_generator_sets(m, pool, w=2, require_zero_sumfree=False)
    assert len(results) > 0
    for r in results:
        assert r["N"] == 2**m
        assert r["rate"] == pytest.approx(r["K"] / r["N"])
    # sorted by descending rate
    rates = [r["rate"] for r in results]
    assert rates == sorted(rates, reverse=True)


def test_search_generator_sets_respects_max_candidates():
    m = 4
    pool = [np.eye(m, dtype=np.uint8)[i] + np.eye(m, dtype=np.uint8)[(i + 1) % m] for i in range(m)]
    # w=2 keeps the total generator count (4 + 2 = 6) even, so combinations
    # actually survive the validity check and this test has teeth.
    results_unbounded = search_generator_sets(m, pool, w=2, require_zero_sumfree=False)
    results_bounded = search_generator_sets(
        m, pool, w=2, max_candidates=1, require_zero_sumfree=False
    )
    assert len(results_unbounded) > 1  # otherwise the bound below is untested
    assert len(results_bounded) <= 1
    assert len(results_bounded) <= len(results_unbounded)


def test_toric_code_params():
    params = toric_code_params(4)
    assert params == {"name": "toric", "N": 32, "K": 2, "D": 4}


def test_surface_code_params():
    params = surface_code_params(3)
    assert params == {"name": "surface", "N": 3**2 + 2**2, "K": 1, "D": 3}


def test_compare_to_known_families_structure():
    comparison = compare_to_known_families(N=32, D=4)
    assert comparison["cdz"] == {"N": 32, "D": 4}
    assert comparison["toric"]["name"] == "toric"
    assert comparison["surface"]["name"] == "surface"
    # closest toric N to 32 should be exact (L=4 gives N=32)
    assert comparison["toric"]["N"] == 32
