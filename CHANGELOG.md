# Changelog

## [Unreleased]
- Implemented `_derive_stabilizers`: Hx = Hz = adjacency matrix of
  Cay(F_2^n, S), valid whenever |S| is even (Couvreur-Delfosse-Zemor,
  Example 2). `construct_cdz_code` is now fully functional end-to-end.
- Added `CDZCode.min_distance_bruteforce()`, exact but exponential in
  dim(Ker(Hx)); practical for small test/reference codes only.
- Validated against the paper's own worked example (Theorem 18 / Prop
  25): n=3 repetition-code case gives exactly [[8, 4, 2]]; n=5 gives
  N=32, K=8 (spot-checked, not in the committed suite since brute-force
  distance there is too slow for CI).
- Added `CDZCode.to_qiskit_qec()`, converting Hx/Hz into a
  `qiskit_qec.codes.StabSubSystemCode` via sparse Pauli strings. Note
  `qiskit-qec` is not on PyPI and must be installed from source; it is
  intentionally not a project dependency for that reason.

## [0.1.0] - Unreleased
### Added
- F_2 linear algebra utilities (`utils.py`): rank, nullspace, vector/int
  encoding.
- Cayley graph construction over F_2^n (`graph.py`), built on networkx.
- ℓ-zero-sumfree set checking and Davenport constant utilities
  (`zero_sumfree.py`).
- `CDZCode` container with CSS orthogonality validation and (N, K)
  computation (`construction.py`).
- Initial test suite covering utils, graph, and zero-sumfree modules.
