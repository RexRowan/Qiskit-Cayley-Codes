# Changelog

## [0.1.1] - Unreleased
- Fixed a real, still-present doc bug: `graph.py`'s module docstring said "one qubit
  per edge," left over from before `_derive_stabilizers` was
  implemented (qubits are on vertices, N=2^n) -- the wrong sentence in
  `construction.py` had been fixed earlier but this one was missed.
  This bug shipped in the published 0.1.0.
- `cayley_graph` now emits a `UserWarning` naming exactly how many
  zero/duplicate generators were dropped, instead of silently changing
  |S| -- since |S|'s parity matters downstream for
  `construct_cdz_code`, silently normalizing it away could mislead a
  caller about what they actually built. Also not present in 0.1.0.
- Fixed LICENSE: the previous file had truncated several sections of
  the Apache-2.0 text from memory (missing definitions in Section 1,
  abbreviated Sections 3-9). Replaced with the exact canonical text
  (verified against GitHub's own choosealicense.com template) so
  GitHub's license detector can identify it correctly.
- Added `qiskit_cayley_codes/analysis.py`:
  - `theorem16_lower_bound`: CDZ's Theorem 16 quantum distance bound,
    computed from the small classical code C(W) (dimension w) rather
    than brute-forcing the full 2^m-qubit code.
  - `search_generator_sets`: ranks candidate generator combinations by
    rate (K/N), using zero-sumfreeness as a girth proxy.
  - `compare_to_known_families`: rough benchmark against toric and
    planar surface code parameters at comparable code length.
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
