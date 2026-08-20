# Changelog

## [Unreleased]
- Stabilizer derivation (`_derive_stabilizers`) pending port from validated
  standalone CDZ replication.

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
