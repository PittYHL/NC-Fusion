# Legacy implementation map

The files in this directory contain the research implementation used by the
artifact adapter. They are kept separate from `src/ncfusion/` so the public
CLI remains small and easy to follow.

The runnable path is:

```text
NC_Fusion(hamiltonian, budget=1|2, synthesize=True)
  └── returns (compiled_qc, clifford_t_qc|None)

src/ncfusion/legacy.py
  ├── baseline.py                  GridSynth, Rustiq, and Phoenix baselines
  ├── grouping.py                  Pauli grouping for NC-Fusion
  ├── new_gaussian.py              GF(2) dependency analysis
  ├── circuit_generation_greedy.py Clifford circuit generation
  ├── compressor.py                One- and two-qubit compression
  ├── reorder.py                   Independent-group scheduling
  └── docker.py                    Optional Synthetiq container interface
```

`commuting_graph.py` and `error_evaluation.py` provide shared helpers for the
modules above. The `phoenix/` package is the Phoenix baseline implementation.

Debug-only equivalence checks, duplicate commutation implementations, and
experimental modules that were not reachable from the artifact workflows were
removed from this GitHub copy. The original source tree outside `NC-Fusion/`
was not modified by this cleanup.

Research modules are imported lazily by `src/ncfusion/legacy.py`; this keeps
`make list` and `make smoke` usable before the optional quantum-computing
dependencies are installed.
