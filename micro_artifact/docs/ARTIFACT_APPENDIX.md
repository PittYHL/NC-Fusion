# NC-Fusion Artifact Appendix

## Abstract

This artifact packages the implementation and workflows used to evaluate
NC-Fusion, a Hamiltonian-simulation compiler that groups Pauli Product
Rotations, conjugates groups with Clifford circuits, schedules groups, and
fuses rotations before Clifford+T synthesis. It supports the paper's main
T-gate count, T-gate depth, Clifford-count, error, sensitivity, scalability,
optimizer-comparison, and spacetime-volume workloads. A dependency-free smoke
test provides a fast functional check; the full workloads use the original
Qiskit-based implementation and record structured metrics and environment
metadata.

## Itemized metadata

* **Paper:** `NC-Fusion: Optimizing T-Gate Cost for Hamiltonian Simulation`.
* **Artifact status:** source artifact plus reproducibility harness; the
  evaluator should record the final archival DOI here when available.
* **Primary language:** Python 3.9+.
* **Quantum software:** Qiskit, Qiskit Nature, Qiskit Aer, Qiskit Gridsynth
  plugin, Trasyn, Rustiq, Phoenix, and PyZX.
* **External software:** PySCF for molecular Hamiltonians, the Synthetiq
  container for the two-qubit synthesis experiments, [tzap](https://github.com/qqq-wisc/tzap)
  for the Rust optimizer comparison, and [T-Optimizer](https://github.com/iqubit-org/T-Optimizer)
  for the Pauli-based T-gate optimizer comparison. Section 5.10 also uses
  Infleqtion's `resource-superstaq` at the pinned revision recorded by its
  dedicated driver.
* **Hardware:** CPU is sufficient for inspection and small runs. The paper's
  largest synthesis jobs require a high-memory machine and may benefit from an
  NVIDIA GPU; the two-qubit Synthetiq workflow requires Docker and its image.
* **Randomness:** default seed `0`; every full output records the selected
  seed. Random-order experiments use 30 repetitions as specified in the paper.
* **Expected outputs:** `metrics.csv`, `manifest.json`, and, when requested,
  generated QASM under `micro_artifact/circuits/`.

## Access to the artifact

From the repository root, the artifact is in this project. The paper is
`micro_artifact/NC_Fusion.pdf`. No credentials or network service are required for the smoke
test. Full molecular experiments require the PySCF data path installed by the
selected Python environment. The Synthetiq image is not redistributed here;
evaluators should obtain it through the approved artifact channel and set its
input/output volume paths locally.

## System requirements and dependencies

The quick check requires Python and the core requirements listed in
`requirements-core.txt`. Full evaluation uses `requirements-paper.txt`, a
working PySCF installation, and (for two-qubit synthesis) Docker plus the
Synthetiq image. GPU execution is optional for the harness, but the paper's
original timings were collected on cluster hardware and should not be
interpreted as portable wall-clock guarantees.

## Experiment workflow

1. Read `micro_artifact/configs/paper.json` to select a paper section and its exact benchmark
   and parameter set.
2. Run `make smoke` to verify the harness and output format.
3. Install the paper dependencies in an isolated environment.
4. Run a single LiH/Gridsynth job and inspect `metrics.csv`.
5. Run the desired configured section, preserving its output directory and
   `manifest.json`.
6. Compare the main metrics with `micro_artifact/results/reference/table4.csv` using the
`validate` command, and use the section-specific settings for the remaining
figures.

Each requested evaluation also has a dedicated module under
`micro_artifact/`. These modules are the preferred artifact entry points;
`micro_artifact/MISSING_IMPLEMENTATIONS.md` identifies evaluations for which
the source driver or paper-specific formula is still missing.

## Steps for evaluation

```bash
# From the repository root
make smoke

# Discover all configured workloads
PYTHONPATH=src python -m ncfusion list

# Functional full-stack check on one benchmark/method
PYTHONPATH=src python -m ncfusion run table4 \
  --benchmark LiH --method gridsyn --output micro_artifact/results/runs/check-lih

# Main comparison
PYTHONPATH=src python -m ncfusion run table4 \
  --output micro_artifact/results/runs/table4
```

The full comparison may be long-running. For a reproducibility report, retain
the terminal log alongside the JSON/CSV outputs and note any dependency or
hardware substitution. Workloads containing `ncf-two` require Docker and the
configured Synthetiq image; the CLI stops with an actionable error if those
prerequisites are not available.

For the T-count optimizer comparison, use the dedicated entry point and pass
the locally built/cloned external tools:

```bash
PYTHONPATH=src:. python -m micro_artifact.t_count_methods_comparison \
  --benchmark H2 \
  --tzap-bin /path/to/tzap \
  --t-optimizer-root /path/to/T-Optimizer \
  --output micro_artifact/results/runs/t-count-methods
```

This writes each intermediate circuit and a long-form row for the original
Clifford+RZ circuit, GridSynth, tzap before and after RZ synthesis, PyZX on the
GridSynth Clifford+T circuit, and T-Optimizer on the GridSynth Clifford+T
circuit.

## Results

The paper reports average single-qubit NC-Fusion reductions of 57.3% in
T-count, 74.5% in T-depth, and 52.8% in Clifford count relative to Gridsynth;
relative to Phoenix+Trasyn, the corresponding values are 57.5%, 33.7%, and
45.0%. Table 4's per-benchmark values are captured in
`micro_artifact/results/reference/table4.csv` for evaluator comparison. The full artifact
does not claim a reproduced badge in advance: evaluators should determine
whether the generated outputs match the key paper results on their selected
environment.
