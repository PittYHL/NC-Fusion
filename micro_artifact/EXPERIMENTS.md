# NC-Fusion artifact experiments

This document is the execution guide for the artifact evaluation. Run all
commands from the repository root, `NC-Fusion/`.

## Setup

Install the paper dependencies and expose the project, legacy implementation,
and artifact modules on `PYTHONPATH`:

```bash
python3 -m pip install -e ".[paper]"
export PYTHONPATH=src:legacy:.
```

The artifact uses the following optional dependencies for specific sections:

- `ncf-two` requires Docker and the configured Synthetiq image.
- The optimizer comparison requires locally built `tzap` and a cloned
  T-Optimizer repository.
- The spacetime-volume experiment requires Infleqtion's
  `resource-superstaq` package at revision
  `717cbbfc62e558be3f2f9acb512e992d3cd43529`.

The dependency-free harness check is:

```bash
make smoke
```

To list configured benchmarks and paper experiments:

```bash
make list
```

## Common command options

The evaluation modules use these options where applicable:

```text
--benchmark NAME       Select one benchmark; repeat for multiple benchmarks.
--method NAME          Select one method; repeat for multiple methods.
--output DIRECTORY     Write metrics.csv and manifest.json here.
--seed INTEGER         Random seed; default is 0.
--gpu 0|1              0 uses CPU; 1 enables GPU synthesis in Trasyn.
```

The default paper synthesis threshold is `0.001`. `--gpu 1` is a Trasyn GPU
enable flag; it does not select among multiple GPU devices.

Every completed evaluation writes `metrics.csv` and `manifest.json`. Run
directories under `micro_artifact/results/runs/` are ignored by Git. QASM
inputs and generated producer QASM are kept under
`micro_artifact/circuits/`.

## Stored data and reuse policy

The canonical single-qubit QASM files are under
`micro_artifact/circuits/single-qubit/`. The naming convention is:

```text
<benchmark>_grid_rz.qasm       Original RZ circuit
<benchmark>_grid_c+t.qasm      GridSynth Clifford+T circuit
<benchmark>_ncf_rz.qasm        NC-Fusion rotation circuit
<benchmark>_ncf_c+t.qasm       NC-Fusion Clifford+T circuit
<benchmark>_rustiq_c+t.qasm    Rustiq Clifford+T circuit
<benchmark>_phoenix_c+t.qasm   Phoenix Clifford+T circuit
```

`single_qubit_result` and `two_qubit_result` are the producer evaluations.
They can either read stored QASM (`--source existing`) or regenerate it
(`--source generate`). Downstream analytical estimation reads the producer
CSV at the fixed path `micro_artifact/results/runs/single_qubit_result` or
`two_qubit_result`; use those default output directories when the result will
be consumed by another experiment.

The single-qubit producer records the NC-Fusion unitary count,
`ncf_unitaries_generated`, the original RZ count,
`original_rz_gate_count`, and `compilation_time_seconds`. Analytical
estimation uses the first two counts directly. Existing-data workflows reuse
the stored QASM and recorded compilation time whenever both are available;
otherwise they compile the missing circuit and mark the row as generated.

The two sensitivity studies always regenerate their circuits and never save
QASM. In both ablations, the full NC-Fusion arm reuses the single-qubit
producer QASM when it is available; the other ablation arms are regenerated.

## 1. Single-qubit result

This is the main single-qubit producer and Table 4 metric report. It reports
T count, T depth, Clifford count, compilation time, original RZ count, and
NC-Fusion unitary count. It also reports reductions relative to GridSynth,
Rustiq, and Phoenix. Rustiq and Phoenix have 13 supplied benchmark files;
MgO and NaCl are excluded from those two comparisons. T depth uses the paper
definition:

```python
new_qc.depth(lambda gate: gate[0].name == "t")
```

Use stored QASM:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.single_qubit_result \
  --source existing \
  --benchmark LiH \
  --benchmark Ising-2D-30 \
  --output micro_artifact/results/runs/single_qubit_result
```

Regenerate selected benchmarks and save their QASM under
`micro_artifact/circuits/single-qubit/`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.single_qubit_result \
  --source generate \
  --benchmark Ising-2D-30 \
  --gpu 1 \
  --error-threshold 0.001 \
  --output micro_artifact/results/runs/single_qubit_result
```

For a temporary threshold study, pass another positive value such as
`--error-threshold 0.05`. The default remains `0.001`.

With no `--benchmark`, the producer selects 15 benchmarks: the 11 Table 4
benchmarks plus H2S, CO2, MgO, and NaCl. `--source generate` supports these
benchmarks through the configured Table 4 and scalability Hamiltonians.

## 2. Two-qubit result

This produces the `ncf-two` data used by the two-qubit comparison. Stored data
can be loaded directly:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --source existing \
  --benchmark LiH \
  --output micro_artifact/results/runs/two_qubit_result
```

To generate missing two-qubit results, Docker and the Synthetiq configuration
must be available:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --source generate \
  --benchmark LiH \
  --gpu 1 \
  --output micro_artifact/results/runs/two_qubit_result
```

The default set is the Table 4 set with unsupported H2S and CO2 excluded.

## 3. Analytical T-count estimation (Section 5.4)

This experiment does not compile circuits. It reads
`ncf_unitaries_generated` and `original_rz_gate_count` from a producer result
and evaluates the paper's analytical model over epsilon values.

Run the single-qubit model after producing the default single-qubit dataset:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.analytical_estimation \
  --method ncf-one \
  --benchmark LiH \
  --epsilon 0.1 \
  --epsilon 0.01 \
  --output micro_artifact/results/runs/analytical_estimation
```

Use `--method ncf-two` and the two-qubit producer for the two-qubit model.
Without `--epsilon`, the sweep is `10^-1` through `10^-9`. The model uses

```text
single-qubit unitary: 3 * log2(1 / epsilon)
two-qubit unitary:    15 * log_base_2.76(1 / epsilon)
NC-Fusion precision:  epsilon * original_RZ_count / NC-Fusion_unitary_count
```

## 4. Window-size sensitivity (Section 5.5)

This sweep always recompiles every requested window and does not save QASM:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.window_size_sensitivity \
  --benchmark LiH \
  --method ncf-one \
  --output micro_artifact/results/runs/window_size_sensitivity
```

With no selections, it uses the four configured benchmarks and both NC-Fusion
budgets. The configured single-qubit windows are `full, 128, 64, 32, 16, 8,
4`; two-qubit windows are `full, 256, 128, 64, 32, 16`.

## 5. Pauli-string order sensitivity (Section 5.6)

This experiment performs 30 randomized Pauli-order runs per selected
benchmark, always regenerates the circuits, and does not save QASM:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.pauli_string_order_sensitivity \
  --benchmark LiH \
  --output micro_artifact/results/runs/pauli_string_order_sensitivity
```

The default benchmark set is LiH, H2O, Ising-2D-60, and Heisenberg-2D-60.

## 6. Component ablation (Section 5.6.1)

The three variants are `anti_commuting_only`,
`anti_commuting_plus_commuting`, and `full_ncfusion`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.components_abalation \
  --benchmark LiH \
  --variant anti_commuting_only \
  --variant anti_commuting_plus_commuting \
  --variant full_ncfusion \
  --output micro_artifact/results/runs/components_abalation
```

If the default single-qubit producer contains LiH, the `full_ncfusion` row is
loaded from that producer QASM and retains its recorded compilation time. The
other variants are regenerated for the ablation. Ablation circuits are not
written to the canonical QASM input directory.

## 7. Precision ablation (Section 5.6.2)

This compares proportional total-error scaling with fixed per-unitary error:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.precision_abalation \
  --benchmark LiH \
  --method ncf-one \
  --output micro_artifact/results/runs/precision_abalation
```

For single-qubit NC-Fusion, the scaled-total-error arm reuses the producer
QASM when available. The fixed-per-unitary arm is regenerated. Selecting
`--method ncf-two` runs the two-qubit variants, which require the two-qubit
dependencies and are regenerated.

## 8. Trotter operator-norm error (Section 5.7.1)

This evaluates the unsynthesized rotation circuit `rz_qc` for GridSynth and
NC-Fusion:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.trotter_error \
  --benchmark LiH \
  --method gridsyn \
  --method ncf-one \
  --trotter-steps 1 \
  --trotter-steps 5 \
  --trotter-steps 10 \
  --trotter-steps 20 \
  --output micro_artifact/results/runs/trotter_error
```

For Trotter step 1, stored QASM and compilation time are reused when present.
The other step counts are compiled for the evaluation and are not saved as
canonical producer QASM.

## 9. Application-level fidelity (Section 5.7.2)

This evaluates the synthesized Clifford+T circuit `clifford_t_qc` at the
configured logical error rates:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.application_level_fidelity \
  --benchmark LiH \
  --method gridsyn \
  --method ncf-one \
  --logical-error 1e-6 \
  --logical-error 1e-7 \
  --output micro_artifact/results/runs/application_level_fidelity
```

As with the operator-norm evaluation, step 1 reuses existing circuits and
recorded compilation time; other Trotter steps are regenerated for the
evaluation.

## 10. T-count optimizer comparison (Section 5.9)

This workflow compares GridSynth, tzap, PyZX, and T-Optimizer. Build or clone
the external tools first, then run:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.t_count_methods_comparison \
  --benchmark H2 \
  --tzap-bin /path/to/tzap \
  --t-optimizer-root /path/to/T-Optimizer \
  --output micro_artifact/results/runs/t_count_methods_comparison
```

Repeat `--method` to select a subset of `gridsyn`, `tzap`, `pyzx`, and
`t-optimizer`. Existing GridSynth RZ and Clifford+T QASM are used as the
reference when available, including the recorded compilation time. The
optimizer intermediates are written under the run directory's `circuits/`
subdirectory, not under the canonical producer directory.

The default benchmarks are H2, LiH-reduced, Heisenberg-4, and Heisenberg-6.

## 11. Spacetime-volume analysis (Section 5.10)

Install the resource estimator at the pinned revision:

```bash
python3 -m pip install -e ".[resource]"
git clone https://github.com/Infleqtion/resource-superstaq.git
git -C resource-superstaq checkout 717cbbfc62e558be3f2f9acb512e992d3cd43529
python3 -m pip install -e resource-superstaq
```

Run it only on existing Clifford+T QASM:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.space_volume_analysis \
  --benchmark Ising-3D-30 \
  --method gridsyn \
  --method rustiq \
  --method ncf-one \
  --output micro_artifact/results/runs/space_volume_analysis
```

The estimator evaluates one and ten magic-state factories. It records
physical qubits, serial and parallel time, primitive moments, error, and
physical-qubit-time volume. It does not compile or save QASM. The full
configured set currently requires Rustiq baseline files for every selected
benchmark; if a baseline QASM is missing, restrict
`--benchmark` or provide the missing input first.

## Generic configured runner

The lower-level runner can execute the configured paper experiments directly:

```bash
PYTHONPATH=src:legacy:. python -m ncfusion run table4 \
  --benchmark LiH \
  --method gridsyn \
  --output micro_artifact/results/runs/table4_lih

PYTHONPATH=src:legacy:. python -m ncfusion run scalability \
  --benchmark Ising-2D-30 \
  --method ncf-one \
  --output micro_artifact/results/runs/scalability_is2d30
```

Configured experiment names are `table4`, `sensitivity`,
`error-evaluation`, `random-order`, `scalability`, `optimizer-comparison`,
and `spacetime-volume`. Prefer the dedicated modules above when an experiment
has one; they implement the artifact-specific reuse and output policy.

To compare a generated Table 4 CSV with the transcribed reference:

```bash
PYTHONPATH=src:legacy:. python -m ncfusion validate \
  micro_artifact/results/runs/table4/metrics.csv
```
