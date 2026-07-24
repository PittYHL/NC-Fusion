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
- The optimizer comparison uses PyZX plus the external T-Zap and T-Optimizer
  tools. They can be installed manually or with the opt-in installer below.
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

### CSV checkpoint and merge policy

Artifact CSV output uses merge semantics rather than blindly appending or
discarding the previous file. A rerun replaces the row for the same
configuration key and appends rows for new benchmarks, precisions, Trotter
steps, windows, methods, or random-run IDs. Existing rows outside the current
selection are retained. The main keys are:

- producer and ablation reports: benchmark plus method/variant;
- analytical estimation: method plus precision;
- error evaluations: benchmark, method, Trotter step, and logical-error rate;
- optimizer comparisons: benchmark plus method/stage;
- spacetime volume: benchmark, method, and factory count;
- random-order sensitivity: benchmark plus run ID.

`relative_metrics.csv`, `summary.csv`, plot inputs, and aggregate rows are
deterministic reports rebuilt from the retained raw rows; they do not create
duplicate averages. Each long-running evaluator checkpoints its merged raw
CSV after completed work.

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
They read stored QASM by default. Pass `--source generate` to regenerate the
selected producer inputs. Downstream analytical estimation reads the producer
CSV at the fixed path `micro_artifact/results/runs/single_qubit_result` or
`two_qubit_result`; use those default output directories when the result will
be consumed by another experiment.

The single-qubit producer records the NC-Fusion unitary count,
`ncf_unitaries_generated`, the original RZ count,
`original_rz_gate_count`, and `compilation_time_seconds`. Analytical
estimation uses the first two counts directly. Existing-data workflows reuse
the stored QASM and recorded compilation time whenever both are available;
generation is performed only when explicitly requested.

Sensitivity and ablation experiments read stored result CSVs by default. Pass
`--source generate` to rerun them. Generated sensitivity circuits remain
in-memory only; ablation circuits are not written to the canonical QASM
directory. The scheduling ablation can reuse the single-qubit producer QASM.

## 1. Single-qubit result (Section 5.2)

This is the main single-qubit producer and Table 4 metric report. It reports
T count, T depth, Clifford count, compilation time, original RZ count, and
NC-Fusion unitary count. It also reports reductions relative to GridSynth,
Rustiq, and Phoenix. Rustiq and Phoenix have 13 supplied benchmark files;
MgO and NaCl are excluded from those two comparisons.

For single-qubit NC-Fusion, `ncf_unitaries_generated` retains the original
compressor's combined rotation count: one for each nonempty transformed group
plus one for every commuting Pauli rotation. The separated
`ncf_unitaries_generated` and `ncf_rz_generated` fields apply only to the
two-qubit analytical estimate.
`original_rz_gate_count` follows the original NC-Fusion implementation and is
the non-identity Pauli count, not the literal number of transpiled QASM `rz`
instructions.

T depth uses the paper definition:

```python
new_qc.depth(lambda gate: gate[0].name == "t" or gate[0].name == "tdg")
```

Use stored QASM:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.single_qubit_result \
  --source existing \
  --benchmark LiH \
  --benchmark Ising-2D-30 \
  --output micro_artifact/results/runs/single_qubit_result
```

Use `--method` to refresh only one comparison arm for a selected benchmark.
The NC-Fusion circuit remains the candidate: `rustiq` and `phoenix` refresh
only their corresponding comparison, while `ncf-one` refreshes the default
GridSynth comparison. Repeat `--method` for multiple arms; omit it to run all
GridSynth, Rustiq, and Phoenix comparisons. Previously recorded comparison
fields for the benchmark are retained when a partial method run is merged.

For example:

```bash
# Only LiH versus Rustiq
PYTHONPATH=src:legacy:. python -m micro_artifact.single_qubit_result \
  --source existing \
  --benchmark LiH \
  --method rustiq \
  --output micro_artifact/results/runs/single_qubit_result

# Only LiH NC-Fusion versus GridSynth
PYTHONPATH=src:legacy:. python -m micro_artifact.single_qubit_result \
  --source existing \
  --benchmark LiH \
  --method ncf-one \
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

## 2. Two-qubit result (Section 5.3)

This produces the Table 4 comparison for GridSynth, Rustiq, single-qubit
NC-Fusion, and two-qubit NC-Fusion by following
reference same-error workflow. It records T-count, T-depth, and Clifford count
for all 11 Table 4 benchmarks and reports the average reduction of NCF-two
relative to each other method. The NCF-two threshold is per rotation: `0.11`
for Ising benchmarks and `0.12` for the other benchmarks. Existing QASM is
used whenever available; missing circuits are generated and saved.

By default the runner reads the existing QASM files and does not synthesize
missing inputs. Run the stored-data workflow with:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --output micro_artifact/results/runs/two_qubit_result
```

Select one or more benchmarks with repeated `--benchmark` options. Short
directory names such as `IS-2D-30` and `Hei-2D-60` are accepted and mapped to
their canonical benchmark names. Existing rows for other benchmarks remain in
the merged CSV:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --source existing \
  --benchmark LiH \
  --benchmark IS-2D-30 \
  --output micro_artifact/results/runs/two_qubit_result
```

To regenerate all selected inputs, use `--source generate`. The optional
`--source auto` mode reuses files when present and generates missing inputs.
Docker and the Synthetiq configuration are required when an NCF-two circuit
must be generated:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --source auto \
  --gpu 1 \
  --output micro_artifact/results/runs/two_qubit_result
```

The output contains one row per benchmark and an `AVERAGE_11` row. Average
reductions use `100 * (other method metric - NCF-two metric) / other method
metric`.

## 3. Analytical T-count estimation (Section 5.4)

This experiment does not compile circuits. It reads
`ncf_unitaries_generated` and `original_rz_gate_count` from the single-qubit
producer, and all three fields—including `ncf_rz_generated`—from the
two-qubit producer. The separated fields apply only to the two-qubit model.
It evaluates the paper's analytical model over epsilon values. By default it
averages the 13 Rustiq/Phoenix benchmarks: the 11 Table 4 benchmarks plus H2S
and CO2. MgO and NaCl are excluded. The output contains only average
T-count reduction for each of the nine precisions. Precision is the x-axis;
average GridSynth and NC-Fusion T-gate counts are used internally but are not
emitted in the CSV. The default run also consumes the two-qubit producer for
the 13 analytical benchmarks and writes one estimated-T-count figure for
GridSynth, single-qubit NC-Fusion, and two-qubit NC-Fusion, plus one reduction
figure for the two NC-Fusion methods.

If `two_qubit_result/metrics.csv` is absent, this analytical command creates
a fill-in template for the required two-qubit metadata rows. It does not run
Synthetiq, perform synthesis, or write QASM; fill
`ncf_unitaries_generated`, `ncf_rz_generated`, and `original_rz_gate_count`
before rerunning.

Run the single-qubit model after producing the default single-qubit dataset:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.analytical_estimation \
  --method ncf-one \
  --output micro_artifact/results/runs/analytical_estimation
```

The default run uses both producer datasets and writes both analytical
figures. To run only the single-qubit model, pass `--method ncf-one`; to run
only the two-qubit model, pass `--method ncf-two`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.analytical_estimation \
  --method ncf-one \
  --benchmark LiH \
  --output micro_artifact/results/runs/analytical_estimation
```

Use `--method ncf-two` and the two-qubit producer for the two-qubit model.
The analytical sweep always evaluates the fixed nine precisions from `10^-1`
through `10^-9`; precision selection is not a command-line option.

The average reduction table is written to `metrics.csv`. The figures are
`estimated_t_count.png` and `t_count_reduction.png`. The full per-benchmark
estimates and average gate counts are not emitted in the artifact result. NC-Fusion uses the scaled precision separately for each benchmark before the averages are calculated.

## 4. Window-size sensitivity (Section 5.5)

This sweep does not save QASM. Use `--source generate` to recompile every
requested window:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.window_size_sensitivity \
  --benchmark LiH \
  --method ncf-one \
  --source generate \
  --output micro_artifact/results/runs/window_size_sensitivity
```

With no selections, it evaluates LiH, H2O, Ising-2D-60, and
Heisenberg-2D-60 for both NC-Fusion budgets. Single-qubit uses windows
`full, 128, 64, 32, 16, 8, 4` at threshold `0.005`; two-qubit uses windows
`full, 256, 128, 64, 32, 16` at threshold `0.03` for LiH/H2O and `0.07` for
the two 60-qubit spin benchmarks. By default, stored sweep results are read;
use `--source generate` to regenerate circuits in memory. Generated runs do
not save QASM. Rows are checkpointed after each window. `relative_metrics.csv`
reports T-count, T-depth, Clifford-count, and compilation-time percentages
relative to the corresponding `full` window, including per-window averages.

## 5. Pauli-string order sensitivity (Section 5.6)

This experiment follows the reference randomized-order workflow with a single-qubit
window of 4. Each invocation appends new randomized runs to
`metrics.csv`, checkpoints after every completed benchmark, and recomputes
`summary.csv`; no randomized QASM is saved. The summary reports the average,
minimum, maximum, and population standard deviation of the T-count, T-depth,
and Clifford-count reductions relative to the stored GridSynth circuit.
Run one new order per benchmark with:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.pauli_string_order_sensitivity \
  --source generate \
  --output micro_artifact/results/runs/pauli_string_order_sensitivity
```

The default benchmarks are LiH, H2O, Ising-2D-60, and Heisenberg-2D-60.
Aliases `IS-2D-60` and `Hei-2D-60` are accepted. Select benchmarks and add
multiple new orders per invocation with, for example:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.pauli_string_order_sensitivity \
  --source generate \
  --benchmark LiH \
  --benchmark H2O \
  --runs 10 \
  --precision 0.001 \
  --seed 100 \
  --gpu 1 \
  --output micro_artifact/results/runs/pauli_string_order_sensitivity
```

By default, existing randomized runs are read. Use `--source generate` to
append new run IDs; repeated generated commands preserve existing rows. The
raw records are in `metrics.csv`; the per-benchmark and accumulated `Average`
statistics are in `summary.csv`. The source-script precision is `0.001`; use
`--precision 0.02` only when intentionally testing a different synthesis
threshold.

## 6. Component ablation (Section 5.6.1)

The three variants are `scheduling`, `commuting-grouping`, and
`anti-commuting-grouping`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.components_abalation \
  --source generate \
  --benchmark LiH \
  --variant scheduling \
  --variant commuting-grouping \
  --variant anti-commuting-grouping \
  --output micro_artifact/results/runs/components_abalation
```

If the default single-qubit producer contains LiH, the `scheduling` row is
loaded from that producer QASM and retains its recorded compilation time. The
The `commuting-grouping` arm skips the independent-group reordering stage, and
the `anti-commuting-grouping` arm skips both commuting grouping and reordering.
By default, stored ablation results are read. Use `--source generate` to
regenerate the variants. Ablation circuits are not written to the canonical
QASM input directory. Results are merged by
benchmark, variant, budget, window, Trotter settings, threshold, and
Pauli-order seed.
The run also writes `relative_metrics.csv` and
`components_abalation_relative_reductions.png`. These normalize each
component's GridSynth reduction to the scheduling (NCF-one) reduction, so
Scheduling is 100% for every metric; averages are included in the relative
CSV and plot. The plot follows the Precision-ablation style: the Scheduling
100% reference is the wide bar and the other configurations are overlaid as
narrower bars, with a shared legend above the three metric panels.

## 7. Precision ablation (Section 5.6.2)

This compares proportional total-error scaling with fixed per-unitary error
using the stored QASM files. It reports T-count, T-depth, and Clifford-count
reductions relative to GridSynth for the 11 Table 4 benchmarks, plus the
average. It also reports the no-scaling reduction as a percentage of the
scaled reduction for each metric.

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.precision_abalation \
  --benchmark LiH \
  --output micro_artifact/results/runs/precision_abalation
```

The scaled-total-error arm uses `*_ncf_c+t.qasm`; the fixed-per-unitary arm
uses `*_ncf_fix_error_c+t.qasm`; and GridSynth uses `*_grid_c+t.qasm`. This
experiment does not run synthesis or write QASM. It writes
`precision_ablation_reductions.png`, a three-panel plot corresponding to
Fig. 11(d), (e), and (f).

## 8. Trotter operator-norm error (Section 5.7.1)

This evaluates the stored Clifford+T (`c+t`) circuit for GridSynth and
NC-Fusion at Trotter step 1. Other step counts regenerate the unsynthesized
rotation circuit `rz_qc`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.trotter_error \
  --source generate \
  --benchmark LiH \
  --method gridsyn \
  --method ncf-one \
  --trotter-steps 1 \
  --trotter-steps 5 \
  --trotter-steps 10 \
  --trotter-steps 20 \
  --output micro_artifact/results/runs/trotter_error
```

By default, stored circuits are read. Use `--source generate` to regenerate
all selected Trotter circuits. For Trotter step 1, stored `c+t.qasm` is reused
when present. The uploaded `micro_artifact/results/runs/trotter_error/U_exact.npy` is used as the exact
unitary cache. Other step counts are compiled for the evaluation and are not
saved as canonical producer QASM, but both generated forms are saved under
`trotter_error/circuits/<benchmark>/` with `trotter_steps_<N>` in each
filename. This lets the application-fidelity experiment reuse the exact
matching Trotter circuit.

`metrics.csv` is merged and checkpointed after each completed configuration.
Rerunning a configuration replaces only the row with the same benchmark,
method, and `trotter_steps`; existing rows for other configurations are kept.
This operator-norm experiment has no logical-error parameter because it
computes the deterministic spectral-norm error.

## 9. Application-level fidelity (Section 5.7.2)

This evaluates the synthesized Clifford+T circuit `clifford_t_qc` at the
configured logical error rates:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.application_level_fidelity \
  --source generate \
  --benchmark LiH \
  --method gridsyn \
  --method ncf-one \
  --logical-error 1e-6 \
  --logical-error 1e-7 \
  --output micro_artifact/results/runs/application_level_fidelity
```

By default, stored circuits are read. Use `--source generate` to regenerate
all selected circuits. Generated runs save both RZ and Clifford+T circuits
under the fidelity run directory. In all cases, `evolution_time` remains 1.0 while
`Trotter_steps` is the selected step count.

This experiment runs LiH at Trotter steps 1, 5, 10, and 20 by default. GPU
execution is also enabled by default; pass `--gpu 0` to use the CPU. The
`metrics.csv` report contains the `noisy_fidelity` values returned by
`legacy.error_evaluation.density_matrix_error_from_hamiltonian`, one row per
logical-error rate and Trotter step. The default logical-error rates are
`1e-6` and `1e-7`; use repeated `--logical-error` options to select rates.
Its CSV is merged and checkpointed after each result using the key
`(benchmark, method, trotter_steps, logical_error_rate)`, so reruns replace
only matching rows and preserve all other results.

## 10. T-count optimizer comparison (Section 5.9)

This workflow compares GridSynth, T-Zap, Pauli rotation merging (T-optimizer),
and PyZX. PyZX is included in `.[paper]` and can also be installed through the
dedicated `.[optimizers]` extra. T-Zap is a Rust executable, while
T-Optimizer is a source checkout that also needs QuaEC, NumPy, `gmpy2`, and
Cython. Install all missing optimizer dependencies automatically with:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.install_optimizers
```

This places the T-Optimizer checkout under `micro_artifact/.external/` and
prints the resulting paths. The installer requires network access, Git, and
Rust/Cargo for T-Zap. It is never invoked by existing-result workflows. To
install only selected tools, repeat `--method` with `pyzx`, `tzap`, or
`t-optimizer`.

Alternatively, install the tools manually:

```bash
python3 -m pip install -e ".[paper,optimizers]"
cargo install tzap-opt
git clone https://github.com/iqubit-org/T-Optimizer micro_artifact/.external/T-Optimizer
git clone https://github.com/cgranade/python-quaec micro_artifact/.external/QuaEC
python3 -m pip install micro_artifact/.external/QuaEC numpy gmpy2 Cython
```

Omitting `--benchmark` runs all 13 benchmarks: the
11 Table 4 benchmarks plus H2S and CO2. Use one output directory so all
intermediate files and the final relative report remain under
`t_count_optimizer_relative`:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.t_count_methods_comparison \
  --source generate \
  --method tzap \
  --method t-optimizer \
  --method pyzx \
  --install-missing \
  --output micro_artifact/results/runs/t_count_optimizer_relative
```

`--install-missing` is an opt-in shortcut for generated runs; it installs only
the selected missing tools before starting. When using manually installed
tools, omit `--install-missing` and pass `--tzap-bin` and
`--t-optimizer-root` instead. Repeat `--method` to select a subset of `gridsyn`, `tzap`, `t-optimizer`, and
`pyzx`; repeat `--benchmark` to select a subset of the 13. Full benchmark names
and the short aliases `IS-2D-*` and `Hei-2D-*` are accepted. For example, this
runs only LiH and Ising-2D-30:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.t_count_methods_comparison \
  --source generate \
  --benchmark LiH \
  --benchmark IS-2D-30 \
  --method tzap \
  --output micro_artifact/results/runs/t_count_optimizer_relative
```

Existing results
and GridSynth QASM are read by default. Use `--source generate` to rerun the
selected optimizer workflows. The optimizer intermediates are written under the
run directory's `circuits/` subdirectory, not under the canonical producer
directory. When T-Zap is selected, `tzap_reductions.csv` reports per-benchmark
and arithmetic-average T-count, T-depth, and Clifford-count reductions versus
fresh GridSynth synthesis of the same stored `grid_rz` QASM. The T-Zap path is
`grid_rz -> tzap pre -> per-RZ GridSynth synthesis -> tzap post`; T-count and
T-depth both count `t` and `tdg`, matching `ncf/NCF/tzap_test.py`.
The PyZX arm follows the reference PyZX workflow: it reads `grid_c+t` QASM,
runs `full_reduce`, removes extracted `swap` lines, and converts only exact
`pi/4`-multiple RZ phases back to Clifford+T before measuring reductions.
PyZX on H2S and CO2 can take more than one day. CO2 may also fail during
reduction; such a failure is recorded per benchmark and the relative report
leaves that PyZX row blank instead of aborting the remaining benchmarks.

For the consolidated relative report over all 13 benchmarks, run:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.t_count_optimizer_relative
```

The consolidated report also accepts repeated `--benchmark` options, for
example `--benchmark LiH --benchmark Hei-2D-60`; omitted options process all 13
benchmarks. Unselected rows already present in the CSV are preserved.

This compares T-Zap, Pauli rotation merging (T-optimizer), PyZX, and
single-qubit NC-Fusion against
the stored GridSynth `grid_c+t` circuit. Each metric is reported as
`100 * method / GridSynth`, capped at 100% for display. Unfinished methods
remain blank, and each average uses only that method's available rows. The
single plot contains only the four method averages, with the three metrics
grouped as in Fig. 14.


## 11. Spacetime-volume analysis (Section 5.10)

Install the resource estimator at the pinned revision:

```bash
python3 -m pip install -e ".[resource]"
git clone https://github.com/Infleqtion/resource-superstaq.git
git -C resource-superstaq checkout 717cbbfc62e558be3f2f9acb512e992d3cd43529
python3 -m pip install -e resource-superstaq
```

Run it on the existing single-qubit GridSynth and NC-Fusion Clifford+T QASM
for all 13 configured benchmarks:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.space_volume_analysis \
  --source generate \
  --output micro_artifact/results/runs/space_volume_analysis
```

Select one or more benchmarks with repeated `--benchmark` options. Full names
and short aliases such as `IS-2D-30` and `Hei-2D-60` are accepted; omitted
options retain the default of all 13 benchmarks:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.space_volume_analysis \
  --source existing \
  --benchmark LiH \
  --benchmark IS-2D-30 \
  --output micro_artifact/results/runs/space_volume_analysis
```

By default, existing volume estimates are read. Use `--source generate` to
recompute them. The estimator evaluates one and ten magic-state factories using
`<benchmark>_grid_c+t.qasm` and `<benchmark>_ncf_c+t.qasm`. It records
physical qubits, serial and parallel time, primitive moments, error, and
physical-qubit-time volume in `metrics.csv`. The relative report in
`relative_metrics.csv` contains NC-Fusion/GridSynth spacetime volume for each
benchmark and an `AVERAGE_13` row for each factory count. The average is over
benchmarks with valid estimates. The CSV files and manifest are checkpointed
after each completed benchmark, so an interrupted run retains the completed
prefix instead of waiting for all 13 benchmarks.

## Generic configured runner

The lower-level runner can execute the configured paper experiments directly:

```bash
PYTHONPATH=src:legacy:. python -m ncfusion run table4 \
  --benchmark LiH \
  --method gridsyn \
  --source existing \
  --output micro_artifact/results/runs/table4_lih

PYTHONPATH=src:legacy:. python -m ncfusion run scalability \
  --benchmark Ising-2D-30 \
  --method ncf-one \
  --source generate \
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
