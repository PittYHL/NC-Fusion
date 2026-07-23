# NC-Fusion MICRO artifact evaluations

Each requested paper evaluation has one Python module in this directory. Run
an available evaluation from the repository root with:

```bash
PYTHONPATH=src:. python -m micro_artifact.single_qubit_result \
  --benchmark LiH --source existing \
  --output micro_artifact/results/runs/single_qubit_result
```

Use `--source generate` to regenerate the single-qubit QASM and producer
records. The producer records `ncf_unitaries_generated`,
`original_rz_gate_count`, `compilation_time_seconds`, and the QASM paths.
The two-qubit producer has the same interface; use `--source generate` when
the two-qubit QASM is not already present:

```bash
PYTHONPATH=src:. python -m micro_artifact.two_qubit_result \
  --benchmark LiH --source generate
```

The modules expose a `run(...)` function as well as the command-line entry
point. Shared output handling is in `common.py`; generated runs contain a
`manifest.json` and `metrics.csv`.

Analytical estimation follows Section 5.4 of the paper:

```bash
PYTHONPATH=src:. python -m micro_artifact.analytical_estimation \
  --method ncf-one --benchmark LiH --output micro_artifact/results/runs/analytical-lih
```

Use repeated `--epsilon` options to select a smaller precision sweep. The
default sweep is `10^-1` through `10^-9`. Analytical estimation consumes the
producer CSV directly and does not recompile NC-Fusion.

The single-qubit CSV compares NC-Fusion with the stored GridSynth QASM for
T-count, T-depth, and Clifford-count reductions. It includes all 15 supplied
single-qubit benchmarks and compares against GridSynth, Rustiq, and Phoenix.
Rustiq and Phoenix have 13 available benchmark files; MgO and NaCl have no
corresponding files for those two baselines. T-depth follows the paper-script convention
`new_qc.depth(lambda gate: gate[0].name == "t")`; very large QASM files use a
streaming equivalent of that filtered-depth calculation to avoid materializing
an oversized Qiskit circuit. The CSV contains separate average rows for
GridSynth (15), Rustiq (13), and Phoenix (13).

The precision ablation compares the two threshold policies directly:

```bash
PYTHONPATH=src:. python -m micro_artifact.precision_abalation \
  --benchmark LiH --output micro_artifact/results/runs/precision-ablation
```

The T-count method comparison is an external-tool workflow:

```bash
PYTHONPATH=src:. python -m micro_artifact.t_count_methods_comparison \
  --benchmark H2 \
  --tzap-bin /path/to/tzap \
  --t-optimizer-root /path/to/T-Optimizer
```

It runs tzap on the original Clifford+RZ circuit, synthesizes the resulting
RZ gates with GridSynth, and runs tzap again. PyZX is run only on the
GridSynth Clifford+T circuit, as in `pyzx_path.py`. T-Optimizer receives the
GridSynth Clifford+T circuit through `from_qiskit.py`.

Availability and missing-source details are listed in
[`MISSING_IMPLEMENTATIONS.md`](MISSING_IMPLEMENTATIONS.md).

The two sensitivity studies (`window_size_sensitivity` and
`pauli_string_order_sensitivity`) always regenerate their circuits and run
without saving QASM. The two ablations reuse the full NC-Fusion arm from the
single-qubit producer whenever that QASM is available; their other arms are
regenerated. Other evaluations prefer stored QASM and producer records,
including the recorded compilation time, before compiling new circuits.

The Section 5.10 spacetime-volume driver uses the original stored QASM files
and Infleqtion's `resource-superstaq` estimator:

```bash
PYTHONPATH=src:. python -m micro_artifact.space_volume_analysis \
  --benchmark Ising-3D-30 --output micro_artifact/results/runs/spacetime-volume
```

Install the Python dependencies with `pip install -e ".[resource]"`, then
install the pinned `resource-superstaq` revision documented in the repository
README. The driver evaluates `grid`, `rustiq`, and `ncf` at one and ten
magic-state factories.
