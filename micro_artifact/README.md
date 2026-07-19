# NC-Fusion MICRO artifact evaluations

Each requested paper evaluation has one Python module in this directory. Run
an available evaluation from the repository root with:

```bash
PYTHONPATH=src:. python -m micro_artifact.single_qubit_result \
  --benchmark LiH --output results/runs/single_qubit_result
```

The modules expose a `run(...)` function as well as the command-line entry
point. Shared output handling is in `common.py`; generated runs contain a
`manifest.json` and `metrics.csv`.

Analytical estimation follows Section 5.4 of the paper:

```bash
PYTHONPATH=src:. python -m micro_artifact.analytical_estimation \
  --method ncf-one --benchmark LiH --output results/runs/analytical-lih
```

Use repeated `--epsilon` options to select a smaller precision sweep. The
default sweep is `10^-1` through `10^-9`.

The precision ablation compares the two threshold policies directly:

```bash
PYTHONPATH=src:. python -m micro_artifact.precision_abalation \
  --benchmark LiH --output results/runs/precision-ablation
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
