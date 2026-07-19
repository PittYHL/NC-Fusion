# NC-Fusion

Reproducibility project for **NC-Fusion: Optimizing T-Gate Cost for
Hamiltonian Simulation**.

The project is organized for GitHub and MICRO-style artifact evaluation:

```text
NC-Fusion/
├── circuits/                 # empty by design; add generated QASM files here
├── configs/paper.json        # paper workloads and experiment parameters
├── docs/                     # artifact appendix and submission documentation
├── legacy/                   # cleaned research implementation and Phoenix
├── micro_artifact/           # one entry point per paper evaluation
├── paper/                    # local paper copy; ignored by Git
├── results/reference/        # reference values transcribed from Table 4
├── results/runs/             # generated outputs; ignored except .gitkeep
└── src/ncfusion/             # stable CLI and reproducibility harness
```

The format follows the [MICRO Artifact Evaluation requirements](https://www.microarch.org/micro59/submit/artifacts.php), including metadata, access, dependencies, workflow, evaluation steps, and results.

## Quick validation

From the `NC-Fusion` directory:

```bash
make smoke
```

This runs without Qiskit and writes `results/smoke/smoke.json`.

List all configured workloads:

```bash
make list
```

## Install the full environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-paper.txt
```

The full experiments use Qiskit Nature/PySCF, Gridsynth, Trasyn, Rustiq,
Phoenix, PyZX, and the Synthetiq container for two-qubit synthesis. The exact
environment used for a run is recorded in its manifest.

The optimizer comparison also uses [tzap](https://github.com/qqq-wisc/tzap)
and [T-Optimizer](https://github.com/iqubit-org/T-Optimizer). Build tzap and
clone T-Optimizer outside this repository, then pass their locations when
running the comparison:

```bash
PYTHONPATH=src:. python -m micro_artifact.t_count_methods_comparison \
  --benchmark H2 \
  --tzap-bin /path/to/tzap \
  --t-optimizer-root /path/to/T-Optimizer \
  --output results/runs/t-count-methods
```

The comparison records the original Clifford+RZ circuit, the GridSynth
Clifford+T baseline, tzap before and after RZ synthesis, PyZX on the GridSynth
Clifford+T circuit, and T-Optimizer applied to the GridSynth circuit. Use
repeated `--method` options (`gridsyn`, `tzap`, `pyzx`, or `t-optimizer`) to
run only a subset.

## Run experiments

Start with one benchmark and one method:

```bash
make run EXPERIMENT=table4 BENCHMARK=LiH METHOD=gridsyn OUTPUT=results/runs/lih
```

Run the complete configured main comparison:

```bash
make run EXPERIMENT=table4 OUTPUT=results/runs/table4
```

Available experiments include `table4`, `sensitivity`, `error-evaluation`,
`random-order`, `scalability`, `optimizer-comparison`, and
`spacetime-volume`. Use `make list` for the benchmark and method sets.

The evaluation-specific entry points are in
[`micro_artifact/`](micro_artifact/). For example:

```bash
PYTHONPATH=src:. python -m micro_artifact.window_size_sensitivity \
  --benchmark LiH --output results/runs/window-size
```

See [`micro_artifact/MISSING_IMPLEMENTATIONS.md`](micro_artifact/MISSING_IMPLEMENTATIONS.md)
for evaluations whose original driver or formula is not present yet.

Validate a generated main-table CSV against the reference:

```bash
PYTHONPATH=src python -m ncfusion validate \
  results/runs/table4/metrics.csv
```

## QASM files

`circuits/` is intentionally empty except for `.gitkeep`. Add the generated
QASM files there before uploading the project to GitHub. Generated run outputs
under `results/runs/` and local caches are ignored by Git.

## Paper PDF

The supplied PDF is marked confidential. It is therefore not copied into this
public-project folder and `paper/NC_Fusion.pdf` is ignored. Keep it locally if
needed for evaluation, or add a distributable camera-ready version explicitly.

See [docs/ARTIFACT_APPENDIX.md](docs/ARTIFACT_APPENDIX.md) for the submission-
ready artifact appendix.

See [legacy/README.md](legacy/README.md) for a map of the retained research
modules and the artifact execution path.

## Python API

The main project entry point is `NC_Fusion`. It accepts a Qiskit
`SparsePauliOp`-compatible Hamiltonian and returns the compiled circuit with
rotations plus its Clifford+T circuit:

```python
from ncfusion import NC_Fusion

compiled_qc, clifford_t_qc = NC_Fusion(
    hamiltonian,
    budget=1,
    error_threshold=1e-3,
    trotter_steps=1,
    synthesize=True,
    evolution_time=1.0,
    t_budget=60,
    fix_error_threshold=False,
)
```

Use `budget=1` for single-qubit NC-Fusion or `budget=2` for two-qubit
NC-Fusion. Set `synthesize=False` to return only the compiled rotation
circuit; in that case the second return value is `None`.

See `ncfusion.api.NC_Fusion` for the complete list of optional parameters.
