# NC-Fusion Python API

`NC_Fusion` accepts a Qiskit `SparsePauliOp`-compatible Hamiltonian and
returns its NC-Fusion rotation circuit and, when requested, its Clifford+T
circuit:

```python
from ncfusion import NC_Fusion

compiled_qc, clifford_t_qc = NC_Fusion(
    hamiltonian,
    budget=1,
    window=4,
    error_threshold=1e-3,
    t_budget=60,
    gpu=0,
    trotter_steps=1,
    evolution_time=1.0,
    synthesize=True,
    use_gridsynth=False,
    fix_error_threshold=False,
)
```

Use `budget=1` for single-qubit NC-Fusion and `budget=2` for two-qubit
NC-Fusion. Set `synthesize=False` to return only the rotation circuit; the
second return value is then `None`.

`gpu=0` uses CPU synthesis. Set `gpu=1` to forward `gpu=1` to Trasyn and
enable GPU synthesis.

See `ncfusion.api.NC_Fusion` for the complete API definition.
