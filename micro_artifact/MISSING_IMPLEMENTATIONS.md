# Missing evaluation implementations

The requested module names are now present under `micro_artifact/`. The
following modules are intentionally explicit about their status:

| Module | Status | What is missing |
| --- | --- | --- |
| `space_volume_analysis.py` | Available | Loads the stored `grid` and `ncf` Clifford+T QASM files for 13 benchmarks, runs Infleqtion `resource-superstaq` with one and ten T factories, and writes NC-Fusion/GridSynth relative volume. |

The configuration also names `MgO` and `NaCl` for this section, but their
Clifford+T QASM inputs are not present in the current checkout. A full run
fails before estimation and reports those missing paths; provide those files
to reproduce the complete scalability set.

The following modules are wired to existing configured experiments and reuse
producer data where applicable:

* `analytical_estimation.py` (paper Section 5.4 formulas consuming producer
  unitary/RZ counts)
* `components_abalation.py` (paper Section 5.6.1, using the three grouping/
  and scheduling variants)
* `precision_abalation.py` (paper Section 5.6.2, `fix_error_threshold=0`
  versus `1`)
* `trotter_error.py` (paper Section 5.7.1, using stored `c+t` at step 1 and generated `rz` circuits otherwise)
* `application_level_fidelity.py` (paper Section 5.7.2, using
  `clifford_t_qc`)
* `single_qubit_result.py`
* `two_qubit_result.py` (requires the Synthetiq Docker setup for `ncf-two`)
* `window_size_sensitivity.py`
* `pauli_string_order_sensitivity.py`
* `t_count_methods_comparison.py`

The remaining input gap is the missing `MgO` and `NaCl` spacetime-volume QASM
set noted above; the evaluator reports those paths before starting a full run.
