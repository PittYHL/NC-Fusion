# Missing evaluation implementations

The requested module names are now present under `micro_artifact/`. The
following modules are intentionally explicit about their status:

| Module | Status | What is missing |
| --- | --- | --- |
| `space_volume_analysis.py` | Partial | The fault-tolerant spacetime-volume model. The configuration contains the benchmark list and factory counts, but the current runner emits circuit metrics only. |

The following modules are wired to existing configured experiments:

* `analytical_estimation.py` (paper Section 5.4 formulas plus unsynthesized
  NC-Fusion unitary counts)
* `components_abalation.py` (paper Section 5.6.1, using the three grouping/
  scheduling variants from `main_alg.py`)
* `precision_abalation.py` (paper Section 5.6.2, `fix_error_threshold=0`
  versus `1`)
* `trotter_error.py` (paper Section 5.7.1, using `rz_qc`)
* `application_level_fidelity.py` (paper Section 5.7.2, using
  `clifford_t_qc`)
* `single_qubit_result.py`
* `two_qubit_result.py` (requires the Synthetiq Docker setup for `ncf-two`)
* `window_size_sensitivity.py`
* `pauli_string_order_sensitivity.py`
* `t_count_methods_comparison.py`

Please upload the missing evaluation scripts, formulas, or parameter tables
for the partial/missing entries. The files can then be replaced without
changing the public `NC_Fusion` API.
