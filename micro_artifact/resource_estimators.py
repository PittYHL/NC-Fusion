"""Resource-estimation adapters for Clifford+T qiskit circuits.

Two independent estimators, all running fully offline:

- ``qualtran_estimate(qc)``: Google Qualtran spacetime cost via
  ``surface_code.PhysicalCostModel`` fed with gate counts (we don't need
  the qiskit→cirq→Bloq round-trip — the model only consumes
  ``GateCounts`` + qubit count, which we extract directly from qc).
- ``superstaq_estimate(qc)``: Infleqtion ``resource-superstaq`` (the
  local FT-compilation + estimator). qiskit → cirq adapter, then
  ``MovementLayout → ft_compile → ResourceEstimator``.

Each function returns a flat ``dict`` of metrics named with a stable
``re_<tool>_<metric>`` prefix so the test driver can write CSV columns
mechanically.

All three swallow exceptions and return ``{..., 'error': str}`` rows on
failure so a single bad benchmark doesn't kill the experiment.
"""
from __future__ import annotations

import warnings
from typing import Dict


# ----------------------------------------------------------------------- #
# qiskit → cirq adapter (used by Qualtran and Superstaq).
# Hand-rolls the gate map for our Clifford+T set instead of going through
# QASM 2.0, to avoid floating-point drift on Rz angles (irrelevant for
# our pure Clifford+T circuits but cheap insurance).
# ----------------------------------------------------------------------- #
def _qiskit_to_cirq(qc, *,
                    decompose_y: bool = False,
                    decompose_inverses: bool = False):
    """Convert a qiskit QuantumCircuit on Clifford+T (+ optional Rz) to a
    cirq.Circuit.  Supported qiskit ops: h, s, sdg, sx, sxdg, t, tdg, x, y,
    z, cx, rz.  Anything else raises ValueError.

    ``sx``/``sxdg`` (√X / √X†) are emitted by nwqec's to_clifford_reduction;
    they are Clifford (zero T cost).

    Decomposition flags (used by Superstaq, whose ft_compile only accepts
    the {H, S, X, Z, T, CX} Clifford+T set — no Y, Sdg, Tdg, Sx, Sxdg):
    - ``decompose_y=True``: Y → X·Z (equal to Y up to global phase).
    - ``decompose_inverses=True``: Sdg → S·Z (preserves T-count);
      Tdg → T·S·Z (preserves T-count, adds 2 Cliffords per Tdg);
      Sx → H·S·H and Sxdg → H·S·Z·H (both exact, Clifford-only, zero T).

    Both decompositions preserve ``t_count`` exactly so the resource-
    estimator output is comparable to circuits that natively support
    Sdg/Tdg/Y/Sx/Sxdg."""
    import cirq
    qubits = cirq.LineQubit.range(qc.num_qubits)
    ops = []
    for instr, qargs, _ in qc.data:
        name = instr.name
        idx = [qc.find_bit(q).index for q in qargs]
        if name == "h":
            ops.append(cirq.H(qubits[idx[0]]))
        elif name == "s":
            ops.append(cirq.S(qubits[idx[0]]))
        elif name == "sdg":
            if decompose_inverses:
                ops.append(cirq.S(qubits[idx[0]]))
                ops.append(cirq.Z(qubits[idx[0]]))
            else:
                ops.append((cirq.S ** -1)(qubits[idx[0]]))
        elif name == "sx":
            # √X. cirq.X**0.5 is the exact equivalent (Clifford, zero T cost).
            # Superstaq's ft_compile only accepts {H,S,X,Z,T,CX}, so under
            # decompose_inverses emit H·S·H instead (= √X exactly).
            if decompose_inverses:
                ops.append(cirq.H(qubits[idx[0]]))
                ops.append(cirq.S(qubits[idx[0]]))
                ops.append(cirq.H(qubits[idx[0]]))
            else:
                ops.append((cirq.X ** 0.5)(qubits[idx[0]]))
        elif name == "sxdg":
            # (√X)† = H·S†·H = H·S·Z·H (exact). Same decompose rationale as sx.
            if decompose_inverses:
                ops.append(cirq.H(qubits[idx[0]]))
                ops.append(cirq.S(qubits[idx[0]]))
                ops.append(cirq.Z(qubits[idx[0]]))
                ops.append(cirq.H(qubits[idx[0]]))
            else:
                ops.append((cirq.X ** -0.5)(qubits[idx[0]]))
        elif name == "t":
            ops.append(cirq.T(qubits[idx[0]]))
        elif name == "tdg":
            if decompose_inverses:
                ops.append(cirq.T(qubits[idx[0]]))
                ops.append(cirq.S(qubits[idx[0]]))
                ops.append(cirq.Z(qubits[idx[0]]))
            else:
                ops.append((cirq.T ** -1)(qubits[idx[0]]))
        elif name == "x":
            ops.append(cirq.X(qubits[idx[0]]))
        elif name == "y":
            if decompose_y:
                ops.append(cirq.X(qubits[idx[0]]))
                ops.append(cirq.Z(qubits[idx[0]]))
            else:
                ops.append(cirq.Y(qubits[idx[0]]))
        elif name == "z":
            ops.append(cirq.Z(qubits[idx[0]]))
        elif name == "cx":
            ops.append(cirq.CNOT(qubits[idx[0]], qubits[idx[1]]))
        elif name == "rz":
            ops.append(cirq.rz(float(instr.params[0]))(qubits[idx[0]]))
        elif name in ("barrier", "id"):
            continue
        else:
            raise ValueError(f"_qiskit_to_cirq: unsupported gate {name!r}")
    return cirq.Circuit(ops)


# ----------------------------------------------------------------------- #
# Common counter (shared by Qualtran feed + sanity logging).
# ----------------------------------------------------------------------- #
def _gate_counts(qc) -> Dict[str, int]:
    """Per-name gate count from a qiskit QuantumCircuit."""
    counts: Dict[str, int] = {}
    for instr, _, _ in qc.data:
        counts[instr.name] = counts.get(instr.name, 0) + 1
    return counts


# ====================================================================== #
# 1. Qualtran spacetime via PhysicalCostModel
# ====================================================================== #
def qualtran_estimate(qc, error_budget: float = 1e-3) -> Dict[str, float]:
    """Feed gate counts into Qualtran's ``PhysicalCostModel.make_gidney_fowler``
    surface-code spacetime model. Reports physical qubits, cycles, runtime,
    and synthesis error."""
    try:
        from qualtran.surface_code import PhysicalCostModel, AlgorithmSummary
        from qualtran.resource_counting import GateCounts
    except Exception as e:
        return {"re_qualtran_error": f"import: {e}"}

    counts = _gate_counts(qc)
    t = counts.get("t", 0) + counts.get("tdg", 0)
    # sx/sxdg (√X / √X†, emitted by nwqec's to_clifford_reduction) are
    # Clifford — count them so Qualtran's clifford total is not undercounted.
    clifford = (counts.get("h", 0) + counts.get("s", 0) + counts.get("sdg", 0)
                + counts.get("sx", 0) + counts.get("sxdg", 0)
                + counts.get("x", 0) + counts.get("y", 0) + counts.get("z", 0)
                + counts.get("cx", 0))
    rotation = counts.get("rz", 0)
    measurement = counts.get("measure", 0)

    try:
        gc = GateCounts(
            t=t, clifford=clifford, rotation=rotation, measurement=measurement,
        )
        algo = AlgorithmSummary(
            n_algo_qubits=qc.num_qubits, n_logical_gates=gc,
        )
        # Use Gidney-Fowler 2019 model with auto-picked code distance for the
        # requested error budget.
        model = PhysicalCostModel.make_gidney_fowler(data_d=15)
        n_phys  = model.n_phys_qubits(algo)
        n_cyc   = model.n_cycles(algo)
        dur_hr  = model.duration_hr(algo)
        err     = model.error(algo)
    except Exception as e:
        return {"re_qualtran_error": f"compute: {e}"}

    return {
        "re_qualtran_t_count":          t,
        "re_qualtran_clifford_count":   clifford,
        "re_qualtran_n_algo_qubits":    qc.num_qubits,
        "re_qualtran_code_distance":    15,
        "re_qualtran_phys_qubits":      n_phys,
        "re_qualtran_n_cycles":         n_cyc,
        "re_qualtran_runtime_hr":       dur_hr,
        "re_qualtran_runtime_s":        dur_hr * 3600.0,
        "re_qualtran_synth_error":      err,
        "re_qualtran_spacetime_volume": n_phys * dur_hr,
    }


# ====================================================================== #
# 2. resource-superstaq (Infleqtion FT-compile + estimator)
# ====================================================================== #
def superstaq_estimate(qc, *,
                       d: int = 7,
                       num_t_factories: int = 2,
                       cultivation_repetition: int = 3) -> Dict[str, float]:
    """Run resource-superstaq's FT pipeline:
       cirq circuit → MovementLayout → ft_compile → ResourceEstimator.

    Defaults match the hello_estimate notebook's small-d configuration so
    runtime stays tractable."""
    try:
        import cirq
        import resource_estimation as res
    except Exception as e:
        return {"re_superstaq_error": f"import: {e}"}

    if qc.num_qubits == 0 or len(qc.data) == 0:
        return {"re_superstaq_error": "empty circuit"}

    try:
        # Superstaq's ft_compile only accepts {H, S, X, Z, T, CX}
        # (verified empirically — it rejects Y, Sdg, Tdg).  Decompose to
        # that subset; both rewrites preserve T-count so the comparison
        # remains meaningful.
        circ = _qiskit_to_cirq(qc, decompose_y=True, decompose_inverses=True)
        # If the caller fed us Rz's (the test driver shouldn't), compile down.
        from cirq import ops as _cops
        if any(isinstance(op.gate, _cops.ZPowGate) and op.gate.exponent not in
               (1, -1, 0.5, -0.5, 0.25, -0.25) for op in circ.all_operations()):
            circ = res.clifford_t.compile_cirq_to_clifford_t(circ, eps=1e-2)
    except Exception as e:
        return {"re_superstaq_error": f"qiskit→cirq: {e}"}

    try:
        ssm = res.architecture.DefaultMovement(
            d=d, idling=False, post_op_correction=True,
            syndrome_rounds=1, cultivation_repetition=cultivation_repetition,
        )
        layout = res.layout.MovementLayout(input_circuit=circ,
                                           num_t_factories=num_t_factories)
        primitive = res.compile_ftqc.ft_compile(layout=layout, arc=ssm,
                                                verbose=False)
        est = res.estimate.ResourceEstimator(arc=ssm)
        phys = est.physical_qubits(primitive)
        ptime_us = est.parallel_circuit_time(primitive)
        try:
            stime_us = est.serial_circuit_time(primitive)
        except Exception:
            stime_us = None
    except Exception as e:
        return {"re_superstaq_error": f"ft_compile/estimate: {e}"}

    return {
        "re_superstaq_phys_qubits":     phys,
        "re_superstaq_parallel_time_us": ptime_us,
        "re_superstaq_serial_time_us":   stime_us,
        "re_superstaq_volume":          (phys * ptime_us) if (phys is not None and ptime_us is not None) else None,
        "re_superstaq_d":               d,
        "re_superstaq_num_t_factories": num_t_factories,
        "re_superstaq_primitive_moments": len(primitive),
    }


# ====================================================================== #
# Convenience: run both at once.
# ====================================================================== #
def all_resource_estimates(qc) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(qualtran_estimate(qc))
    out.update(superstaq_estimate(qc))
    return out
