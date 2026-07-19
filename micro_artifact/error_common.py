"""Shared circuit construction for the Section 5.7 evaluations."""

from __future__ import annotations

from typing import Any


SUPPORTED_METHODS = ("gridsyn", "ncf-one")


def validate_methods(methods: list[str] | None) -> tuple[str, ...]:
    selected = tuple(methods or SUPPORTED_METHODS)
    unknown = [method for method in selected if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            "error evaluations support only gridsyn and ncf-one; "
            f"received {', '.join(unknown)}"
        )
    return tuple(dict.fromkeys(selected))


def compile_method(
    hamiltonian: Any,
    method: str,
    *,
    synthesize: bool,
    error_threshold: float,
    t_budget: int,
    gpu: int,
    trotter_steps: int,
    evolution_time: float,
    window: int | None,
    pauli_order_seed: int | None,
) -> tuple[Any, Any]:
    """Return ``(rz_qc, clifford_t_qc)`` for one method.

    The first circuit is always the unsynthesized rotation circuit. The second
    is the synthesized Clifford+T circuit when ``synthesize=True``.
    """

    if method == "gridsyn":
        from baseline import baseline_circuit

        return baseline_circuit(
            hamiltonian,
            1,
            error_threshold=error_threshold,
            gpu=gpu,
            Trotter_steps=trotter_steps,
            evolution_time=evolution_time,
            rustiq=False,
            GRIDSYNTH=True,
            t_budget=t_budget,
            synthesize=synthesize,
            benchmark=None,
            method="grid",
        )

    if method == "ncf-one":
        from ncfusion import NC_Fusion

        return NC_Fusion(
            hamiltonian,
            budget=1,
            window=window,
            error_threshold=error_threshold,
            t_budget=t_budget,
            gpu=gpu,
            trotter_steps=trotter_steps,
            evolution_time=evolution_time,
            synthesize=synthesize,
            pauli_order_seed=pauli_order_seed,
        )

    raise ValueError(f"unsupported error-evaluation method: {method}")
