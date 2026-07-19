"""Pauli-string order sensitivity evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import run_cli, run_configured


STATUS = "available"


def run(
    output: Path | str = "results/runs/pauli_string_order_sensitivity",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
) -> dict[str, Any]:
    """Run the configured Section 5.6 randomized Pauli-order sweep."""

    return run_configured(
        "random-order",
        output,
        benchmarks=benchmarks,
        methods=methods,
        seed=seed,
        gpu=gpu,
    )


if __name__ == "__main__":
    run_cli("pauli_string_order_sensitivity", run)
