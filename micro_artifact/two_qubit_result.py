"""Two-qubit NC-Fusion result evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import run_cli, run_configured


STATUS = "available"


def run(
    output: Path | str = "results/runs/two_qubit_result",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
) -> dict[str, Any]:
    """Run Table 4 using the two-qubit NC-Fusion method."""

    return run_configured(
        "table4",
        output,
        benchmarks=benchmarks,
        methods=methods or ["ncf-two"],
        seed=seed,
        gpu=gpu,
    )


if __name__ == "__main__":
    run_cli("two_qubit_result", run)
