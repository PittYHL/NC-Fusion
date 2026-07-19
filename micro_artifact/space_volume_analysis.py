"""Spacetime-volume evaluation entry point.

The paper configuration records the benchmark set and magic-state factory
counts.  The current runner still emits circuit metrics only; its
fault-tolerant spacetime-volume model has not been connected yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import run_cli, run_configured


STATUS = "partial"


def run(
    output: Path | str = "results/runs/space_volume_analysis",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
) -> dict[str, Any]:
    """Run the configured precursor for Section 5.10.

    Until the volume model is uploaded, this produces the underlying circuit
    metrics and manifest, not the paper's final volume values.
    """

    return run_configured(
        "spacetime-volume",
        output,
        benchmarks=benchmarks,
        methods=methods,
        seed=seed,
        gpu=gpu,
    )


if __name__ == "__main__":
    run_cli("space_volume_analysis", run)
