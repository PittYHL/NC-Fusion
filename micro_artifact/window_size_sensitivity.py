"""Window-size sensitivity evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import run_cli, run_configured


STATUS = "available"


def run(
    output: Path | str = "micro_artifact/results/runs/window_size_sensitivity",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
) -> dict[str, Any]:
    """Run the configured Section 5.5 window-size sweep."""

    return run_configured(
        "sensitivity",
        output,
        benchmarks=benchmarks,
        methods=methods,
        seed=seed,
        gpu=gpu,
        reuse_existing=False,
        save_qasm=False,
    )


if __name__ == "__main__":
    run_cli("window_size_sensitivity", run)
