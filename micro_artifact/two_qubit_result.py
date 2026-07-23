"""Two-qubit NC-Fusion result evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ncfusion.metrics import write_json, write_records_csv
from ncfusion.spec import find_benchmark, find_experiment

from .common import run_cli, run_configured
from .data import reusable_record


STATUS = "available"


def run(
    output: Path | str = "micro_artifact/results/runs/two_qubit_result",
    *,
    benchmarks: list[str] | None = None,
    methods: list[str] | None = None,
    seed: int = 0,
    gpu: int = 0,
    source: str = "existing",
) -> dict[str, Any]:
    """Load or generate the two-qubit NC-Fusion producer dataset."""

    if source not in {"existing", "generate"}:
        raise ValueError("source must be existing or generate")
    selected = list(find_experiment("table4").benchmarks if benchmarks is None else benchmarks)
    selected = [name for name in selected if find_benchmark(name).two_qubit_supported]
    if source == "generate":
        result = run_configured(
            "table4",
            output,
            benchmarks=selected,
            methods=["ncf-two"],
            seed=seed,
            gpu=gpu,
            reuse_existing=False,
            save_qasm=True,
        )
        result["manifest"]["evaluation"] = "two_qubit_result"
        result["manifest"]["source"] = "generate"
        result["manifest"]["producer_dataset"] = "two_qubit_result"
        return result

    records: list[dict[str, Any]] = []
    for benchmark_name in selected:
        record = reusable_record(find_benchmark(benchmark_name), "ncf-two")
        if record is None:
            raise RuntimeError(
                f"No stored two-qubit QASM for {benchmark_name}; rerun with --source generate."
            )
        records.append(record)
    output_path = Path(output)
    manifest = {
        "artifact_version": "0.1.0",
        "evaluation": "two_qubit_result",
        "source": "existing",
        "benchmarks": selected,
        "methods": ["ncf-two"],
        "record_count": len(records),
        "producer_dataset": "two_qubit_result",
    }
    write_json(output_path / "manifest.json", manifest)
    write_records_csv(output_path / "metrics.csv", records)
    return {"manifest": manifest, "records": records}


if __name__ == "__main__":
    run_cli("two_qubit_result", run, include_source=True)
