# NC-Fusion artifact

See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete artifact-evaluation
instructions, including setup, benchmark selection, existing-result reuse,
explicit regeneration, CSV checkpoint/merge behavior, and commands for every
experiment.

Result CSVs use configuration-key merge semantics: rerunning an existing
configuration replaces that row, while new benchmarks or parameter settings
are appended and earlier selections are retained. Summary/relative CSVs are
recomputed from the retained raw rows.

Benchmark and method selection

Producer experiments accept repeated `--benchmark` options, so a run can be
limited to one benchmark or a small subset. For example:

```bash
PYTHONPATH=src:legacy:. python -m micro_artifact.two_qubit_result \
  --source existing \
  --benchmark LiH \
  --benchmark IS-2D-30
```

`single_qubit_result` also accepts `--method`. Use `--method rustiq` or
`--method phoenix` to refresh only that comparison, or `--method ncf-one` for
NC-Fusion versus GridSynth. Omitting `--method` runs all comparisons. Partial
runs preserve previously recorded benchmark and comparison fields.
Spacetime-volume analysis also accepts repeated `--benchmark` options and
supports the short Ising/Heisenberg aliases; omitted options use all 13
benchmarks.
Window-size sensitivity accepts repeated `--window-size` options such as
`--window-size full --window-size 64`; omitted options run the complete sweep.
Application-level fidelity accepts repeated `--trotter-steps` options to run
only selected Trotter steps; omitted options use 1, 5, 10, and 20.

Artifact experiments read existing inputs or results by default. Use their
`--source generate` option to regenerate selected results; producer and error-evaluation
commands save generated circuits when applicable. See [EXPERIMENTS.md](EXPERIMENTS.md)
for all commands and accepted benchmark aliases.

Analytical T-count estimation runs its fixed nine-precision sweep
automatically; it does not require an epsilon command-line option.

The T-count optimizer comparison includes PyZX in the paper dependencies. If
T-Zap or T-Optimizer is not available, install the missing tools with
`PYTHONPATH=src:legacy:. python -m micro_artifact.install_optimizers`, or pass
`--install-missing` together with `--source generate` to the comparison
command. Existing-result runs never download or modify optimizer
dependencies.
