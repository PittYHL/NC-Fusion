# NC-Fusion artifact

See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete artifact-evaluation
instructions, including setup, benchmark selection, QASM reuse, regeneration
rules, CSV checkpoint/merge behavior, and commands for every experiment.

Result CSVs use configuration-key merge semantics: rerunning an existing
configuration replaces that row, while new benchmarks or parameter settings
are appended and earlier selections are retained. Summary/relative CSVs are
recomputed from the retained raw rows.
