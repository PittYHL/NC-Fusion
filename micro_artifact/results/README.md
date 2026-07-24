# Results

`reference/` contains values transcribed from the paper for comparison.
Generated experiment outputs belong under `runs/` and are ignored by Git.

Experiment writers merge CSV rows by configuration key. Existing rows are
preserved, matching benchmark/configuration rows are replaced on rerun, and
new rows are appended. Aggregate, relative, and summary CSVs are regenerated
from the merged raw records so repeated runs do not duplicate averages.
