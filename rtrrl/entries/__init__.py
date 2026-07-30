"""What this fork can be asked to run, and under whose vocabulary.

Everything else in this project is the authors' code as forked, and it is meant
to stay that way: it is the arm of the comparison whose whole value is that
nobody here has touched its arithmetic. This package is the one place that is
ours. It holds the translation from the parameter names the control plane
samples -- the same names the four `memo` arms are configured through -- onto
the `RTRRLParams` the authors' training function reads, and the logger that
turns what their loop prints into what the control plane scores.

A module in here is an entry when it declares `SPACE`, `METRICS` and `main`,
which is the same contract `memo/runner/catalog.py` discovers over. Nothing is
registered anywhere; `scripts/build_catalog.py` finds them by looking.
"""
