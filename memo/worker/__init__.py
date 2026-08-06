"""Run the configurations a manifest names, and report what they were worth.

The worker's whole job, and its whole interface to the control plane: it is
handed a manifest of run configurations and it hands back a score per run. What
it reports on the way -- the metrics file, the Aim run, the Rerun recording --
is its own business, which is why none of it is in the contract.

Nothing here is imported by the control plane. It starts a process; it does not
hold this package.
"""
