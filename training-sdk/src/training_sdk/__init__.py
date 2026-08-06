"""What the control plane and the worker both have to know.

Three things, and shrinking: the wire contract, the object store both sides read
and write through, and the episode types the metrics are shaped as. The
reporter, its sinks and the worker itself used to live here too; they are the
worker's own business and they moved there, because a shared package is paid for
by both sides and the control plane has no use for an Aim client.

Nothing is re-exported here. Import the module you need — `training_sdk.contract`,
`training_sdk.objects`, `training_sdk.episode` — so that a caller pays for only
what it uses and the import graph stays readable.
"""
