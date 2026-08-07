"""What the control plane and the worker both have to know.

Two things now: the wire contract, and the object store both sides read and
write through. The reporter, its sinks, the worker itself and the episode types
used to live here too; each of them belongs to one side, and a shared package is
paid for by both, so each went to the side that uses it.

What is left is what genuinely crosses the boundary. `objects` is there because
both sides talk to the same S3; `contract` because both sides have to agree on
what a run configuration is. Nothing may join them for any other reason.

Nothing is re-exported here. Import the module you need — `training_sdk.contract`,
`training_sdk.objects` — so that a caller pays for only what it uses and the
import graph stays readable.
"""
