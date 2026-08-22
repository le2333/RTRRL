"""Deployment composition for RTRRL run as one graph over a trial's seeds.

The same algorithm, the same declarations and the same run documents as
``rtrrl``: what differs is that the worker hands this entry a whole group and
every member is computed on one device pass. A member's artifacts, metrics and
result are still its own.

It is a separate entry rather than a mode of the other one so that an image says
in its catalog whether it can do this. The control plane reads the catalog to
decide what it may ask for, and a capability that has to be inferred from a
version number is a capability someone will infer wrongly.
"""

from __future__ import annotations

import sys

from memorax.algorithms.rtrrl_aaai import METRICS as METRICS
from memorax.algorithms.rtrrl_aaai import PARAMETERS as PARAMETERS
from memorax.algorithms.rtrrl_aaai import RTRRL

from ._ensemble import main_for
from .rtrrl import build_request, runtime_config

main = main_for(
    RTRRL, build_request=build_request, runtime_config=runtime_config
)


if __name__ == "__main__":
    sys.exit(main())
