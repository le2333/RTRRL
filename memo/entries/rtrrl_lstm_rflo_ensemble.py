"""RTRRL-LSTM-RFLO run as one graph over a trial's seeds.

The same algorithm, the same declarations and the same run documents as
``rtrrl_lstm_rflo``: what differs is that the worker hands this entry a whole
group and every member is computed on one device pass. A member's artifacts,
metrics and result are still its own.

It is a separate entry rather than a mode of the other one so that an image says
in its catalog whether it can do this. The control plane reads the catalog to
decide what it may ask for, and a capability that has to be inferred from a
version number is a capability someone will infer wrongly.

Everything the LSTM's shape depends on -- the hidden width, the bias the forget
gate is drawn with, whether a normalization follows the cell, which
differentiation credits it -- is declared static, so a group varying one is
refused before anything compiles. What may vary across members is what the graph
reads arithmetically: the discounts, the three trace decays, the two etas, the
entropy rate, and every optimizer's own numbers.
"""

from __future__ import annotations

import sys

from memorax.algorithms.rtrrl_lstm_rflo import METRICS as METRICS
from memorax.algorithms.rtrrl_lstm_rflo import PARAMETERS as PARAMETERS
from memorax.algorithms.rtrrl_lstm_rflo import RTRRLLstmRflo

from ._ensemble import GROUPED as GROUPED
from ._ensemble import main_for
from .rtrrl_lstm_rflo import build_request, runtime_config

main = main_for(
    RTRRLLstmRflo,
    build_request=build_request,
    runtime_config=runtime_config,
    declared=PARAMETERS,
)


if __name__ == "__main__":
    sys.exit(main())
