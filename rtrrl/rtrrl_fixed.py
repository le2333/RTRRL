"""RTRRL entrypoint with action/log-prob and trace-order fixes enabled."""

from __future__ import annotations

import simple_parsing

from logging_util import with_logger
from rtrrl import RTRRLParams, train_rtrrl


if __name__ == "__main__":
    hparams: RTRRLParams = simple_parsing.parse(RTRRLParams, add_config_path_arg=True)
    hparams.align_action_logprob = True
    hparams.update_trace_before_td = True

    with_logger(
        train_rtrrl,
        hparams,
        logger_name=hparams.logging,
        project_name="RTRRL",
        run_name=hparams.env_params.env_name,
        hparams_type=RTRRLParams,
    )
