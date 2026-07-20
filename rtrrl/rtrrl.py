"""Historical RTRRL entry point backed exclusively by Memorax."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MEMO_ROOT = _REPOSITORY_ROOT / "memo"
if str(_MEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMO_ROOT))

from memorax.algorithms.rtrrl.compatibility import (  # noqa: E402
    LegacyRTRRLConfig,
)
from memorax.algorithms.rtrrl.entrypoint import (  # noqa: E402
    audit_repository_configs,
    describe_legacy_build,
    emit_json,
    load_legacy_mapping,
    normalize_legacy_invocation,
    parse_compatibility_cli,
    run_mock_epoch,
)


# Import compatibility for callers that historically used these names.
RTRRLParams = LegacyRTRRLConfig


def train_rtrrl(args: Any, logger=None):
    """Delegate the historical callable API to the Memorax experiment runner."""

    experiments = _MEMO_ROOT / "experiments"
    base = experiments / "base"
    for path in (experiments, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner = import_module("rtrrl_hopper.run")
    return runner.train_legacy(args, logger)


def main(argv: Sequence[str] | None = None) -> int:
    options, overrides = parse_compatibility_cli(argv)
    if options.compat_action == "mock-epoch":
        emit_json(run_mock_epoch())
        return 0
    if options.compat_action == "audit":
        emit_json(audit_repository_configs(_REPOSITORY_ROOT))
        return 0

    raw = load_legacy_mapping(options.config_path, overrides)
    config = normalize_legacy_invocation(raw)
    if options.compat_action == "build":
        emit_json(describe_legacy_build(config))
        return 0

    experiments = _MEMO_ROOT / "experiments"
    base = experiments / "base"
    for path in (experiments, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runner = import_module("rtrrl_hopper.run")
    runner.run_legacy_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
