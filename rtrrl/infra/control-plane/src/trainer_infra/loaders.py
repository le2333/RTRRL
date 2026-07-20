from pathlib import Path
from typing import Any

import yaml

from trainer_infra.models import ExperimentSpec, ScriptCatalog


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_experiment(path: Path) -> ExperimentSpec:
    return ExperimentSpec.model_validate(_load_yaml(path))


def load_script_catalog(path: Path) -> ScriptCatalog:
    return ScriptCatalog.model_validate(_load_yaml(path))
