#!/usr/bin/env bash
set -uo pipefail

export DEBIAN_FRONTEND=noninteractive
export TASK12_RESULTS_DIR=/tmp/task12-results
export TASK12_FUNCTIONAL_HEAD_SHA=33448c2a12ef93edf4389b9286b1da60d8a8a17f
export TASK12_FEATURE_BASE_SHA=5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6
export TASK12_TASK10_BASE_SHA=5a89953b5d09909b35c5016118dc11a1adb0dec2
export TASK12_REPORT_PARENT_SHA=62246110d39256dba5641293920cebbb0b626a65
export TASK12_REVIEW_PATCH_SHA256=9452b8661b2de7ee2afb09ab80a30dc8f94ec5527021e43d46192f8e45770052
export NODE_OPTIONS=--max-old-space-size=4096

HEAD_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-review/head-33448c2.tar
BASE_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-review/base-5f7ff4e.tar
SOURCE_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-review/sources
RESULT_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-review/results

rm -rf /tmp/head /tmp/base /tmp/venv "$TASK12_RESULTS_DIR"
mkdir -p /tmp/head /tmp/base "$TASK12_RESULTS_DIR"
apt-get update -qq
apt-get install -y -qq git python3-venv build-essential time >/dev/null
aws s3 cp "$HEAD_URI" /tmp/head.tar >/dev/null
aws s3 cp "$BASE_URI" /tmp/base.tar >/dev/null
tar -xf /tmp/head.tar -C /tmp/head
tar -xf /tmp/base.tar -C /tmp/base
for source in \
  task12_numerical_evidence.py \
  task12_brax_smoke.py \
  task12_collect_evidence.py \
  task12_nodeids.py; do
  aws s3 cp "$SOURCE_URI/$source" \
    "/tmp/head/memo/tests/rtrrl_parity/$source" >/dev/null
done
aws s3 cp "$SOURCE_URI/experiment.py" \
  /tmp/head/memo/experiments/base/experiment.py >/dev/null
aws s3 cp "$SOURCE_URI/memorax_init.py" \
  /tmp/head/memo/memorax/__init__.py >/dev/null
aws s3 cp "$SOURCE_URI/algorithms_init.py" \
  /tmp/head/memo/memorax/algorithms/__init__.py >/dev/null
aws s3 cp "$SOURCE_URI/rtrrl_init.py" \
  /tmp/head/memo/memorax/algorithms/rtrrl/__init__.py >/dev/null
aws s3 cp "$SOURCE_URI/rtrrl_legacy.py" \
  /tmp/head/memo/memorax/algorithms/rtrrl/legacy.py >/dev/null
aws s3 cp "$SOURCE_URI/online_ac_golden.py" \
  /tmp/head/memo/tests/online_ac/golden.py >/dev/null
aws s3 cp "$SOURCE_URI/test_meta_parity.py" \
  /tmp/head/memo/tests/online_ac/test_meta_parity.py >/dev/null
python -m venv /tmp/venv
/tmp/venv/bin/pip install -q -e '/tmp/head/memo[brax]' \
  pytest ruff pyright jax==0.10.0 flax==0.12.7 \
  optax==0.2.8 distrax==0.1.8
ln -s /tmp/venv /tmp/head/memo/.venv
ln -s /tmp/venv /tmp/base/memo/.venv

/tmp/venv/bin/python - <<'PY' >"$TASK12_RESULTS_DIR/runtime.json"
import importlib.metadata as metadata
import json
import jax
import platform

print(json.dumps({
    "python": platform.python_version(),
    "jax": jax.__version__,
    "jaxlib": metadata.version("jaxlib"),
    "flax": metadata.version("flax"),
    "brax": metadata.version("brax"),
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
}, sort_keys=True))
PY

run_cmd() {
  local name="$1"
  local cwd="$2"
  local environment_json="$3"
  local command="$4"
  printf '%s\n' "$command" >"$TASK12_RESULTS_DIR/$name.command"
  printf '%s\n' "$cwd" >"$TASK12_RESULTS_DIR/$name.cwd"
  printf '%s\n' "$environment_json" >"$TASK12_RESULTS_DIR/$name.env"
  echo "TASK12_BEGIN $name"
  echo "TASK12_COMMAND $command"
  echo "TASK12_CWD $cwd"
  echo "TASK12_ENV $environment_json"
  (
    cd "$cwd" &&
      /usr/bin/time -v -o "$TASK12_RESULTS_DIR/$name.time" \
        bash -lc "$command"
  ) >"$TASK12_RESULTS_DIR/$name.log" 2>&1
  local status=$?
  printf '%s\n' "$status" >"$TASK12_RESULTS_DIR/$name.exit"
  echo "TASK12_STATUS $name $status"
  echo "TASK12_LOG_BEGIN $name"
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())' \
    "$TASK12_RESULTS_DIR/$name.log"
  echo "TASK12_LOG_END $name"
  python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())' \
    "$TASK12_RESULTS_DIR/$name.time"
  return 0
}

HEAD_PYTHONPATH=/tmp/head/rtrrl:/tmp/head/memo
BASE_PYTHONPATH=/tmp/base/rtrrl:/tmp/base/memo

run_cmd finite_differences /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo","RTRRL_RUN_ACCELERATED_NUMERICS":"1"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH RTRRL_RUN_ACCELERATED_NUMERICS=1 /tmp/venv/bin/pytest -q -s tests/rtrrl_parity/test_lru_credit_parity.py::test_two_step_credit_directional_finite_differences --junitxml=$TASK12_RESULTS_DIR/finite_differences.xml"

run_cmd strict_parity /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo","RTRRL_RUN_ACCELERATED_NUMERICS":"1"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH RTRRL_RUN_ACCELERATED_NUMERICS=1 /tmp/venv/bin/pytest -q tests/rtrrl_parity --junitxml=$TASK12_RESULTS_DIR/strict_parity.xml"

run_cmd selected_online_ac /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac -k 'rtrrl or meta or independent' --junitxml=$TASK12_RESULTS_DIR/selected_online_ac.xml"

run_cmd independent /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/test_independent_rtrrl.py --junitxml=$TASK12_RESULTS_DIR/independent.xml"

run_cmd online_ac_head /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python tests/rtrrl_parity/task12_nodeids.py /tmp/head/memo $TASK12_RESULTS_DIR/online_ac_head.nodes && PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac --junitxml=$TASK12_RESULTS_DIR/online_ac_head.xml"

run_cmd online_ac_base /tmp/base/memo \
  '{"PYTHONPATH":"/tmp/base/rtrrl:/tmp/base/memo"}' \
  "PYTHONPATH=$BASE_PYTHONPATH /tmp/venv/bin/python /tmp/head/memo/tests/rtrrl_parity/task12_nodeids.py /tmp/base/memo $TASK12_RESULTS_DIR/online_ac_base.nodes && PYTHONPATH=$BASE_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac --junitxml=$TASK12_RESULTS_DIR/online_ac_base.xml"

run_cmd numerical_harness /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python -m tests.rtrrl_parity.task12_numerical_evidence > $TASK12_RESULTS_DIR/numerical_harness.stdout.json"

run_cmd brax_smoke /tmp/head \
  '{"PYTHONPATH":"/tmp/head/rtrrl:/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python -m tests.rtrrl_parity.task12_brax_smoke > $TASK12_RESULTS_DIR/brax_smoke.stdout.json"

run_cmd ruff /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/ruff check memorax/algorithms/rtrrl memorax/algorithms/independent_rtrrl.py experiments/base/experiment.py experiments/rtrrl_hopper/run.py tests/rtrrl_parity tests/online_ac/test_meta_parity.py tests/online_ac/test_legacy_builders.py tests/test_independent_rtrrl.py ../rtrrl/rtrrl.py"

run_cmd pyright_head /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/pyright"

run_cmd pyright_base /tmp/base/memo \
  '{}' \
  "/tmp/venv/bin/pyright"

run_cmd pyright_review_head /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/pyright experiments/base/experiment.py memorax/__init__.py memorax/algorithms/__init__.py memorax/algorithms/rtrrl/__init__.py tests/test_independent_rtrrl.py"

run_cmd pyright_review_base /tmp/base/memo \
  '{}' \
  "/tmp/venv/bin/pyright experiments/base/experiment.py memorax/__init__.py memorax/algorithms/__init__.py tests/test_independent_rtrrl.py"

run_cmd compileall /tmp/head \
  '{}' \
  "/tmp/venv/bin/python -m compileall -q memo/memorax memo/experiments memo/tests/rtrrl_parity memo/tests/online_ac memo/tests/test_independent_rtrrl.py rtrrl/rtrrl.py"

/tmp/venv/bin/python - <<'PY' >"$TASK12_RESULTS_DIR/source_hashes.json"
from hashlib import sha256
import json
from pathlib import Path

root = Path("/tmp/head/memo/tests/rtrrl_parity")
paths = {
    name: root / name
    for name in (
        "task12_numerical_evidence.py",
        "task12_brax_smoke.py",
        "task12_collect_evidence.py",
        "task12_nodeids.py",
    )
}
paths["task12_batch_verification.sh"] = Path(
    "/tmp/task12_batch_verification.sh"
)
paths.update({
    "memo/experiments/base/experiment.py": Path(
        "/tmp/head/memo/experiments/base/experiment.py"
    ),
    "memo/memorax/__init__.py": Path("/tmp/head/memo/memorax/__init__.py"),
    "memo/memorax/algorithms/__init__.py": Path(
        "/tmp/head/memo/memorax/algorithms/__init__.py"
    ),
    "memo/memorax/algorithms/rtrrl/__init__.py": Path(
        "/tmp/head/memo/memorax/algorithms/rtrrl/__init__.py"
    ),
    "memo/memorax/algorithms/rtrrl/legacy.py": Path(
        "/tmp/head/memo/memorax/algorithms/rtrrl/legacy.py"
    ),
    "memo/tests/online_ac/golden.py": Path(
        "/tmp/head/memo/tests/online_ac/golden.py"
    ),
    "memo/tests/online_ac/test_meta_parity.py": Path(
        "/tmp/head/memo/tests/online_ac/test_meta_parity.py"
    ),
})
print(json.dumps(
    {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    sort_keys=True,
))
PY

overall=0
if ! PYTHONPATH="$HEAD_PYTHONPATH" /tmp/venv/bin/python \
  -m tests.rtrrl_parity.task12_collect_evidence \
  >"$TASK12_RESULTS_DIR/evidence.json"; then
  overall=1
fi
aws s3 cp --recursive "$TASK12_RESULTS_DIR/" \
  "$RESULT_URI/$AWS_BATCH_JOB_ID/" >/dev/null
echo "TASK12_EVIDENCE_URI $RESULT_URI/$AWS_BATCH_JOB_ID/evidence.json"

for required in finite_differences strict_parity independent \
  numerical_harness brax_smoke ruff compileall; do
  if [[ "$(<"$TASK12_RESULTS_DIR/$required.exit")" != "0" ]]; then
    overall=1
  fi
done
echo "TASK12_OVERALL $overall"
exit "$overall"
