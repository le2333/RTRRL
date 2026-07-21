#!/usr/bin/env bash
set -uo pipefail

export DEBIAN_FRONTEND=noninteractive
export TASK12_RESULTS_DIR=/tmp/task12-results
export TASK12_FUNCTIONAL_HEAD_SHA=b50100dc66305e4005bed93f3d1750df8b474862
export TASK12_FEATURE_BASE_SHA=5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6
export TASK12_TASK10_BASE_SHA=5a89953b5d09909b35c5016118dc11a1adb0dec2
export TASK12_REPORT_PARENT_SHA=b50100dc66305e4005bed93f3d1750df8b474862
export TASK12_REVIEW_PATCH_SHA256=3ece46030ffd747a13d884273bac5b62b0e39c0c6b61f42c6697fc776625fdda
export NODE_OPTIONS=--max-old-space-size=4096

HEAD_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-preserved/head-b50100d-final-gates-overlay.tar
BASE_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-review/base-5f7ff4e.tar
ORACLE_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-preserved/aaai25-4301943.tar
RESULT_URI=s3://rtrrl-artifacts-007122174918/oracle/task12-preserved/results

rm -rf /tmp/head /tmp/base /tmp/oracle /tmp/venv /tmp/preserved-venv \
  /tmp/oracle-venv "$TASK12_RESULTS_DIR"
mkdir -p /tmp/head /tmp/base /tmp/oracle "$TASK12_RESULTS_DIR"
apt-get update -qq
apt-get install -y -qq git python3-venv build-essential time >/dev/null
aws s3 cp "$HEAD_URI" /tmp/head.tar >/dev/null
aws s3 cp "$BASE_URI" /tmp/base.tar >/dev/null
aws s3 cp "$ORACLE_URI" /tmp/oracle.tar >/dev/null
tar -xf /tmp/head.tar -C /tmp/head
tar -xf /tmp/base.tar -C /tmp/base
tar -xf /tmp/oracle.tar -C /tmp/oracle
python -m venv /tmp/venv
/tmp/venv/bin/pip install -q -e '/tmp/head/memo[brax]' \
  pytest ruff pyright jax==0.10.0 flax==0.12.7 \
  optax==0.2.8 distrax==0.1.8
ln -s /tmp/venv /tmp/head/memo/.venv
ln -s /tmp/venv /tmp/base/memo/.venv
python -m venv /tmp/preserved-venv
python -m venv /tmp/oracle-venv
/tmp/preserved-venv/bin/pip install -q \
  jax==0.5.0 jaxlib==0.5.0 flax==0.10.2 \
  distrax==0.1.5 optax==0.2.4 chex==0.1.88
/tmp/oracle-venv/bin/pip install -q \
  jax==0.4.38 jaxlib==0.4.38 flax==0.10.2 \
  distrax==0.1.5 optax==0.2.4 chex==0.1.88

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

HEAD_PYTHONPATH=/tmp/head/memo
BASE_PYTHONPATH=/tmp/base/memo

run_cmd finite_differences /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo","RTRRL_RUN_ACCELERATED_NUMERICS":"1"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH RTRRL_RUN_ACCELERATED_NUMERICS=1 /tmp/venv/bin/pytest -q -s tests/rtrrl_parity/test_lru_credit_parity.py::test_two_step_credit_directional_finite_differences --junitxml=$TASK12_RESULTS_DIR/finite_differences.xml"

run_cmd strict_parity /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo","RTRRL_AAAI25_ROOT":"/tmp/oracle","RTRRL_RUN_ACCELERATED_NUMERICS":"1"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH RTRRL_AAAI25_ROOT=/tmp/oracle RTRRL_RUN_ACCELERATED_NUMERICS=1 /tmp/venv/bin/pytest -q tests/rtrrl_parity --junitxml=$TASK12_RESULTS_DIR/strict_parity.xml"

run_cmd selected_online_ac /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac -k 'rtrrl or meta or independent' --junitxml=$TASK12_RESULTS_DIR/selected_online_ac.xml"

run_cmd independent /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/test_independent_rtrrl.py --junitxml=$TASK12_RESULTS_DIR/independent.xml"

run_cmd online_ac_head /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python tests/rtrrl_parity/task12_nodeids.py /tmp/head/memo $TASK12_RESULTS_DIR/online_ac_head.nodes && PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac --junitxml=$TASK12_RESULTS_DIR/online_ac_head.xml"

run_cmd online_ac_base /tmp/base/memo \
  '{"PYTHONPATH":"/tmp/base/memo"}' \
  "PYTHONPATH=$BASE_PYTHONPATH /tmp/venv/bin/python /tmp/head/memo/tests/rtrrl_parity/task12_nodeids.py /tmp/base/memo $TASK12_RESULTS_DIR/online_ac_base.nodes && PYTHONPATH=$BASE_PYTHONPATH /tmp/venv/bin/pytest -q tests/online_ac --junitxml=$TASK12_RESULTS_DIR/online_ac_base.xml"

run_cmd numerical_harness /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python -m tests.rtrrl_parity.task12_numerical_evidence > $TASK12_RESULTS_DIR/numerical_harness.stdout.json"

run_cmd preserved_probe /tmp/head \
  '{"PYTHONPATH":"/tmp/head/rtrrl","JAX_VERSION":"0.5.0"}' \
  "PYTHONPATH=/tmp/head/rtrrl /tmp/preserved-venv/bin/python memo/tests/rtrrl_parity/preserved_original_probe.py --source-root /tmp/head/rtrrl > $TASK12_RESULTS_DIR/preserved_probe.stdout.json"

run_cmd oracle_probe /tmp/head \
  '{"PYTHONPATH":"/tmp/oracle","JAX_VERSION":"0.4.38"}' \
  "PYTHONPATH=/tmp/oracle /tmp/oracle-venv/bin/python memo/tests/rtrrl_parity/preserved_original_probe.py --source-root /tmp/oracle > $TASK12_RESULTS_DIR/oracle_probe.stdout.json"

run_cmd preserved_compare /tmp/head \
  '{}' \
  "/tmp/venv/bin/python memo/tests/rtrrl_parity/preserved_original_compare.py --preserved $TASK12_RESULTS_DIR/preserved_probe.stdout.json --oracle $TASK12_RESULTS_DIR/oracle_probe.stdout.json > $TASK12_RESULTS_DIR/preserved_compare.stdout.json"

run_cmd source_audit /tmp/head \
  '{}' \
  "/tmp/venv/bin/python memo/tests/rtrrl_parity/preserved_original_source_audit.py --preserved-root /tmp/head/rtrrl --oracle-root /tmp/oracle > $TASK12_RESULTS_DIR/source_audit.stdout.json"

run_cmd brax_smoke /tmp/head/memo \
  '{"PYTHONPATH":"/tmp/head/memo"}' \
  "PYTHONPATH=$HEAD_PYTHONPATH /tmp/venv/bin/python -m tests.rtrrl_parity.task12_brax_smoke > $TASK12_RESULTS_DIR/brax_smoke.stdout.json"

run_cmd ruff /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/ruff check memorax/algorithms/rtrrl memorax/algorithms/independent_rtrrl.py experiments/base/experiment.py experiments/rtrrl_hopper/run.py tests/rtrrl_parity tests/online_ac/test_meta_parity.py tests/online_ac/test_legacy_builders.py tests/test_independent_rtrrl.py"

run_cmd pyright_head /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/pyright --outputjson"

run_cmd pyright_base /tmp/base/memo \
  '{}' \
  "/tmp/venv/bin/pyright --outputjson"

run_cmd pyright_review_head /tmp/head/memo \
  '{}' \
  "/tmp/venv/bin/pyright --outputjson experiments/base/experiment.py memorax/__init__.py memorax/algorithms/__init__.py memorax/algorithms/rtrrl/__init__.py tests/test_independent_rtrrl.py"

run_cmd pyright_review_base /tmp/base/memo \
  '{}' \
  "/tmp/venv/bin/pyright --outputjson experiments/base/experiment.py memorax/__init__.py memorax/algorithms/__init__.py memorax/algorithms/rtrrl.py tests/test_independent_rtrrl.py"

run_cmd compileall /tmp/head \
  '{}' \
  "/tmp/venv/bin/python -m compileall -q memo/memorax memo/experiments memo/tests/rtrrl_parity memo/tests/online_ac memo/tests/test_independent_rtrrl.py"

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
        "preserved_original_probe.py",
        "preserved_original_compare.py",
        "preserved_original_source_audit.py",
        "test_task12_evidence_gates.py",
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
  numerical_harness preserved_probe oracle_probe preserved_compare source_audit \
  brax_smoke ruff compileall; do
  if [[ "$(<"$TASK12_RESULTS_DIR/$required.exit")" != "0" ]]; then
    overall=1
  fi
done
echo "TASK12_OVERALL $overall"
exit "$overall"
