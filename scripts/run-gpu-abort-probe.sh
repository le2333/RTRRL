#!/usr/bin/env bash
# Submit the GPU abort probe to Batch and print what it said.
#
# This is `.github/workflows/gpu-abort-probe.yml` without the OIDC step, for a
# host whose own credentials already reach Batch. The CI role cannot run it:
# `rtrrl-github-actions-role` was created to push images to ECR, and submitting
# a job needs batch:DescribeJobDefinitions, batch:RegisterJobDefinition,
# batch:SubmitJob, batch:DescribeJobs, logs:GetLogEvents and iam:PassRole --
# none of which it has. Both workflow runs died on the first of those.
#
# Run from the repository root:
#
#     scripts/run-gpu-abort-probe.sh <image-digest>
#
# where the digest is the gpu variant of the memo trainer, as
# `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:...`. The build
# that produces it prints it in its job summary.
#
# The probe refuses to pass on the CPU, so a job that lands without a device
# says so instead of quietly compiling the graph on the host and exiting zero.
# Set PROBE_ALLOW_CPU=1 only if compiling on the CPU is what you wanted.
#
# Bisecting: PROBE_BATCH, PROBE_TRUNCATION and PROBE_CORE move one axis each.
# Batch size and truncation are the shape of the differentiated window, which is
# both the arithmetic a GPU is here for and the arithmetic an emitter has to
# fuse, so those are the two worth walking.

set -euo pipefail

readonly ACCOUNT=007122174918
readonly REGION=eu-north-1
readonly QUEUE=dev-gpu-queue
readonly LOG_GROUP=/trainer/jobs
readonly PROBE=memo/docs/gpu-abort-probe.py

if [ $# -ne 1 ]; then
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2
  exit 2
fi
readonly IMAGE="$1"

case "$IMAGE" in
  *@sha256:*) ;;
  *)
    echo "the image must be pinned to a digest: name@sha256:..." >&2
    exit 2
    ;;
esac

test -f "$PROBE" || {
  echo "run this from the repository root: $PROBE is not here" >&2
  exit 2
}

export AWS_DEFAULT_REGION="$REGION"
test "$(aws sts get-caller-identity --query Account --output text)" = "$ACCOUNT"

NAME="gpu-abort-probe-$(date -u +%Y%m%d-%H%M%S)"

# The g6x definition already carries the queue's instance type, its GPU resource
# requirement and the job role that reaches S3 and CloudWatch. Cloning it and
# replacing only the image and the command keeps the probe on exactly the
# machine an experiment would get.
BASE="$(aws batch describe-job-definitions --status ACTIVE --output json \
  | jq -c '[.jobDefinitions[] | select(.jobDefinitionName | startswith("trainer-g6x-"))] | last')"
test "$BASE" != null || {
  echo "no ACTIVE trainer-g6x-* job definition to clone" >&2
  exit 1
}

# The probe arrives as an environment variable rather than in the image, so the
# image under test stays the one an experiment would run. gzip because ECS caps
# a container override at 8192 characters and the plain base64 is near it.
WRAPPER='import base64,os,subprocess,sys,zlib; rc=subprocess.run([sys.executable,"-c",zlib.decompress(base64.b64decode(os.environ["PROBE_SOURCE"]),31)]).returncode; raise SystemExit(rc)'

DEFINITION="$(printf '%s' "$BASE" | jq -c \
  --arg name "$NAME" \
  --arg image "$IMAGE" \
  --arg wrapper "$WRAPPER" \
  'del(.jobDefinitionArn,.revision,.status,.tags)
   | .jobDefinitionName=$name
   | .containerProperties.image=$image
   | .containerProperties.command=["python","-c",$wrapper]')"
DEFINITION_ARN="$(aws batch register-job-definition --cli-input-json "$DEFINITION" \
  --query jobDefinitionArn --output text)"
echo "registered $DEFINITION_ARN"

SOURCE="$(gzip -9 -c "$PROBE" | base64 -w 0)"
echo "probe override is ${#SOURCE} characters of the 8192 ECS allows"
test "${#SOURCE}" -lt 7000

OVERRIDES="$(jq -cn \
  --arg source "$SOURCE" \
  --arg batch "${PROBE_BATCH:-8}" \
  --arg truncation "${PROBE_TRUNCATION:-10}" \
  --arg core "${PROBE_CORE:-lru}" \
  --arg envs "${PROBE_ENVS:-1}" \
  --arg steps "${PROBE_STEPS:-512}" \
  '{environment:[
    {name:"PROBE_SOURCE",value:$source},
    {name:"PROBE_BATCH",value:$batch},
    {name:"PROBE_TRUNCATION",value:$truncation},
    {name:"PROBE_CORE",value:$core},
    {name:"PROBE_ENVS",value:$envs},
    {name:"PROBE_STEPS",value:$steps},
    {name:"XLA_FLAGS",value:"--xla_dump_to=/tmp/hlo"}
  ]}')"

JOB_ID="$(aws batch submit-job --job-name "$NAME" \
  --job-queue "$QUEUE" --job-definition "$DEFINITION_ARN" \
  --timeout attemptDurationSeconds=1200 \
  --container-overrides "$OVERRIDES" \
  --query jobId --output text)"
echo "submitted $JOB_ID to $QUEUE"

while :; do
  JOB="$(aws batch describe-jobs --jobs "$JOB_ID" --output json)"
  STATUS="$(printf '%s' "$JOB" | jq -r '.jobs[0].status')"
  echo "  $(date -u +%H:%M:%S) $STATUS"
  case "$STATUS" in
    SUCCEEDED | FAILED) break ;;
  esac
  sleep 15
done

# The exit reason names a native abort where the log cannot: a process killed by
# SIGABRT leaves "Essential container in task exited" here and nothing after its
# last flushed line there.
printf '%s' "$JOB" | jq -r '.jobs[0] | "reason: \(.statusReason // "none")",
  "container exit: \(.container.exitCode // "none") \(.container.reason // "")"'

STREAM="$(printf '%s' "$JOB" | jq -r '.jobs[0].container.logStreamName')"
echo "--- $LOG_GROUP $STREAM"
aws logs get-log-events --log-group-name "$LOG_GROUP" --log-stream-name "$STREAM" \
  --start-from-head --query 'events[].message' --output text || true

# Registered per run so a failed probe cannot be re-run by accident against a
# definition someone has since edited.
aws batch deregister-job-definition --job-definition "$DEFINITION_ARN" >/dev/null || true

test "$STATUS" = SUCCEEDED
