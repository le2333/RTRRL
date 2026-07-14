#!/usr/bin/env bash
# Benchmark one training workload across several EC2 instance types on AWS Batch.
#
# For each instance type it spins up a throwaway compute environment + queue
# pinned to that type (using the WHOLE instance: vCPUs and memory are read from
# the EC2 API), runs the SAME config, and measures the end-to-end job runtime
# (container start -> stop = JIT compile + train + eval). It then queries the
# current Spot price to rank speed and cost. Runs log normally (tagged
# bench-<type>); pass --logging none for pure compute.
#
# Usage:
#   infra/benchmark.sh --config config/ppo_hopper_xxx.yml \
#     [--types "c7a.xlarge c6a.xlarge c7i.xlarge c6i.xlarge c5a.xlarge"] \
#     [--steps 500000] [--keep] [--entry ppo_baseline.py]
#
#   --config PATH   config to run (compute should be identical across types)
#   --types "..."   space-separated full instance types (default: 5 x .xlarge).
#                   Include sizes for a matrix, e.g. "c7a.large c7a.xlarge c7a.2xlarge".
#   --steps N       override num_timesteps (keeps the benchmark short/identical)
#   --entry FILE    ppo_baseline.py (default) or rtrrl.py
#   --logging MODE  aim | wandb | aim+wandb | none (default: $LOGGING)
#   --keep          do NOT tear down the compute envs/queues afterwards
#
# Resources are named rtrrl-bench-<type>; they cost nothing while idle
# (minvCpus=0) and are deleted at the end unless --keep is given.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_DIR="${PROJECT_DIR:-$PWD}"
source "${HERE}/env.sh"

CONFIG=""; STEPS=""; ENTRY="${DEFAULT_ENTRY:-ppo_baseline.py}"; KEEP=0; MODE="${LOGGING}"
TYPES="c7a.xlarge c6a.xlarge c7i.xlarge c6i.xlarge c5a.xlarge"
MEM_HEADROOM_MIB=768   # leave room for ECS agent/OS; small enough for 2 GiB nodes
TIMEOUT_MIN=18   # max wait per job (instance launch + run) before marking TIMEOUT

while [ $# -gt 0 ]; do
  case "$1" in
    --config)      CONFIG="$2"; shift 2 ;;
    --types)       TYPES="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --entry)       ENTRY="$2"; shift 2 ;;
    --logging)     MODE="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --keep)        KEEP=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[ -z "${CONFIG}" ] && { echo "ERROR: --config is required" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "ERROR: config not found: ${CONFIG}" >&2; exit 1; }

CONFIG_B64=$(base64 -w0 "${CONFIG}")
SUBNET_JSON=$(printf '"%s",' ${SUBNET_IDS//,/ }); SUBNET_JSON="[${SUBNET_JSON%,}]"
INSTANCE_PROFILE_ARN="arn:aws:iam::${ACCOUNT_ID}:instance-profile/${ECS_INSTANCE_ROLE}"

san() { echo "${1//./-}"; }

declare -A VCPU MEMMB CE JQ JOB STAT DUR PRICE

echo "== reading instance specs (vCPU / memory) =="
for t in ${TYPES}; do
  read -r v m < <(aws ec2 describe-instance-types --instance-types "$t" --region "${REGION}" \
    --query "InstanceTypes[0].[VCpuInfo.DefaultVCpus,MemoryInfo.SizeInMiB]" --output text 2>/dev/null)
  if [ -z "${v:-}" ] || [ "${v}" = "None" ]; then
    echo "  ! $t: cannot read specs (unsupported in ${REGION}?), skipping"; continue
  fi
  # Keep the request comfortably below ECS allocatable memory. A fixed headroom
  # can still exceed allocatable RAM on larger nodes and Batch marks the job as
  # MISCONFIGURATION before it even tries to launch an instance.
  VCPU[$t]=$v; MEMMB[$t]=$(( m * 80 / 100 ))
  echo "  $t: ${v} vCPU, $(( m/1024 )) GiB -> job mem ${MEMMB[$t]} MiB"
done
TYPES_OK="${!VCPU[*]}"
[ -z "${TYPES_OK}" ] && { echo "no usable instance types" >&2; exit 1; }

cleanup() {
  [ "${KEEP}" = "1" ] && { echo "--keep: leaving compute envs/queues in place"; return; }
  echo "== tearing down =="
  for t in ${TYPES_OK}; do
    [ -n "${JQ[$t]:-}" ] && aws batch update-job-queue --job-queue "${JQ[$t]}" --state DISABLED --region "${REGION}" >/dev/null 2>&1
  done
  for t in ${TYPES_OK}; do
    [ -n "${JQ[$t]:-}" ] || continue
    for _ in $(seq 1 20); do
      st=$(aws batch describe-job-queues --job-queues "${JQ[$t]}" --region "${REGION}" --query "jobQueues[0].status" --output text 2>/dev/null)
      [ "$st" = "VALID" ] || [ "$st" = "None" ] && break; sleep 5
    done
    aws batch delete-job-queue --job-queue "${JQ[$t]}" --region "${REGION}" >/dev/null 2>&1
  done
  sleep 5
  for t in ${TYPES_OK}; do
    [ -n "${CE[$t]:-}" ] && aws batch update-compute-environment --compute-environment "${CE[$t]}" --state DISABLED --region "${REGION}" >/dev/null 2>&1
  done
  for t in ${TYPES_OK}; do
    [ -n "${CE[$t]:-}" ] || continue
    for _ in $(seq 1 24); do
      st=$(aws batch describe-compute-environments --compute-environments "${CE[$t]}" --region "${REGION}" --query "computeEnvironments[0].status" --output text 2>/dev/null)
      [ "$st" = "VALID" ] || [ "$st" = "None" ] && break; sleep 5
    done
    aws batch delete-compute-environment --compute-environment "${CE[$t]}" --region "${REGION}" >/dev/null 2>&1 \
      && echo "  deleted ${CE[$t]}"
  done
}
trap cleanup EXIT

echo "== creating compute environments =="
for t in ${TYPES_OK}; do
  ce="rtrrl-bench-$(san "$t")"; CE[$t]=$ce
  aws batch create-compute-environment --region "${REGION}" \
    --compute-environment-name "$ce" --type MANAGED --state ENABLED \
    --compute-resources "{
      \"type\": \"EC2\", \"allocationStrategy\": \"BEST_FIT\",
      \"minvCpus\": 0, \"maxvCpus\": ${VCPU[$t]}, \"desiredvCpus\": 0,
      \"instanceTypes\": [\"$t\"], \"subnets\": ${SUBNET_JSON},
      \"securityGroupIds\": [\"${SECURITY_GROUP_ID}\"],
      \"instanceRole\": \"${INSTANCE_PROFILE_ARN}\"
    }" >/dev/null 2>&1 && echo "  creating $ce ($t)" || echo "  $ce may already exist"
done

echo "== waiting for compute environments to be VALID =="
for t in ${TYPES_OK}; do
  for _ in $(seq 1 30); do
    st=$(aws batch describe-compute-environments --compute-environments "${CE[$t]}" --region "${REGION}" --query "computeEnvironments[0].status" --output text 2>/dev/null)
    [ "$st" = "VALID" ] && { echo "  ${CE[$t]} VALID"; break; }
    [ "$st" = "INVALID" ] && { echo "  ${CE[$t]} INVALID!"; break; }
    sleep 8
  done
done

echo "== creating queues =="
for t in ${TYPES_OK}; do
  q="rtrrl-bench-$(san "$t")-q"; JQ[$t]=$q
  aws batch create-job-queue --region "${REGION}" --job-queue-name "$q" --state ENABLED \
    --priority 1 --compute-environment-order "order=1,computeEnvironment=${CE[$t]}" >/dev/null 2>&1 \
    && echo "  $q" || echo "  $q may already exist"
done
sleep 8

echo "== submitting benchmark jobs (logging: ${MODE:-none}) =="
for t in ${TYPES_OK}; do
  # Per-type command: log normally and tag the run by instance type so the
  # benchmark runs are comparable in Aim / W&B.
  CMD="[\"python\",\"${ENTRY}\",\"--config_path\",\"/tmp/run-config.yml\""
  if [ "${ENTRY}" != "rtrrl.py" ]; then
    CMD="${CMD},\"--run_name\",\"bench-$(san "$t")\""
  fi
  if [ -n "${MODE}" ]; then
    CMD="${CMD},\"--logging\",\"${MODE}\""
    case "${MODE}" in *aim*) CMD="${CMD},\"--log_repo\",\"${AIM_SERVER}\"" ;; esac
  fi
  [ -n "${STEPS}" ] && CMD="${CMD},\"--num_timesteps\",\"${STEPS}\""
  CMD="${CMD}]"
  JOB[$t]=$(aws batch submit-job --region "${REGION}" --job-name "bench-$(san "$t")" \
    --job-queue "${JQ[$t]}" --job-definition "${JOB_DEF}" \
    --container-overrides "{\"command\": ${CMD},
      \"resourceRequirements\": [{\"type\":\"VCPU\",\"value\":\"${VCPU[$t]}\"},{\"type\":\"MEMORY\",\"value\":\"${MEMMB[$t]}\"}],
      \"environment\": [{\"name\":\"CONFIG_B64\",\"value\":\"${CONFIG_B64}\"}]}" \
    --query "jobId" --output text 2>/dev/null)
  echo "  $t -> ${JOB[$t]}"
done

echo "== waiting for jobs (instance launch + run); timeout ~${TIMEOUT_MIN} min each =="
for i in $(seq 1 $(( TIMEOUT_MIN * 60 / 15 )) ); do
  pending=0
  for t in ${TYPES_OK}; do
    [ -n "${STAT[$t]:-}" ] && continue
    s=$(aws batch describe-jobs --jobs "${JOB[$t]}" --region "${REGION}" --query "jobs[0].status" --output text 2>/dev/null)
    case "$s" in
      SUCCEEDED|FAILED) STAT[$t]=$s; echo "  [$(date +%H:%M:%S)] $t -> $s" ;;
      *) pending=$((pending+1)) ;;
    esac
  done
  [ "$pending" = "0" ] && break
  sleep 15
done
for t in ${TYPES_OK}; do [ -z "${STAT[$t]:-}" ] && STAT[$t]="TIMEOUT"; done

echo "== collecting timings + spot prices =="
for t in ${TYPES_OK}; do
  read -r started stopped < <(aws batch describe-jobs --jobs "${JOB[$t]}" --region "${REGION}" \
    --query "jobs[0].[startedAt,stoppedAt]" --output text 2>/dev/null)
  if [ "${started}" != "None" ] && [ "${stopped}" != "None" ] && [ -n "${stopped}" ]; then
    DUR[$t]=$(awk "BEGIN{printf \"%.1f\", (${stopped}-${started})/1000}")
  else
    DUR[$t]="NA"
  fi
  p=$(aws ec2 describe-spot-price-history --instance-types "$t" --region "${REGION}" \
    --product-descriptions "Linux/UNIX" --start-time "$(date -u +%Y-%m-%dT%H:%M:%S)" \
    --query "min(SpotPriceHistory[].SpotPrice)" --output text 2>/dev/null)
  PRICE[$t]=${p:-NA}
done

echo
echo "================= BENCHMARK RESULTS ================="
printf "%-14s %5s %8s %10s %12s %10s\n" "instance" "vCPU" "status" "runtime_s" "spot_\$/hr" "\$/run"
printf "%-14s %5s %8s %10s %12s %10s\n" "--------" "----" "------" "---------" "---------" "------"
# Sort by runtime (fastest first); NA timings sort last.
for t in ${TYPES_OK}; do
  cost="NA"
  if [ "${DUR[$t]}" != "NA" ] && [ "${PRICE[$t]}" != "NA" ]; then
    cost=$(awk "BEGIN{printf \"%.4f\", ${DUR[$t]}/3600*${PRICE[$t]}}")
  fi
  key="${DUR[$t]}"; [ "$key" = "NA" ] && key=999999
  printf "%s\t%-14s %5s %8s %10s %12s %10s\n" "$key" "$t" "${VCPU[$t]}" "${STAT[$t]}" "${DUR[$t]}" "${PRICE[$t]}" "$cost"
done | sort -n | cut -f2-
echo "===================================================="
echo "runtime_s = container start->stop for the SAME workload (lower = faster)."
echo "\$/run     = runtime x current Spot price (lower = more cost-efficient)."
echo "Note: jobs ran on-demand for timing stability; cost is projected at Spot."
