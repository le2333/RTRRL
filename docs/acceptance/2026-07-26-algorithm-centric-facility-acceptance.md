# Algorithm-centric training facility — acceptance run

**Date:** 2026-07-26 (UTC)
**Commit:** `1323850c9981719d2f23f07927277e68646b1d4a`
**Account / region:** 007122174918 / eu-north-1
**Algorithm:** `brax_ppo_acceptance`, an infrastructure-owned Brax PPO trainer. No
part of `memo/` participates, and none was modified.

Three Batch jobs, all succeeded, exit code 0. The control plane ran on the
permanently-on micro instance in the foreground, one CLI call per launch.

## Images

Built and pushed by `build-infra-acceptance-image.yml` run 30178762702. Both
experiment files name the digest, never the tag: a tag can be moved under a run
that is meant to be a record.

| Variant | Tag | Digest |
| --- | --- | --- |
| CPU | `infra-acceptance-brax-ppo-cpu-20260725` | `sha256:d84ccca3d066ed070bd39840aebb0b04dc23d97dcaac544d2fcbca28d73dd9d9` |
| GPU | `infra-acceptance-brax-ppo-gpu-20260725` | `sha256:5153ca698521cde78f5a61eea46ed4b38898197491633176ca2cef8c263bb9d0` |

Both carry the contract 2 catalog as label `org.rtrrl.trainer.catalog.v2`, and both
default to `python -m training_sdk.worker`. Job definitions
`trainer-{c7am,c7al,c7ax,g6x}-<digest>` were registered at revision 1 by
`scripts/deploy_facility.py --register`, logging to `/trainer/jobs`.

Every run reports source hash `sha256:f4a3979cb20b6684…` — one algorithm, two
images.

## CPU study

```bash
uv run trainerctl validate examples/experiment-acceptance.yaml --backend batch
uv run trainerctl run examples/experiment-acceptance.yaml --backend batch \
  > /tmp/acceptance-cpu.out 2> /tmp/acceptance-cpu.err
```

Exit 0. Launch `20260726-003613`, status `succeeded`, 426.3s.

Two rounds of two trials, `parallel_jobs: 1`, so each round is one job that runs
its two trials one after the other. That is the point of this shape: serial packing
inside a worker had never run on Batch before.

| Trial | learning_rate | Score | Job |
| --- | --- | --- | --- |
| 0 | 1.307e-4 | 21.0 | `276d0353-1619-401d-a06c-8932a19767c6` |
| 1 | 1.591e-4 | 21.0 | `276d0353-1619-401d-a06c-8932a19767c6` |
| 2 | 6.549e-4 | 25.0 | `fb1197c4-3802-48b4-ab32-9e43a7249da4` |
| 3 | 1.117e-4 | 21.0 | `fb1197c4-3802-48b4-ab32-9e43a7249da4` |

Best: trial 2, 25.0. Four trials across two job ids, two trials to each — the
serial path. Round two's rates are not round one's: TPE sampled them after reading
round one's scores from S3, which is the loop this run exists to prove.

## GPU configuration

```bash
uv run trainerctl validate examples/experiment-acceptance-gpu.yaml --backend batch
uv run trainerctl run examples/experiment-acceptance-gpu.yaml --backend batch \
  > /tmp/acceptance-gpu.out 2> /tmp/acceptance-gpu.err
```

Exit 0. Launch `20260726-004409`, status `succeeded`, 282.3s. One trial at
`learning_rate: 3e-4` scored 22.0 on `g6.xlarge`, job
`8a936497-c3a2-4217-b06d-14f702be6b06`.

## Artefacts

Under `s3://rtrrl-artifacts-007122174918/trainer/infra-acceptance/<name>/<launch>/`
both launches hold `experiment.yaml`, `launch.json`, `space.json`, `report.json`
and a manifest per round. Every trial has `config.json`, `score.json` and at least
one `.rrd`:

```
trials/t2/config.json
trials/t2/score.json                {"run_id": "...-t2", "trial": 2, "value": 25.0}
trials/t2/episodes/episode-000002.rrd
```

The scores in S3 match the report and the reported best, so what Optuna read is
what the worker wrote.

Aim holds all six runs from today — the dev smoke plus these five — under
experiment `infra-acceptance`, each with `launch_id`, `trial`, `entry`, the full
parameter set, the image digest and the source hash, and both `episode_return` and
`episode_length` tracked live over the direct RPC connection to the control plane's
private address.

## What the paid path caught that nothing else could

Recorded because each of these was invisible to every test that mocks AWS, and each
would have failed on a billed host:

- **`ecr:DescribeImages` is not granted.** `resolve_image` used it to turn a tag
  into a digest. `batch_get_image` returns the digest beside the manifest it needed
  anyway, so the call is gone. Both reference forms are now proven against the real
  account.
- **A loopback Aim endpoint.** The endpoint is copied verbatim into every run
  config, so `127.0.0.1` pointed each job at itself while preflight — running on the
  control plane — connected to the real server and reported success. Preflight now
  reads the address and refuses loopback for batch launches.
- **No region, twice.** `trainerctl` built its boto3 session without one, and the
  submitted containers were never told one either; boto3 does not ask the instance,
  so the worker's first S3 call would have raised `NoRegionError`.
- **A bucket that does not exist.** The examples named `rtrrl-training-data`.
- **A builder that never worked.** The image workflow had failed on every run since
  the catalog was introduced, because it invoked `build_catalog.py` from the wrong
  directory. The images in ECR predated the catalog label that preflight requires,
  so nothing usable had actually been published.
