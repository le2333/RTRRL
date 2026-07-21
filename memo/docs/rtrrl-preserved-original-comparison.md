# Preserved Streaming RTRRL vs AAAI25

## Verdict

The preserved `rtrrl/rtrrl.py` LRU path is **not completely numerically
equivalent** to AAAI25 commit
`4301943c349171d828d0fcf3e40944c286451415`. Its default forward values,
initialization/PRNG schedule, LRU online-credit equations, eligibility-trace
equations, and incoming-trace update order map to AAAI25. Its actor
instantaneous gradient does not: the preserved script always applies
`stop_gradient` to the sampled action before `log_prob`, while AAAI25
differentiates through its reparameterized sample.

This status does not affect Memo strict parity. The external script is an
unchanged backup/reference and is not invoked by Memo.

## Identity and preservation contract

- preserved source: functional base
  `5f7ff4e40e66da0b7df4f3edc9a928185ad73ae6`;
- preserved byte SHA-256:
  `f8aedcd9c315445af93e7f4a2475c50e9828c5188bd487ed39b85d7ec7da61cf`;
- canonical AST SHA-256:
  `46d3b46a45ab72c3a9550763ae6f6fb0c5bda49a103731e953183f707e388ee9`;
- oracle source: AAAI25
  `4301943c349171d828d0fcf3e40944c286451415`.

`test_legacy_entrypoint.py` enforces both hashes. Final acceptance also requires
`git diff 5f7ff4e..HEAD -- rtrrl` to be empty.

## Measured focused probe

The committed `preserved_original_probe.py` runs the preserved and AAAI25
modules in separate Python processes and pinned dependency environments. Each
process evaluates both detached and reparameterized actor objectives using the
same explicit parameters and fixed noise. This 2×2 control measures the
within-runtime semantic delta and the cross-runtime same-semantic delta
separately. The comparison is performed by
`preserved_original_compare.py`.

Final Batch job `f2a0d8ba-bdc2-44fb-85c2-890cfc99989f` measures exact equality
for explicit parameters, initial carry, both forward carries/outputs,
accumulated trace, and trace-derived update. In both pinned runtimes, changing
only actor semantics produces these maximum absolute gradient deltas:

- location gradient: `1.0449076890945435`;
- raw-scale gradient: `0.5879773795604706`.

Holding semantics fixed across JAX 0.5.0 and 0.4.38 produces maximum absolute
differences of zero for objective, location gradient, and raw-scale gradient
under both detached and reparameterized semantics. The semantic delta therefore
does not arise from changing runtime in this control.

The source-native JAX environments did not produce equal PRNG/initialization
observables: the uint32 split maximum was `2793280701`, native LRU parameter
maximum was `5.048454284667969`, and the first/second native output maxima were
`1.8039704756811261`/`1.5446054488420486`. This is measured runtime behavior,
not attributed solely to source: the preserved process used JAX/JAXLIB 0.5.0,
while the pinned AAAI25 process used 0.4.38. Explicit equal parameters isolate
the LRU equations from that runtime PRNG difference and produce exact outputs.

Fresh Batch evidence, including both within-runtime semantic deltas,
cross-runtime same-semantic controls, source AST audit, and exact runtimes, is
recorded in `rtrrl-task12-evidence.json`.

## Source and operation-order audit

The normalization check is deliberately precise:

- `traces.py` ASTs are equal after removing conventional leading docstrings;
- `models/online_lru.py` ASTs are **not** equal after only that normalization,
  because the class contains an additional standalone descriptive string
  expression after field declarations;
- both files are equal after removing every standalone string expression
  statement. Comments are absent from Python ASTs and require no normalization.

Separately, a structural AST check finds that the preserved `log_prob` argument
is `stop_gradient(action)`, while AAAI25's is `action`. This is static
structural evidence; the controlled synthetic 2×2 objective measures the
numerical effect without claiming to execute the nested training closure
directly.

The preserved top-level script differs in four functional areas:

1. `run_name` is an added logging/configuration field and does not alter math.
2. `align_action_logprob` is added. With its default `False`, sampled action
   values match AAAI25; with `True`, continuous sampled actions are clipped
   before log-probability evaluation.
3. The sampled action is always wrapped in `jax.lax.stop_gradient` before
   `log_prob`. No preserved option disables this, so no preserved configuration
   exactly recovers the AAAI25 actor-gradient semantics.
4. `update_trace_before_td` is added. Its default `False` uses the incoming
   eligibility trace for the current TD update and accumulates the new
   instantaneous gradient afterward, matching AAAI25. Setting it to `True`
   intentionally selects fresh-trace semantics and differs from AAAI25.

The unchanged matching operation sequence is: six-way initial key split,
environment reset, carry initialization, `init_with_output` pre-sampling,
environment-first transition, terminal carry reset, meta-RL input assembly,
LRU/VJP and TD-head evaluation, TD target/error, incoming-trace update
direction, batch mean, optimizer application, slow-target update, emphasis or
average-reward update, then trace reset/accumulation.

## Reproducible and intentionally different settings

The closest preserved settings are the defaults
`align_action_logprob=False` and `update_trace_before_td=False`. They reproduce
AAAI25 forward/action values and trace timing, but **cannot** reproduce its
actor gradient because sampled-action detachment is unconditional.

Intentional alternatives are:

- `align_action_logprob=True`: clipped action/log-probability alignment;
- `update_trace_before_td=True`: current-gradient/fresh-trace TD update;
- `run_name`: logging identity only.

All other `RTRRLParams` defaults shared by the two source files are unchanged.

## Verification boundary

Measured: source-native PRNG split and initialization (different across pinned
runtimes), explicit-parameter two-step LRU forward/carry (exact), trace
accumulation/update (exact), fixed-noise actor objective (exact), and actor
gradients (different). Statically mapped: complete LRU step order and
configuration defaults.

Not independently executed as complete preserved-script environments:
terminal/reset scans, multi-environment reduction, normalization, Dutch traces,
average reward, discrete actions, CTRNN/no-RNN, rendering/evaluation, dropout,
MLP actor, and a full optimizer-backed preserved training scan. Because the
actor-gradient mismatch feeds future traces and optimizer state, complete-step
parity must not be inferred even where forward observables initially match.
