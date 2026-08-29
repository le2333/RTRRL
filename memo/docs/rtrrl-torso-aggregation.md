# Where the two heads' credit for the shared torso is combined

RTRRL has one recurrent torso and two readouts that both read it. Each readout
names an objective, and each objective's ascent reaches the torso's parameters
through the same recurrent sensitivity. Something has to add the two.

Until now the addition happened in one place and was never named: the actor's
cotangent and the critic's were summed, the sum was pulled back through the
torso, and one trace and one optimizer carried it from there. That is the
published topology and it is kept unchanged. What is new is that it is now one
of two positions a run can select, and that the other one is a different
algorithm rather than a different spelling of the same one.

    torso.optimizer.kind:  adam | sgd | d_rtrrl | input_iu | input_obgd
                                                | output_iu | output_obgd
    actor.optimizer.kind:  adam | sgd | d_rtrrl | iu        | obgd
    critic.optimizer.kind: adam | sgd | d_rtrrl | iu        | obgd

|                        | intentional update | ObGD          |
| ---------------------- | ------------------ | ------------- |
| combined before the rule | `input_iu`       | `input_obgd`  |
| combined after the rule  | `output_iu`      | `output_obgd` |

`adam`, `sgd` and `d_rtrrl` are input-aggregated and unchanged. The two plain
rates carry no position because they have nothing to gain from one: both are
linear in the direction they are handed, so `Rate(a + b)` *is*
`Rate(a) + Rate(b)` — to the last bit for SGD, and for Adam up to the moments
it would then have to keep two of. The split is offered exactly where it
changes the answer.

A readout selects from the same set under shorter names. `iu` and `obgd` there
are what `input_iu` / `output_iu` and `input_obgd` / `output_obgd` are on the
torso: a head has one contribution and one trace, so there is no position for
it to name. The asymmetry is the two blocks being different things rather than
a name missing from one of them, and it is what lets a run put **one** update
rule on all three blocks — which an update-rule comparison needs, since an arm
running Adam on the readouts and ObGD on the torso is measuring neither.

## The two positions

Write `u_actor` and `u_critic` for the two heads' cotangents on the torso's
output, and `Jᵀ` for the pullback through the recurrent sensitivity.

**Input.** One derivative, one eligibility trace, one rule state, one step:

    p        = Jᵀ (u_actor + u_critic)
    z_t      = γ·λ_rnn·(1 - reset)·z_{t-1} + m_t·p_t
    Δθ_torso = Rule(p, z, δ·η_f)

**Output.** Two of each, added only where the parameters are written:

    p_actor  = Jᵀ u_actor              p_critic = Jᵀ u_critic
    z_actor  at γ·λ_π                  z_critic at γ·λ_v
    Δθ_actor = Rule_actor(...)         Δθ_critic = Rule_critic(...)

    Δθ_torso = Δθ_actor + Δθ_critic

Both positions read **one** forward pass and **one** recurrent sensitivity. The
output position calls the same pullback twice rather than computing a second
sensitivity, so what is independent is the state each path carries, not the
recurrence they read.

## Why this is two algorithms and not one

Because neither rule that can sit here is linear in what it is given:

- the intentional update preconditions by `nu`, a second moment of the
  instantaneous derivative, and divides its step size by `√(σ̄ · ⟨ρz, z⟩)`, a
  statistic of the trace;
- ObGD shrinks its rate by `1 / max(1, |δ|·‖z‖₁·lr·κ)`, which reads the norm of
  the trace it is handed.

So `Rule(a + b) ≠ Rule(a) + Rule(b)`, and the position is a property of the run
rather than a detail of the implementation.

ObGD's non-linearity is *conditional*: below its bound the rule is `lr·δ·z`,
which does distribute, and the two positions then agree to the last bit. That
is a true statement about the plain bound rather than a defect, and
`tests/unit/algorithms/rtrrl/test_torso_aggregation.py` runs its
"not two names for one rule" case at settings where the bound engages, and
checks that it did.

## What each output branch is

A branch is a whole single-path learner over the torso's parameters. It owns:

- its own instantaneous derivative, pulled back from one head's cotangent;
- its own eligibility trace, at **that head's** `γ·λ` — `γ·λ_π` for the actor
  and `γ·λ_v` for the critic, not the joint path's `γ·λ_rnn`, because the trace
  belongs to the objective and not to the block it credits;
- its own rule state: `nu`, `σ̄`, the TD clipping statistic and the advantage
  scale under the intentional update; the bound and second-moment statistics
  under ObGD;
- its own settings, `eta` and `kappa` and the base rate above all;
- its own dynamic step size and its own readings.

Changing one branch's configuration moves nothing on the other. Two states
sharing one set of statistics and differing only by a scalar would not be this
structure, and the suite asserts it is not what happened.

### The signals

The two branches are the intentional-update paper's two algorithms, each over
the contribution it belongs to:

- **actor → torso** is the intentional policy gradient: the TD error is clipped
  to `C` times its own running RMS, normalized by the running mean of its own
  magnitude, and the step is proportional to that advantage;
- **critic → torso** is Intentional TD: the step is proportional to the clipped
  TD error.

The joint path is **neither**, and should not be described as either. What
reaches it is `Jᵀ(u_actor + u_critic)`, which is not `∇V` and not `∇log π`. Its
step size is a real quantity — the same `eta` over the same denominator — but
the functional reading of `eta` does not survive the sum: it no longer names a
fraction of a TD error that one step sets out to spend. Do not read the joint
torso's `eta` on the same axis as an output branch's, or as a head's.

### The entropy term

The entropy direction belongs to the actor's objective, so it reaches the
actor's branch and no other. Whether it is *traced* or applied on the step it
arises follows from the branch's rule, exactly as it does everywhere else:

- under the intentional update it is folded into the traced derivative and
  signed by the TD error, because the paper's policy gradient is the derivative
  of the log-probability and the entropy together;
- under ObGD it stays untraced, as the published implementation applies it.

### The TD error

Both branches are credited by `δ·η_f`. `η_f` is RTRRL's own dial on how far the
shared representation moves relative to the readouts; it is a property of
crediting the torso rather than of where the crediting is combined, so it
applies at both positions and on every path.

## What bounds what

Each branch's rule bounds or sizes **its own** contribution and nothing else.
The sum is an elementwise addition — no clip, no norm, no rescale — so it may
be longer than either part. That is what it means for the two paths to be
independent all the way to the parameters, and adding an outer bound over the
total would be a limit no configuration declared and no branch could account
for.

For the same reason `torso.grad_clip` must be `0` for all four of the new
branches. The intentional update derives its step size from statistics it
carries; ObGD shrinks its rate so that one update cannot cross the TD target.
Clipping the finished step afterwards is a second, undeclared bound over a rule
that already has one, and the build refuses it with that reason rather than
accepting a run that is not the algorithm it names. `adam` and `d_rtrrl` keep
the clip, because for them it is the only bound there is.

One write, in both positions: the summed update is applied once, the kernel
constraint is projected once, and the followed reading copy advances once.
Applying one branch and then computing the other would make the pair
order-dependent and would not be this topology.

## Readings

A reading exists where the quantity does, so the position decides which names a
run files.

| position | filed under |
| --- | --- |
| input  | `update.torso.{grad_norm, trace_norm, step_size}` |
| output | `update.torso.{actor,critic}.{grad_norm, trace_norm, step_size}` |

Under an output aggregation there is no joint derivative, no joint trace and no
joint step size, so the joint names are absent rather than reporting one branch
or a sum of two. `step_size` is the dynamic step under the intentional update
and the rate the bound left under ObGD, under the one name every rule reports
its step scale by. The catalog advertises both sets, because a catalog says
what some configuration can file; no single run files all of it.

### Behind the step size

The position decides *where* those are filed; the rule decides what else is
filed beside them. Both rules on offer here derive their own step size, and in
both cases that number is a quotient whose terms cannot be recovered from it —
a run that could only see the quotient could not tell a rate held down by a
large trace from one held down by a large TD error. So each rule reports what
the step passed through, under its own name:

| rule | filed under |
| --- | --- |
| intentional | `…intentional.{clipped_delta, signal, advantage_scale, rms_scale, sigma_bar, trace_quadratic, denominator, update_norm, non_finite}` |
| ObGD | `…obgd.{trace_sum, delta_bar, bound_denominator, bound_scale, second_moment_rms}` |

where `…` is `update.torso` at the input position and
`update.torso.{actor,critic}` at the output one, so an output run has one set
per branch and the two are readable apart.

The ObGD five are exactly the terms of

    step_size = lr / max(1, delta_bar · trace_sum · lr · kappa)

- `trace_sum` is the L1 the bound reads: `sum |z|` under the plain bound and
  `sum |z| / sqrt(v_hat)` under the adaptive ones, which is why it is not the
  `trace_norm` filed beside it.
- `delta_bar` is `max(|delta|, 1)`, the TD error *as the bound reads it*. It is
  reported rather than derived from `update.td_error` because the torso's error
  is scaled by `eta_f` first, and the two are then different numbers.
- `bound_denominator` is `delta_bar · trace_sum · lr · kappa`, **before** the
  maximum with one. The bound is active exactly where it exceeds one. Before
  the maximum, because afterwards a run approaching its bound and a run nowhere
  near it are the same number, and which of those a run is is the thing worth
  seeing.
- `bound_scale` is what survived: `step_size / lr`, in `(0, 1]`.
- `second_moment_rms` is the mean of the denominator the adaptive bounds divide
  by — `sqrt(v_hat) + eps` or `sqrt(v_hat + eps)`, whichever was declared. It
  is absent under the plain bound, where nothing normalizes.

Together they answer the question the step size alone cannot: whether a small
rate is a small base rate, a large trace, a large TD error, or a second moment.

A readout files the same five under `update.actor.obgd.*` and
`update.critic.obgd.*` when it selected ObGD. There is one set per block rather
than per position there, because a head has one bound.

## The frozen ObGD arm

For the update-rule comparison — ObGD against Adam, Adam with `beta1 = 0`, SGD
and the intentional update — the arm is **one rule on all three blocks**:

```yaml
actor:   {optimizer: {kind: obgd,       obgd: {bound: {kind: adaptive_ob_fixed}}}}
critic:  {optimizer: {kind: obgd,       obgd: {bound: {kind: adaptive_ob_fixed}}}}
torso:   {optimizer: {kind: input_obgd, input_obgd: {bound: {kind: adaptive_ob_fixed}}}}
```

Two things are pinned rather than swept, and both for the same reason — the arm
is there to evaluate a published method, not to explore its variants:

- **`adaptive_ob_fixed`**, because it is the published denominator
  (`sqrt(v_hat + eps)`). `adaptive_ob` is `sqrt(v_hat) + eps`, which is what
  this repository and everything forked from it computed, and the two stand
  about a factor of two apart while the second moment is near `eps`. An arm
  that let a sweep choose between them would be reporting a number for "ObGD"
  that depends on which of two rules a trial happened to draw.
- **`input_obgd`**, because the input position is the reference RTRRL topology:
  the two heads' recurrent credit is combined before the update, exactly as it
  was before there was a second position. An arm that changed the update rule
  *and* the credit topology could not attribute a difference to either.

`output_obgd`, `ob` and `adaptive_ob` stay available and stay out of the main
experiment. They belong to a study of the aggregation position or of the bound
variants, which is a different question from what this arm asks.

## Migrating

`torso.optimizer.kind: iu` named what is now `input_iu` and meant exactly that.
The branch is gone rather than aliased: a run document that still carries the
old name is refused at build with the branches it could have named. A name that
had one meaning and now has a position in it is worth failing on, so that the
migration is something a reader performed rather than something performed for
them. `adam` and `d_rtrrl` configurations are untouched.

The ObGD branches declare their bound as a nested choice —
`torso.optimizer.input_obgd.bound.kind` is one of `ob`, `adaptive_ob`,
`adaptive_ob_fixed` — and a base rate beside it. They reach
`make_bounded_rule`, the same rule StreamAC answers to; nothing about the bound
is re-implemented at the algorithm's entry.

## Where this lives

`memorax/algorithms/rtrrl_aaai.py` holds `TorsoAggregation` and its two
implementations, `InputAggregation` and `OutputAggregation`, along with the six
branches `RTRRL_TORSO_OPTIMIZERS` offers. The three other RTRRL graphs — the
CTRNN, LSTM and dense state-space torsos — select from the same family and
inherit the flow, because where the two credits meet is a property of sharing a
torso rather than of which recurrence the torso runs.

`tests/unit/algorithms/rtrrl/test_torso_aggregation.py` drives each branch
against a single-path reference built from `Trace` and either
`IntentionalOptimizer` or `make_bounded_rule`, and holds the four modes against
a real graph. `tests/unit/algorithms/rtrrl/test_rtrrl_assembly.py` holds the
input position against everything that was true of the torso before there was a
second one.
