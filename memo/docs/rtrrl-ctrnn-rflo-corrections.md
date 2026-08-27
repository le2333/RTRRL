# RTRRL-CTRNN-RFLO: what the published implementation gets wrong

`memorax/algorithms/rtrrl_ctrnn_rflo.py` and
`memorax/networks/sequence_models/ctrnn.py` are a rebuild of
`RTRRL-AAAI25/models/ctrnn.py` and the CTRNN path of `RTRRL-AAAI25/rtrrl.py`.
They are not a bit-for-bit reproduction. Where the published code and the
paper's equations disagree, the equations decide; the published code is the
authority on the network's shape, the order the transition is processed in, and
the experiment's settings.

This is the list of everything that disagreed, one section each. Every section
says where the published behaviour is, what the correct semantics is and where
it comes from, why the published behaviour is wrong, what this repository does
instead, and the test that fails if the correction is removed.

Six were found; they are not all corrected in the same sense, and the
difference is worth stating before the list rather than leaving a reader to
infer it:

- **Four are corrected here** — 1, 2, 3 and 4. They are on the path this
  algorithm runs, and this repository does something different from the
  published code.
- **One is recorded and not carried** — 5. It is in the `rtrl` branch, which
  this repository does not implement, so there is nothing here to correct. It
  is measured rather than asserted, because a claim about somebody else's code
  should be.
- **One was already corrected** — 6. It belongs to the shared flow and
  `rtrrl_aaai` fixed it before this work; this entry inherits it, and it is
  listed so the accounting of what the published implementation gets wrong is
  complete rather than only covering what was new.

Nothing here was fixed silently, and nothing here is a preference.

## Why all six survived

The published default configuration is `dt = 1`, `wiring: fully_connected`,
`gradient_mode: rflo`, `update_period: 1` — and under it not one of the six
defects below can be observed. `dt = 1` makes a missing `dt` invisible. An
all-ones mask makes a mask that never reached the trace invisible, and takes
the miscut wiring out of the run entirely. Nothing sits in front of the
recurrent cell in RTRRL's torso, so an input cotangent that is structurally
zero is never read. `gradient_mode: rflo` and `update_period: 1` take the other
two branches out of the flow.

They are not therefore harmless. Five of the six are a setting away -- another
`dt`, a wiring, `gradient_mode: rtrl`, an `update_period` -- so they are wrong
answers waiting for a configuration to ask for them, and a defect that only the
*non-default* settings reach is a defect a sweep finds before a reader does. The
sixth needs new code rather than a new setting: it costs nothing until something
is put in front of the cell, and then it costs that thing everything.

So the parity suite is written in both directions:
`tests/test_ctrnn_rflo_parity.py` holds this side to the published cell **to
within two float32 last bits** under that configuration, and holds it *away*
from the published cell — by millions of last bits — as soon as any of the
conditions that hid a defect is lifted. Agreement alone could not distinguish a
correction from a mistake.

| | published behaviour | this side | hidden by |
| --- | --- | --- | --- |
| 1 | `dt` absent from both RFLO recurrences | carried | `dt = 1` |
| 2 | the trace reads the unmasked `W` | reads the network's `W` | `fully_connected` |
| 3 | `no_self` masks the wrong columns | masks the recurrent diagonal | `fully_connected` |
| 4 | no cotangent reaches the input | the step's own Jacobian | nothing precedes the cell |
| 5 | `rtrl` mode credits `W` over the wrong axis | mode not carried | `gradient_mode: rflo` |
| 6 | the reading copy aliases the stepped one | two trees | `update_period: 1` |

---

## 1. The trace does not carry the integration step it is the derivative of

**Published** — `models/ctrnn.py:127-129`:

```python
jw += (1 / tau)[:, None] * (M_immediate - jw)
dh_dtau = ((h - jnp.tanh(u)) / tau) - jtau
jtau += dh_dtau / tau
```

**Correct** — the cell integrates `h_t = h_{t-1} + (dt/tau)(phi(u_t) - h_{t-1})`
(`models/ctrnn.py:60`, which does use `self.dt`). RFLO (Murray 2019, eq. 6) is
that step's forward sensitivity with the recurrent path through `phi` dropped:

```
p^W_{ij}  <- (1 - dt/tau_i) p^W_{ij} + (dt/tau_i) phi'(u_i) v_j
p^tau_i   <- (1 - dt/tau_i) p^tau_i  + (dt/tau_i^2) (h_{t-1,i} - phi(u_i))
```

**Why the published behaviour is wrong** — both the decay and the immediate
Jacobian are scaled by `1/tau` rather than `dt/tau`, so the trace is the
derivative of a step the network did not take. At `dt = 1` the two coincide,
which is why every reported result is unaffected; at any other `dt` the forward
still agrees between the two implementations and the credit no longer does.

**Here** — `CTRNNCell.local_jacobian` derives `rate = dt/tau` once and uses it
for the decay, for the immediate Jacobian, and (as `rate/tau`) for `tau`'s own
term.

**Tests** — `tests/test_ctrnn_rflo.py::test_the_trace_carries_the_integration_step_it_is_the_derivative_of`
holds the corrected recurrence against the equations and *away* from the
published scaling, and shows the two agreeing again at `dt = 1`.
`tests/test_ctrnn_rflo_parity.py::test_the_integration_step_reaches_this_trace_and_not_the_published_one`
does the same against the published code itself.

---

## 2. The trace is computed about a different network than the forward

**Published** — `models/ctrnn.py:54` masks the weights inside the forward:

```python
W = jax.lax.stop_gradient(mask) * W
```

but `OnlineCTRNNCell.__call__` hands `rflo_murray` the raw parameter tree
(`_p = mdl.variables["params"]`), and `models/ctrnn.py:114` forms the
pre-activation from it:

```python
u = v @ W.T          # W is params["W"], never multiplied by the mask
```

**Correct** — the sensitivity is `dh/dtheta` of the function the network
computes. The masked entries are not parameters of that function at all — the
mask is a constant, so `du_i/dW_ij = M_ij v_j` — and `phi'(u)` has to be
evaluated at the `u` the forward produced.

**Why the published behaviour is wrong** — two separate errors follow from one
line. The slope `phi'(u)` is taken at a pre-activation the network never
computed, so every entry of the trace is wrong, not only the masked ones. And
the masked entries accumulate credit and are stepped by the optimiser, which
the forward then multiplies away — so a masked run trains parameters that
cannot affect it while reporting gradient norms that count them.

**Here** — `CTRNNCell._weights` is the single place `W` is read, in the forward
and in `local_jacobian` alike, and the immediate Jacobian is multiplied by the
same mask.

**Tests** — `tests/test_ctrnn_rflo.py::test_a_masked_connection_is_masked_in_the_trace_too`
(zero credit exactly where the mask is, and the surviving entries differ from
the published ones) and
`tests/test_ctrnn_rflo_parity.py::test_a_wiring_reaches_this_trace_and_the_published_mask_is_a_column_out`.

---

## 3. `fully_connected_no_self` removes the wrong connections

**Published** — `models/wirings.py:21-26`:

```python
rem = input_size - output_size
return mask - jnp.concatenate(
    [jnp.zeros((output_size, rem)), jnp.eye(output_size)], axis=1
)
```

**Correct** — the column layout of `W` is `[input, hidden, bias]`
(`models/ctrnn.py:21`), so unit `i`'s self-connection is column
`features + i`. A wiring that removes self-connections zeroes the diagonal of
the recurrent block.

**Why the published behaviour is wrong** — the identity is placed on the *last*
`hidden_dim` columns, which under this layout is one column to the right of the
recurrent block. It therefore removes unit `i`'s reading of unit `i + 1`, and
for the last unit it removes the *bias* — which `make_mask_initializer` then
restores with `mask.at[:, -1].set(1.0)`, leaving that unit fully
self-connected. Every unit keeps the connection the wiring is named after.

**Here** — `wiring_mask("no_self", ...)` subtracts the identity from columns
`features .. features + hidden_dim`, and the name is `no_self` rather than
`fully_connected_no_self` so that a configuration cannot be moved across
without noticing.

**Test** — `tests/test_ctrnn_rflo.py::test_the_no_self_wiring_removes_each_unit_s_own_previous_state`
asserts the recurrent block is `1 - I` and that neither an input column nor the
bias was touched.

---

## 4. Nothing in front of the cell can receive a gradient

**Published** — `models/ctrnn.py:132` returns the input sensitivity unchanged:

```python
dh_dx = jx
```

`jx` is allocated as zeros in `initialize_carry` (`models/ctrnn.py:187`),
written by the `rtrl` branch and never by the `rflo` branch, and contracted
with the cotangent at `models/ctrnn.py:176`:

```python
grads_x = jnp.einsum("...h,...hi->...i", df_dy, jx)
```

**Correct** — RFLO approximates credit through *time*. The cotangent that
leaves the cell towards its input on the current step is not an approximation
of anything: it is `dh_t/dx_t = (dt/tau_i) phi'(u_i) (M W)_{ij}`, the step's own
Jacobian.

**Why the published behaviour is wrong** — a cell that returns exactly zero to
its input silently disconnects everything upstream of it. Nothing precedes the
cell in RTRRL's torso, which is why no reported result is affected, and why the
first graph that puts a projection there would train it on nothing while
looking healthy.

**Here** — the input path is differentiated by autodiff through the step, so it
is the immediate Jacobian by construction.

**Tests** — `tests/test_ctrnn_rflo.py::test_the_input_keeps_its_own_immediate_jacobian`
(against the closed form) and
`tests/test_ctrnn_rflo_parity.py::test_the_input_gradient_is_this_step_s_and_the_published_one_is_zero`
(the published one is exactly zero).

---

## 5. The `rtrl` mode contracts the wrong axis of its own Jacobian

**Published** — `models/ctrnn.py:171`, the backward rule for every mode that is
not `rflo`:

```python
grads_p = jax.tree.map(lambda t: df_dy @ t, jp)
```

**Correct** — under `rtrl`, `jp[name]` is the full sensitivity
`p[i, j, ...] = dh_i / dtheta_{j...}`: axis 0 is the *output* unit, and the
parameter's own axes follow. The cotangent is a cotangent on `h`, so it
contracts axis 0: `grad_theta = sum_i ybar_i p[i, ...]`, which is what
`rtrl_ctrnn` itself does one line earlier when it propagates the trace
(`jnp.tensordot(dh_dh, p, axes=1)` contracts axis 0).

**Why the published behaviour is wrong** — `@` against a 1-D left operand does
not contract the same axis at every rank. For `tau`, whose trace is `(H, H)`,
`ybar @ p` is an ordinary vector-matrix product and contracts axis 0, which is
right. For `W`, whose trace is `(H, H, F + H + 1)`, NumPy's rule treats the
trace as a *stack* of `H` matrices and contracts axis 1 instead — so the
gradient is summed over the unit the parameter belongs to rather than over the
unit the cotangent is on. Measured against backpropagation through the same
three-step unroll: `tau` agrees to 1.6e-8 relative and `W` is 36% off. The mode
that exists to be exact is exact on one leaf of two.

**Not carried.** This repository implements the `rflo` branch, which the paper's
CTRNN experiments use and which the issue asks for; there is no `rtrl` branch
here to correct. `CTRNN_DIFFERENTIATION_FAMILY` carries `rflo` and `tbptt`, and
the `rtrrl_ctrnn_rflo` entry declares only `rflo` -- the entry's name is a claim
about which online gradient produced a result, and `tbptt` is the tests' exact
judge rather than a mode a run may select under that name. It is recorded because a later exact-RTRL CTRNN must not be
built by transcribing that line, and because it says something about how much
of the published online-gradient code was ever compared against anything.

**Test** — `tests/test_ctrnn_rflo_parity.py::test_the_published_exact_mode_is_exact_on_one_leaf_of_two`
characterises the published reference rather than this side, so the claim above
is checked rather than asserted.

### The commented-out identity is not the defect

`models/ctrnn.py:83` reads:

```python
dh_dh = df_dh * cell.dt  # + jnp.identity(cell.num_units)
```

which looks like a dropped `I` from the Euler step, and is not one:
`rtrl_step` two lines down is `p + rec + dh * dt`, and that leading `p` *is*
the identity term. Restoring the commented-out `jnp.identity` would count it
twice. Written down because it is the first thing a reader of that function
reaches for, and the numbers above say the propagation is right and the
contraction is what is wrong.

---

## 6. The reading copy and the stepped copy are one object

**Published** — `rtrrl.py:407` binds them to the same dictionary before the
loop, and `rtrrl.py:710-711` rebinds and then mutates it:

```python
slow_params = params
slow_params["params"]["rnn"] = rnn_slow_params
```

**Correct** — a Polyak-averaged reading copy is a second tree. The parameters
being stepped are one point and the copy the forward reads is another, and
`update_period` is how far the second moves towards the first.

**Why the published behaviour is wrong** — `slow_params = params` is a rebinding
of a name, not a copy, so the assignment on the next line writes the averaged
weights into `params["params"]["rnn"]` as well. The stepped copy is lost, both
names then denote the same tree, and `optax.incremental_update(a, a, rho) == a`
for every `rho` — so `update_period` is a no-op after the first transition
rather than the averaging it is named for. The default `update_period: 1` takes
the branch out of the flow entirely, which is why it was never seen.

**Here** — this is `rtrrl_aaai`'s correction rather than this work's, and this
entry inherits it: `TorsoState` carries `params` and `slow_params` as two
fields of a frozen node, and `Torso.followed` returns a new tree.

**Test** — `tests/test_rtrrl.py::test_the_reading_copy_lags_the_updated_one` and
`::test_following_all_the_way_makes_them_one_tree`, which predate this work.

---

## Deliberate differences that are not corrections

These are places this repository does something else on purpose. They are not
defects in the published code and are listed so that a numerical comparison
does not have to rediscover them.

- **The wiring mask is a constant, not a variable.** The published cell draws
  `random` and `ncp` masks from the model's `params` rng into a `wiring`
  collection. The boundary between a sequence and its differentiation here
  carries only `params`, so a key-dependent mask would have to be redrawn at
  each call site. Only the two key-free wirings are carried; see `WIRINGS`.
- **`tau` is projected by the component that owns it.** The published clip is
  two statements in the training loop (`rtrrl.py:717-721`); here the cell states
  the set in `CTRNNCell.constrain` and `Torso.constrain` applies it — to the
  stepped copy, to the reading copy, and to the initial draw, so the parameter
  is never outside its domain rather than being outside it for one transition.
- **Streams are not averaged inside the cell.** `models/ctrnn.py:174` takes a
  mean over a leading batch axis when there is one. RTRRL vmaps the gradient
  computation per stream, so that branch is unreachable in the published flow
  too; here the per-stream trace is the algorithm's and the averaging happens
  where the update is taken.
- **The affine-free `LayerNorm` behind the cell is a declared choice.**
  `RNNActorCritic` defaults it on, but `RTRRLParams.layer_norm` is `False` and
  is what `train_rtrrl` passes, and `config/brax.yml` does not set it -- so the
  published runs have no normalization behind the cell. It is `torso.layer_norm`
  here and defaults to the published value.

  Worth saying out loud, because it is a claim about the sibling algorithm too:
  `rtrrl_aaai`'s LRU torso appends that `LayerNorm` unconditionally, and its
  own docstring calls it "where and what the published implementation's is",
  which holds for `layer_norm: true` and not for the default. The parity suite
  cannot see it -- it re-hosts the published *wiring* on these same networks
  precisely so a mismatch cannot come from the networks, which leaves what
  follows the cell unexamined by construction, exactly as it leaves what
  precedes it. Not moved here: changing it would change the meaning of every
  landed `rtrrl` run, and it belongs to that algorithm rather than to this one.
- **The published flags that are off by default are not carried.**
  `RTRRLParams` declares `pass_obs`, `mlp_actor`, `f_align`, `pred_obs`,
  `dropout_rate`, `var_scaling`, `act_magnitude_factor` and `slow_rnn_factor`,
  and every one of them defaults to off or zero, is off in `config/brax.yml`,
  and is off in the reported CTRNN configuration. `rtrrl_aaai` carries none of
  them either. They are listed so that "the published structure" is a claim
  about a configuration rather than about a file: a run that turned one on
  would be running something this graph does not express, and would need to say
  so rather than assume it.
- **The two objectives are differentiated separately.** As in `rtrrl_aaai`: the
  published `td_loss` sums the actor's and the critic's objectives and
  differentiates once, this side differentiates each and adds the cotangents.
  They agree by linearity.

## Open questions

None. Every disagreement found between the published implementation and the
paper's equations is resolved above, in favour of the equations, with a test
that fails if the resolution is reverted.
