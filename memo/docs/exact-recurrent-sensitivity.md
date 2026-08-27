# Exact recurrent sensitivity on the structured diagonal core

RTRRL's torso learns online: it never unrolls, and it never truncates. What
makes both true at once is a forward sensitivity carried alongside the hidden
state, so that one step of ordinary autodiff produces the gradient a full unroll
would have produced. This document states that recurrence for each declared
core, says what is verified about it and against what, and says what bounds
the claim.

## The recurrence

Write one step of the recurrent kernel as `h_t = g(h_{t-1}, x_t; θ)` and the
forward sensitivity of the state with respect to a recurrent parameter as
`S_t = ∂h_t/∂θ`. RTRL is

    S_t = (∂g/∂h) S_{t-1} + ∂g/∂θ ,

with `S` reset to zero wherever the transition that produced `h_t` began a new
episode. Both cores instantiate it in closed form.

### RTU (`torso.backbone.rtu`)

State is a real pair per hidden unit, `h = (a, b)`, `h ∈ R^{2H}`. With

    r = exp(-exp(ν_log)),  ϑ = exp(θ_log),
    g = r cos ϑ,  φ = r sin ϑ,  γ = sqrt(1 - r²) + ε,

the step is

    p_t = g ⊙ a_{t-1} - φ ⊙ b_{t-1} + γ ⊙ (B_re x_t)
    q_t = g ⊙ b_{t-1} + φ ⊙ a_{t-1} + γ ⊙ (B_im x_t)
    a_t = f(p_t),  b_t = f(q_t),   f = tanh.

Unit `h` has recurrent Jacobian `A_h = [[g_h, -φ_h], [φ_h, g_h]]` and activation
Jacobian `D_{t,h} = diag(f'(p_{t,h}), f'(q_{t,h}))`, and the sensitivity of the
four recurrent blocks `θ ∈ {ν_log, θ_log, B_re, B_im}` is

    S_{t,h} = D_{t,h} ( A_h S_{t-1,h} + J_{t,h} ),   J_{t,h} = ∂(p_t, q_t)_h/∂θ .

`A_h` is a full 2x2 rotation and is carried entire. There is no `A` term between
units: `h`'s state is a function of `h`'s own previous state and of the input.

### LRU (`torso.backbone.lru`)

State is one complex number per unit. With `λ = exp(-exp(ν_log) + i exp(θ_log))`
and `γ = exp(γ_log)`, the step is linear,

    h_t = λ ⊙ h_{t-1} + (B_re + i B_im) diag(γ) x_t ,

so there is no activation Jacobian and the recurrence for
`θ ∈ {ν_log, θ_log, γ_log, B_re, B_im}` is

    S_t = λ ⊙ S_{t-1} + ∂h_t/∂θ .

The propagation is an associative scan over the window rather than a loop, which
is the same recurrence written for the parallel form.

### Episode boundaries

An ending is read before the step it ends: the carry is returned to its initial
value and `S` is set to zero, so a stream owes nothing to what it did before it
ended. Both are per stream.

### How the gradient is taken

The carried `S` is injected into the carry as a phantom, `h ← stop_grad(h) + Σ_θ
S_θ (θ - stop_grad(θ))`, whose value is unchanged and whose derivative with
respect to `θ` is `S_θ`. One step of autodiff then yields the exact recurrent
gradient, and RTRRL's two readout objectives enter it as two cotangents on the
same vector-Jacobian product.

## What is verified

`tests/test_exact_recurrent_sensitivity.py`, over both cores, sequence lengths
1, 2, 3, 5 and 8, nontrivial random parameters and inputs, and every recurrent
parameter block:

- the online walk arrives at the hidden state the truncation-free walk does;
- the carried sensitivity equals `jacrev` of the whole prefix, and the part of
  that Jacobian which crosses hidden units is exactly zero;
- the actor-source, critic-source and combined recurrent gradients each equal
  autodiff through the truncation-free unroll, and the combined one is the sum
  of the other two;
- an ending restarts the sensitivity, and a live step does not;
- the torso has nothing in front of its recurrence and nothing carrying
  parameters behind it, so exact credit reaches every parameter it has;
- the eligibility recursion `z ← γλ(1 - reset) z + F ∇` holds per block at each
  block's own decay, and the emphasis recursion `F ← γF(1 - reset) + reset`
  holds per stream.

The oracle is autodiff through `TruncatedBPTT` chained over the prefix, which
carries no derivative state; it reads none of `local_jacobian`,
`compute_phantom` or `initialize_sensitivity`. Two aliases of one implementation
are never compared. A companion test holds the truncated gradient *away* from
the unrolled one over the same sequences, so the exactness comparisons cannot
pass by having nothing to catch.

`tests/test_diagonal_rflo_characterization.py` carries the RFLO argument: RFLO
is RTRL with the cross-unit part of `∂g/∂h` dropped, that part is identically
zero for both cores, so the two recurrences produce the same numbers and exact
RTRL costs no more than the approximation would. A deliberately non-diagonal
`tanh` recurrence, defined in that test and registered nowhere, is used to show
the two recurrences can disagree, and their gap there is held to exactly the
dropped term.

That argument bounds the *declared* cores of this document — the LRU and the
RTU, whose `LRU_DIFFERENTIATION_FAMILY` and `RTU_DIFFERENTIATION_FAMILY` offer
`exact_rtrl` and `tbptt` and nothing else, because on them RFLO would be a
second name for `exact_rtrl` at the same cost. It does not bound every core in
the repository. `memorax/networks/sequence_models/ctrnn.py` carries a CTRNN,
whose unit reads every other unit's previous state through one weight matrix:
the cross-unit block there is not small but present, and `RFLO` is a real
approximation rather than an identity. So `CTRNN_DIFFERENTIATION_FAMILY` offers
`rflo` and `tbptt`, and offers no `exact_rtrl` -- and the `rtrrl_ctrnn_rflo`
entry narrows that to `rflo` alone, since `tbptt` there is the tests' judge and
not a mode a run may pick while keeping the name — exact sensitivity on a dense
recurrence costs a factor of the hidden width, and RFLO is what the published
`RTRRL-CTRNN-RFLO` spends instead. `tests/test_ctrnn_rflo.py` holds that gap to
the dropped term from the other side: there the two recurrences must *not*
agree, and their difference is the same expression this section names.

`memorax/networks/sequence_models/lstm.py` is the second such core and the
second family. Its unit reads every other unit's previous *hidden* state
through the recurrent block of four matrices, so the cross-unit block is again
present; what differs from the CTRNN is where the leak comes from. There the
leak is `1 - dt/tau`, a constant per unit that a declared parameter names.
Here it is the forget gate `sigma(W_f v_t)`, a learned function of the state,
different at every transition — so the rate at which credit is forgotten is
part of what the run is learning rather than part of what it was configured
with. The state the trace is a derivative of is `c` and not the cell's output,
and the output gate carries no trace at all because it does not enter `c`; both
are derived in `docs/rtrrl-lstm-rflo.md`. `LSTM_DIFFERENTIATION_FAMILY`
therefore offers `rflo` and `tbptt` and no `exact_rtrl`, and the
`rtrrl_lstm_rflo` entry narrows that to `rflo` — where the CTRNN entry's reason
was that `tbptt` is its tests' judge, this one has a second: an LSTM torso
differentiated by truncated backpropagation is a thing this repository already
runs under the name `drqn`.

`memorax/networks/sequence_models/dense_ssm.py` is the third, and it is the one
that makes the paragraph above testable rather than only arguable. The claim
this section rests on is about the *structure* -- that the cross-unit block is
zero because the parameterisation makes it zero -- and neither the LRU nor the
RTU can be asked what would happen if it were not, because neither has a block
to remove. The dense state-space core is the same linear step with `A` full, so
the block is a thing that can be switched off: with the off-diagonal zeroed,
`tests/test_dense_ssm_rflo.py` shows RFLO reproducing backpropagation through
the unroll at every length it is asked for, and with the off-diagonal restored
it shows the two parting by exactly `(A - diag(A)) S_{t-1}`. That is this
section's argument, run as a measurement on a registered core rather than on
the unregistered `tanh` recurrence
`tests/test_diagonal_rflo_characterization.py` defines for the purpose.

Which is the same statement, said four times. RFLO exists exactly where the
cross-unit block does, and the families say so by what each offers.

## What the claim covers, and what bounds it

The sensitivity is kept for the recurrent kernel's own parameters — and on
RTRRL's torso those are all of them. The cell reads the observation directly,
nothing precedes it, and the `LayerNorm` behind it is affine-free
(`use_scale=False, use_bias=False`, which is what the published implementation
normalises with) and so holds no parameters to credit. Exact online recurrent
sensitivity is therefore true of the torso as a whole. The test suite asserts
that shape directly, so the statement cannot quietly stop being true if
something is put back in front of the cell.

The bound is worth stating anyway, because it is a property of the method rather
than of today's graph. A learned projection ahead of the cell would reach its own
past only through a carry the phantom injection cuts with `stop_gradient`, so it
would take the one-step gradient under `exact_rtrl` exactly as under `tbptt`, and
the torso's gradient could no longer be called exact. That is checked on a
deliberately projected variant, so "exact" names the recurrent kernel rather than
whatever graph the kernel is sitting in.

## Naming and run metadata

The method is **Structured RTRRL**, or **RTU-like RTRRL (exact RTRL)** where the
declared core is the RTU. Neither is a separate entry: the entry is `rtrrl` and
the core is a parameter of it, so the name a result is reported under is read
off the run rather than off the container.

A run already records the core kind and its dimensions and
`differentiation.kind = exact_rtrl` in its parameters, and the immutable image
digest in `digest`. Three of the four things a formal result has to be traced
back to are therefore in place.

The fourth — a version for the verification this document describes, so that a
result answers to a statement of exactness rather than to the name of a method —
is **not** recorded yet, deliberately. It belongs on `RunMetadata` beside
`digest`, and `RunMetadata` is #46's: that issue is adding `seed` and `role` to
the same frozen dataclass for the same reason, and a second issue widening it in
parallel buys a merge conflict rather than a field. Add it there, as a constant
raised whenever what the verification checks changes.
