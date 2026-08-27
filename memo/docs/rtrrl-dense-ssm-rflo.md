# RFLO on a dense state-space core: the control for the diagonal argument

`rtrrl_ssm_rflo` runs RTRRL's flow over a linear state-space torso with a full
`A` matrix and credits it with an RFLO trace. Unlike the CTRNN and LSTM arms it
is not there to be a better torso. It is there because
`docs/exact-recurrent-sensitivity.md` makes an argument about *structure* that
nothing in this repository was able to test, and this is the cell on which it
can be tested.

## The argument this is a control for

That document says: on the LRU and the RTU, RFLO and exact RTRL are the same
recurrence, because the cross-unit block of `∂h_t/∂h_{t-1}` is identically
zero. Unit `h`'s next state is a function of unit `h`'s own previous state and
of the input, so there is nothing for RFLO to drop, and exact RTRL is available
at RFLO's cost rather than as an upgrade over it.

The argument is correct and it is made from the parameterisation. What it could
not do is *vary* the thing it depends on: the LRU has no off-diagonal block to
switch off, so the claim "RFLO would be identical here" could be argued but not
measured. The existing characterisation test reaches for a deliberately
non-diagonal `tanh` recurrence defined inside the test file and registered
nowhere, which shows the two recurrences *can* disagree but is not a core
anything runs.

This is that core, registered and runnable:

    v_t = [x_t, 1]
    h_t = A h_{t-1} + B v_t
    y_t = tanh(C h_t)

`A` full. Set its off-diagonal to zero and it is the LRU's structure with a
real rather than complex parameterisation; leave it dense and the block is
there.

## The two recurrences

    exact:  S_t = A S_{t-1} + ∂h_t/∂θ
    RFLO:   S_t = diag(A) ⊙ S_{t-1} + ∂h_t/∂θ

with

    ∂h_{t,k}/∂A_{kj} = h_{t-1,j}        ∂h_{t,k}/∂B_{kj} = v_{t,j}

and zero off the `k`-th row, which is what lets RFLO carry one row per unit —
`(hidden, hidden)` for `A` and `(hidden, features + 1)` for `B` — where exact
RTRL carries `∂h_k/∂θ_{ab}` for every triple. That is the factor of the hidden
width, in memory and in arithmetic per transition.

This is the cleanest statement of RFLO available in this repository, because
the recurrence is linear: the gap between the two recurrences is one matrix
product by the off-diagonal of `A`, with no activation Jacobian in the way to
share the blame with. On the CTRNN the dropped term is `(dt/τ) φ'(u) R` applied
to the previous sensitivity, and on the LSTM it is a sum of three gate slopes
times three recurrent blocks; here it is `(A - diag(A)) S_{t-1}` and nothing
else.

## `C` is exact, at every length

`∂h_t/∂C = 0`, so `C`'s whole gradient is the instantaneous one through
`y_t = tanh(C h_t)`. It is not approximated at all — not "approximated and
happens to agree on the first transition", which is the situation `A` and `B`
are in. `test_the_readout_matrix_is_exact_at_every_length` asserts that at one,
three and six transitions, and the assembly suite asserts separately that it
nevertheless learns: a graph that quietly routed `C` through the approximation
would still train, and a third of the torso being credited wrongly is not
something a finite metric would show.

The same shape of statement holds for the LSTM's output gate; see
`docs/rtrrl-lstm-rflo.md`.

## `A` has a domain, and a gradient step can leave it

A linear recurrence with spectral radius above one diverges over an episode.
The LRU cannot reach that domain — its parameterisation is
`λ = exp(-exp(ν) + i exp(θ))`, so `|λ| < 1` holds for every real `ν` — and a
free matrix reaches it on an ordinary step.

The cell states the set as a bound on the induced infinity norm:

    max_i Σ_j |A_ij| ≤ spectral_bound

which bounds the spectral radius, is a projection rather than a search, and
costs one row-wise sum. `contract` scales only the rows that are outside, so a
run whose `A` never reaches the boundary is a run the projection never touches.

Two consequences worth stating because they are choices and not facts:

- **The bound is also the memory.** A contraction at `ρ ≤ b` forgets on a scale
  of about `1/(1 - b)` transitions — roughly ten at the launch document's
  `0.9`. This arm cannot hold a dependency longer than that, and if it fails
  where the CTRNN and LSTM arms succeed, the bound is the first thing to move.
- **The initial draw is projected too.** `lecun_normal`'s row sums grow like
  the square root of the width — about `4.5` at `hidden_dim = 32` — so an
  unprojected `A` would be a diverging recurrence at initialisation rather
  than after some number of updates.

The projection is applied by `Torso` through `kernel_constraint`, which is
`rtrrl_aaai`'s. It was private to `rtrrl_ctrnn_rflo` while `tau`'s floor was
its only implementation; this is the second set, and one mechanism with two
implementations is where a shared helper earns its place. The LSTM torso names
no set and gets no projection, which is what makes this the kernel's statement
rather than a step every core has to implement.

## What is verified, and against what

`tests/test_dense_ssm_rflo.py`:

- **the forward and both traces** against a numpy walk written from the
  equations above, over a sequence containing an ending;
- **the diagonal limit**: with the off-diagonal of `A` zeroed, RFLO and
  backpropagation through the unroll agree at lengths 1, 2 and 5. This is the
  claim `exact-recurrent-sensitivity.md` makes about the LRU, run as a
  measurement on a core where the block can actually be removed;
- **the dense case**: with the block restored they disagree at three
  transitions, and an exact RTRL recurrence written in the test — differing
  from the reference in one line, `A @ S` where RFLO has `diag(A) * S` —
  reproduces the unroll;
- **one transition** is exact for all three matrices even dense, from a
  non-empty carry (from an empty one `∂h/∂A ∝ h_{t-1}` is zero and the
  agreement on `A` would be vacuous);
- **`C`** is exact at one, three and six transitions;
- **the phantom**: with a cotangent on the state, the gradient equals the trace
  exactly and `C` receives precisely zero;
- **the ball**: the initial draw is inside it, the projection scales the rows
  that are outside and leaves the others untouched, and a matrix six times too
  large produces a state the same walk inside the ball does not.

`tests/unit/algorithms/rtrrl/test_ssm_rflo_assembly.py` holds the graph: the
three matrices and nothing else, the trace the torso carries being the cell's,
all three matrices learning, both parameter copies staying inside the ball
under a learning rate large enough to leave it, and the entry refusing `tbptt`
under a name that promises RFLO.

## Reading this arm against the others

Four RTRRL torsos now exist and they answer different questions:

| entry | recurrence | cross-unit block | gradient |
| --- | --- | --- | --- |
| `rtrrl` (LRU, RTU) | diagonal, linear or `tanh` | zero | exact RTRL |
| `rtrrl_ssm_rflo` | **dense, linear** | present | RFLO |
| `rtrrl_ctrnn_rflo` | dense, `tanh`, constant leak | present | RFLO |
| `rtrrl_lstm_rflo` | dense, gated, learned leak | present | RFLO |

The first two rows differ in one thing, and that is the comparison this arm
exists for. The last three differ in what the leak is, which is the comparison
the RFLO arms make among themselves.
