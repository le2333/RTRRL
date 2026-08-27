# RFLO on an LSTM: the derivation, and what it drops on purpose

`rtrrl_lstm_rflo` runs RTRRL's flow over an LSTM torso and credits it with a
forward eligibility trace of the RFLO shape. This document states the trace,
derives it, says which terms are dropped and why each one is the term RFLO
drops rather than a term that was inconvenient, and names the test that fails
if a piece of it is changed.

It exists because "RFLO on an LSTM" is not something either source paper
writes. Murray's RFLO is derived for a continuous-time rate network with one
state per unit; the LSTM has two, and the leak Murray's derivation keeps is a
constant where the LSTM's is a learned gate. The mapping between them is a
decision this repository made, and the reasons are below rather than in a
commit message.

## The cell

One packed matrix per gate, columns laid out `[input, hidden, bias]`, which is
`ctrnn.py`'s layout for the same reason: it lets one trace have the shape of
one gate's parameters.

    v_t = [x_t, h_{t-1}, 1]
    i_t = σ(W_i v_t)      f_t = σ(W_f v_t)
    g_t = tanh(W_g v_t)   o_t = σ(W_o v_t)
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

`h_t` is what the sequence hands on. `c_t` is what carries memory.

## The exact forward sensitivity, and the three terms in it

Write `S^c_t = ∂c_t/∂θ` and `S^h_t = ∂h_t/∂θ` for a parameter `θ`. Then, with
no approximation at all,

    S^c_t = f_t ⊙ S^c_{t-1}                       (A) the multiplicative carry
          + (∂c_t/∂h_{t-1}) S^h_{t-1}             (B) the path through the gates
          + ∂c_t/∂θ |_{c,h held}                  (C) this step's own derivative

    S^h_t = (∂h_t/∂h_{t-1})|_{o} S^h_{t-1}
          + ∂h_t/∂θ|_{o, c_t held}
          + o_t ⊙ (1 - tanh²(c_t)) ⊙ S^c_t

with

    ∂c_{t,k}/∂h_{t-1,m} = c_{t-1,k} σ'(a^f_{t,k}) R^f_{km}
                        + g_{t,k}   σ'(a^i_{t,k}) R^i_{km}
                        + i_{t,k}   tanh'(a^g_{t,k}) R^g_{km}

where `R^*` is the recurrent block of `W_*`, columns `features .. features +
hidden_dim`.

Term (B) is the cross-unit block. It is dense — unit `k`'s next cell state
reads every unit's previous hidden state — so carrying it exactly means
carrying `∂c_k/∂θ_{ab}` for every `(k, a, b)`, which is a factor of the hidden
width more memory and the same factor of arithmetic per transition. That is
what exact RTRL costs on this cell, and it is what RFLO is spending instead.

## What RFLO is here

**RFLO keeps (A) and (C) and drops (B).** The trace is a derivative of the
cell state alone, and `h_{t-1}` is read as a constant:

    p^{W_f}_{kj} ← f_{t,k} p^{W_f}_{kj} + c_{t-1,k} σ'(a^f_{t,k}) v_{t,j}
    p^{W_i}_{kj} ← f_{t,k} p^{W_i}_{kj} + g_{t,k}   σ'(a^i_{t,k}) v_{t,j}
    p^{W_g}_{kj} ← f_{t,k} p^{W_g}_{kj} + i_{t,k}   tanh'(a^g_{t,k}) v_{t,j}

and the gradient the algorithm receives for those three is

    dL/dW_{kj} = Σ_t (∂L/∂h_{t,k}) · o_{t,k} (1 - tanh²(c_{t,k})) · p_{t,kj} .

This is the mapping the name claims. RFLO on the CTRNN keeps the leak
`1 - dt/τ` and drops everything that reaches the previous state through the
activation; here the leak is `f_t` and (B) is everything that reaches the
previous state through a gate. The two statements are the same statement, and
the only structural difference is that this cell's leak is itself learned, so
the rate at which credit is forgotten is a function of the input rather than a
declared number.

It is also, term for term, the LSTM eligibility trace of e-prop (Bellec et al.
2020). e-prop derives it from a different starting point — a factorisation of
the true gradient into an eligibility and a learning signal — and arrives at
`f_t` as the carry and the same three immediate terms. A reader coming from
that paper should expect `e` where this says `p` and a per-neuron learning
signal where RTRRL supplies its TD error and emphasis.

## `W_o` has no trace, and that is exact

`∂c_t/∂W_o = 0`: the output gate does not enter the cell state. So under the
approximation above, `W_o`'s entire gradient is

    dL/dW_o,kj = Σ_t (∂L/∂h_{t,k}) · tanh(c_{t,k}) σ'(a^o_{t,k}) v_{t,j} ,

which ordinary autodiff through one step produces without any carried state.
Three traces and not four is therefore a statement about the equations, not a
saving, and not a place where a fourth gate was quietly left uncredited: `W_o`
is a quarter of the recurrence's parameters and it learns.

`test_a_cotangent_on_the_cell_state_is_the_trace_exactly` asserts the zero
where it is a zero — a cotangent on `c` gives `W_o` exactly nothing — and
`test_every_matrix_of_the_cell_learns_including_the_untraced_one` asserts that
the graph nevertheless moves it.

## Reset at an episode boundary

An ending is read before the step it ends. Both halves of the carry return to
zero and all three traces are set to zero, so a restarted stream owes nothing
to the episode before it. Both are per stream. The two halves matter
separately: clearing `c` and not `h` would leave the new episode's first gates
reading the old episode's hidden state, and clearing the carry and not the
trace would credit the new episode's first update with the old one's
sensitivity.

## How the gradient is taken

The same way every other online core in this repository takes it. The
sensitivity is carried as differentiation state, and a phantom — `Σ_θ p_θ (θ -
stop_grad(θ))`, identically zero, with derivative `p` — is added to the cell
state. One step of autodiff then produces the online gradient.

What makes it RFLO rather than RTRL is *where* the phantom is added:

- the gates read `stop_gradient(h_{t-1})`, so no derivative reaches the
  parameters through `σ` or `tanh` — this is (B) being dropped;
- the phantom rides `c_{t-1}` alone, and the step multiplies that by exactly
  `f_t` — this is (A);
- `W_*` are not stopped inside the step, so (C) is autodiff's and appears in
  the carried trace only for the transitions after this one.

The term is dropped in the forward rather than subtracted back out afterwards,
so there is one place to read and one place it can be wrong.

## What is verified, and against what

`tests/test_lstm_rflo.py`:

- **the forward** against a numpy walk written from the equations above;
- **each immediate factor** against `jax.jacrev` of the same step with `c_{t-1}`
  and `h_{t-1}` held — including that the Jacobian is diagonal in the unit
  index, and that `∂c_t/∂W_o` is exactly zero;
- **the three trace recurrences** against the same numpy reference, over a
  sequence containing an ending;
- **the leak is the gate**: with the forget gate drawn shut the final trace is
  the last transition's immediate term alone; with it drawn open it is not;
- **the phantom**: read with a cotangent on `c` the gradient equals the trace
  exactly, and read with one on the output it equals the trace scaled by
  `o_t (1 - tanh²(c_t))`, with `W_o` silent in the first and correct in the
  second;
- **one transition is exact**: from an empty trace RFLO and backpropagation
  through the unroll agree for all four matrices. The carry is *not* empty
  there, because `∂c_t/∂W_f` is proportional to `c_{t-1}` and the comparison
  would otherwise be vacuous for the one gate this method's leak is;
- **three transitions are not**: RFLO and the unroll disagree, and an exact
  RTRL recurrence written in the test — carrying the full `(c, h)` sensitivity
  and term (B) with it — reproduces the unroll. That pair is what separates
  "RFLO is implemented" from "RTRL is implemented" and from "the gradient is
  wrong".

`tests/unit/algorithms/rtrrl/test_lstm_rflo_assembly.py` holds the graph: that
the torso is the four matrices and nothing else, that the differentiation the
entry selects is the one it carries, that all four matrices learn, that an
ending restarts the carry and the trace together, and that the entry cannot
select `tbptt` under a name that promises RFLO.

## Relations to the rest of the repository

`docs/exact-recurrent-sensitivity.md` states the same question for the LRU and
the RTU, where term (B) is identically zero and exact RTRL therefore costs what
RFLO would. `docs/rtrrl-ctrnn-rflo-corrections.md` covers the CTRNN, which is
the other core where (B) is real — and which, unlike this one, has a published
implementation to answer to. There is no parity suite here because there is
nothing to hold one against: the paper's LSTM appears as a DRQN core
differentiated by truncated backpropagation through a replayed window, which is
`drqn` in this repository and is a different method, not a different spelling.
