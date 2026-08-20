# DRQN, and what the reproduction is answerable to

The `drqn` entry is Hausknecht and Stone (arXiv:1507.06527) on this
repository's structured diagonal recurrent core. It exists so that R1 can put
exact online recurrent sensitivity against the same representation trained by
backpropagation through time, which only means something if the replay arm
really is the published learner and the representation really is the same one.

This document says which clause of the paper each part answers, where the
answer is checked, and where the implementation knowingly departs from the
paper.

## The learner, clause by clause

| Paper | Here | Held by |
| --- | --- | --- |
| Uniform replay | `make_uniform_episode_window_buffer`, no priority declared | `test_uniform_replay_keeps_no_priority_to_update` |
| A random update draws completed episodes without replacement | Gumbel top-k over the eligible episodes | `test_an_episode_is_not_drawn_more_often_for_being_longer`, `test_a_minibatch_draws_each_episode_at_most_once` |
| then a point uniformly inside each | `r ~ U{0..L-t}` inside the drawn episode | `test_a_start_is_uniform_over_the_places_a_window_fits` |
| The window unrolls `t` transitions from that point | the episode must be at least `t` long | `test_an_episode_shorter_than_the_truncation_is_never_drawn`, `test_every_drawn_window_carries_the_full_truncation` |
| A minibatch holds `min(episodes, batch_size)` windows | `batch_valid`, and the loss divides by it | `test_the_minibatch_shrinks_to_the_episodes_there_are` |
| The target pass reads the successor sequence from its own zero state | `Core._loss` unrolls the target over `bootstrap_inputs` | `test_the_target_reads_the_successor_sequence_from_its_own_zero_state` |
| Zero hidden state at the sampled window start | `Core._loss` opens on `q_function.reset` | `test_a_window_starts_from_a_zero_hidden_state`, `test_both_branches_open_their_window_on_the_same_zero_state` |
| One-step target-network Q-learning | `Core._successor_values` with `make_td0` | `test_the_target_is_one_step_and_greedy_under_the_target_network` |
| Not double Q-learning | `max` over the target network's own values | `test_the_greedy_action_is_the_target_networks_and_not_the_online_ones` |
| Hard target copy on a period | `periodic_incremental_update(..., 1.0)` | `test_the_copy_is_hard_and_not_an_average` |
| Linear Q head | `DiscreteQNetwork`, no head branch declared | `test_the_head_is_one_linear_map_from_the_recurrent_output` |
| Epsilon-greedy acting, annealed | `Core.act`, `DRQN._epsilon` | `test_act_only_advances_recurrence` |
| Truncation 10 for the acceptance arm | `learning.truncated.length`, a manifest value | `test_the_truncation_is_the_window_and_full_bptt_is_the_episode` |
| ADADELTA, lr 0.1, decay 0.95, gradient clip 10 | `optimizer.adadelta`, `grad_clip` | `test_the_published_solver_is_adadelta_over_a_clipped_gradient` |

R2D2's additions are absent rather than disabled: there is no priority
exponent, no importance-sampling exponent, no `n_step`, no value transform, no
burn-in and no dueling head anywhere in the parameter tree, so no manifest can
select one and no tuning trial can be spent discovering that it should not.
`test_the_declared_tree_offers_no_r2d2_enhancement` asserts the tree, and
`test_the_graph_holds_no_r2d2_machinery` asserts the built graph.

## Why replay has its own buffer

The sampling rows above are one clause of the paper, split up because each half
can be got wrong on its own and each changes what a truncation sweep measures.
Drawing a stored *position* instead of an episode lets a long episode collect
probability in proportion to its length; letting a window run over an ending
leaves it cut short by the validity mask, so a run declaring `t = 64` is in
places training at `t = 5` and `t_eq` becomes a statement about a mixture. A
window here lies inside one completed episode and carries exactly `t`
transitions, or it is not drawn — an episode shorter than `t` contributes
nothing rather than contributing a short window.

An earlier version of this arm expressed that on top of `make_episode_buffer`,
by weighting each admissible position by one over how many its episode offers.
That reproduces the *marginal* — each episode equally likely on a single draw —
and cannot reproduce the rest of the clause, because the rest of the clause is
about a minibatch: "without replacement" is a statement about `B` draws taken
together, and a weight vector describes one draw. It also has nothing to say
when fewer than `B` episodes are eligible, where the published loop shrinks the
minibatch. The layer was wrong, not the arithmetic on top of it.

`make_uniform_episode_window_buffer` separates the two questions. Flashbax
stores the transitions per stream, unchanged; an episode index records where
each completed episode lives, in *logical* time — a counter that only increases,
whose physical slot is `logical % capacity`. Two consequences worth naming,
because each replaces a patch that was there before:

- **The ring seam stops being a special case.** An episode is entirely stored
  exactly when its first transition is (`start >= written - capacity`), so a
  window from inside it cannot splice across the write head. No margin, no
  exclusion zone. `test_no_window_is_spliced_across_the_write_head` and
  `test_an_episode_the_ring_has_overwritten_is_no_longer_drawn` hold it, the
  latter by naming which episodes survive rather than approximately how many.
- **"Enough data" and "something to draw" become the same question.**
  `can_sample` reads the eligible count, so a buffer holding only an unfinished
  episode says so instead of leaving the sampler to invent a window
  (`test_a_buffer_holding_no_finished_episode_says_it_cannot_sample`).

Episode selection is Gumbel top-k: perturb each eligible episode's equal
log-weight by an independent Gumbel and take the largest `B`. That is a uniform
subset without replacement in one fixed-shape operation, and the rows it could
not fill come back marked in `batch_valid` — which is how a shrinking minibatch
survives JIT, and what the loss divides by.

The masks the learner reads are the sampler's, not the window's. Rederiving
validity from the window's own `done` flags would call a window good to its end
whenever it happens to contain no ending, which is true of a padded window and
of a spliced one alike; the claim is about the draw, so it comes from the draw
(`test_the_masks_are_the_sampler_s_and_are_not_rebuilt_from_the_window`).

The buffers this one sits beside are unchanged. DQN's replay unit is a
transition and R2D2's is a fixed-length sequence, so for them the position and
the item are the same thing and a position sampler is the right shape. Two
pre-existing weaknesses in `episode_buffer.py` and the prioritised buffer — a
silent fall back to position zero when no start is admissible, and a full ring
that admits every position including the ones spanning the seam — are real but
are not this arm's to fix; they are filed separately so that fixing them is
judged on DQN's and R2D2's terms rather than smuggled through here.

## The replacement for the convolutional encoder

The paper's network is three convolutional layers over an 84x84 Atari frame,
then an LSTM in place of DQN's first fully-connected layer, then a linear Q
head. R1's environments — MemoryChain, StatelessCartPole, RepeatPrevious — hand
over a short vector, so the encoder has nothing left to do: it existed to turn
an image into a feature vector, and the observation already is one.

**Its replacement is no encoder.** The observation enters the recurrent cell
directly, and an affine-free `LayerNorm` follows the cell. That is not a
convenience: it is the online arm's own topology, and it is the reason the
comparison is a comparison of learners. `exact-recurrent-sensitivity.md` states
the bound this rests on — a learned projection ahead of the cell would reach its
own past only through a carry the phantom injection cuts, so the online arm's
gradient could no longer be called exact, and the two arms would no longer
share a representation. `test_the_core_reads_the_observation_directly_and_
normalises_after_it` holds the shape on this side.

What the two arms share, therefore: the core kind (`lru` or `rtu`), its hidden
size and readout width, the direct input-to-recurrent topology, the
initialisation and dtype the cells declare, and the reset-before-the-step
convention. What they differ in, deliberately: DRQN takes an ordinary
reverse-mode gradient over a replayed window, and Structured RTRRL carries
exact online recurrent sensitivity. The parameter bounds are declared to the
same numbers on both sides so a matched pair can be pinned to identical widths
without either search space excluding a value the other allows.

## The optimiser is the published solver

`recurrent_solver.prototxt` names `ADADELTA` with `base_lr: 0.1`,
`momentum: 0.95` and `clip_gradients: 10`, and that is what the reproduction arm
selects. Caffe's ADADELTA reads its `momentum` field as the decay of the two
running averages the method keeps rather than as a heavy ball, so it is spelled
`rho` here, where nobody will mistake it for the other thing.

An earlier version of this document said the published learner was written over
RMSProp. That was wrong — it was inferred from Nature DQN rather than read off
DRQN's own solver — and both the claim and the Adam-only parameter tree that
followed from it are gone. `adam` remains declared beside `adadelta` for the
**matched** baseline, which tunes its optimiser so that the comparison against
the online arm is not decided by one arm having been given a worse step rule.
The two are different claims and a run says which it is by which branch it
named:

- **DRQN reproduction** selects `adadelta` at the published constants. It
  answers "does the published algorithm behave as published".
- **Matched DRQN** may select either and tunes it. It answers "does replay
  BPTT match exact online sensitivity on the same representation". It is not a
  claim that the optimiser reproduces the paper, and should not be written up
  as one.

## Where this departs from the paper

Two departures, neither hidden behind a branch nothing selects.

**No reward clipping.** The published agent stores `sign(r)` in replay, which on
Atari turns unbounded game scores into `{-1, 0, +1}`. Rewards are stored raw
here. On R1's tasks that is not a difference: MemoryChain pays `±1` at the end
and nothing in between, RepeatPrevious and StatelessCartPole pay in units, so
`sign(r) == r` on every transition they produce and a clipping transform would
be the identity. It stops being the identity the moment an environment with
rewards outside the unit range is added, and at that point this arm is no longer
storing what the published one stores. Anyone adding such an environment to R1
has to add the transform with it.

**Truncation-boundary bootstrap.** Episodes are stored end to end in a stream
that resets itself, so the row after an ending holds the next episode's first
observation. Where an episode ended at its step limit rather than by failing,
the target reads the stored successor with the recurrence that reached it,
instead of valuing a state the transition never entered. The paper's Atari
episodes end by failing and the case does not arise there;
`test_a_cut_off_ending_bootstraps_from_the_state_that_was_reached` holds it,
and `test_a_terminal_ending_has_no_successor_at_all` holds the other ending.

## What is not a departure

**One learner update per environment transition.** An earlier version of this
document listed this as a deviation, on the grounds that the paper updates once
per four frames. That conflated two different things: Atari's frame skip is
action repeat in the environment's preprocessing, not a divisor on the update
rate, and DRQN's own `update_frequency` is 1 — one `UpdateRandom()` per
transition once the replay warmup has passed. One update per environment
transition is therefore what the published agent does, and what this does.

**The recurrent core.** The paper's LSTM is replaced by the structured diagonal
cell the online arm carries exact recurrent sensitivity through. This is the
whole point of the arm rather than an approximation of the paper: R1 compares
two *learners* on one representation, so the representation has to be the one
the other learner can be run on. It is an intentional architecture adaptation,
and a write-up should say "matched DRQN on a structured diagonal core", never
"DRQN as published" when it means the network.

## Known deviations still open

These are differences from the published implementation that are **not** fixed
and that a formal launch has to either resolve or declare. They are listed with
what each one puts at risk, because "a minor deviation" is not a thing that can
be judged without saying what it would change.

**Epsilon anneals on environment steps, not on learner updates.** The published
schedule is a function of the solver iteration and is read once per episode, so
it holds still within an episode and stays at its starting value throughout the
replay warmup, when no update has happened yet. Here it is recomputed every step
against the environment-step count, so exploration already decays during warmup.
Under an equal environment budget the step-based schedule is defensible and is
arguably the fairer one for the matched comparison, but it is not what the
published agent does and the reproduction arm should not claim it is.

**The loss is averaged over the batch; Caffe's is not.** The published net's
Euclidean loss divides by the first dimension of its target blob, which for the
recurrent net is the unroll length, not the minibatch size. So the published
gradient is `batch_size` times this one's. (This arm now divides by the number
of windows actually drawn rather than by the declared `batch_size`, which is the
right denominator for a minibatch that shrinks — but it is a different question
from the factor, and does not settle it.) Under a plain step rule that is a
learning-rate rescaling and nothing more, but the published chain clips the
global gradient norm at ten *before* ADADELTA, and a constant factor of
`batch_size` changes how often that clip binds. It is therefore not simply
absorbed, and an arm calling itself a published-solver reproduction has to
either carry the factor or say it does not.

## What this does not contain

The truncation sweep and everything that would license a claim from it are not
here, and not because they were forgotten. Judging `t_eq` needs a fixed-episode
evaluation, an AUC over environment steps, and an archived separation of tuning
from formal seeds; all three are #46's, which is open. The equal-budget
bracketed search over `t = {1, 4, 16, 64, full}` selects *which* candidates run
and reads their scores, so it is written once there is a score to read.

Until then this entry supports diagnostic and smoke training only. A number
taken from it is not a paper number.
