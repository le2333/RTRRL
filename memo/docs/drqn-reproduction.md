# DRQN, and what the reproduction is answerable to

The `drqn` entry is Hausknecht and Stone (arXiv:1507.06527) on this
repository's structured diagonal recurrent core: the published learner, on a
recurrent cell this repository's online learner can also be run on.

This document says which clause of the paper each part answers, where the
answer is checked, and where the implementation knowingly departs from the
paper. What anyone does with a set of runs is not here.

## The learner, clause by clause

| Paper | Here | Held by |
| --- | --- | --- |
| Uniform replay | `make_uniform_episode_window_buffer`, no priority declared | `test_uniform_replay_keeps_no_priority_to_update` |
| Caffe's Euclidean loss over the unroll length | `published_loss` sums the batch, averages time | `test_the_loss_sums_over_the_batch_and_averages_over_time` |
| Rewards clipped to their sign | `clipped_reward`, on the way into replay only | `test_replay_stores_the_clipped_reward_and_the_metric_keeps_the_raw_one` |
| A random update draws completed episodes without replacement | Gumbel top-k over the eligible episodes | `test_an_episode_is_not_drawn_more_often_for_being_longer`, `test_a_minibatch_draws_each_episode_at_most_once` |
| then a point uniformly inside each | `r ~ U{0..L-t}` inside the drawn episode | `test_a_start_is_uniform_over_the_places_a_window_fits` |
| The window unrolls `t` transitions from that point | the episode must be at least `t` long | `test_an_episode_shorter_than_the_truncation_is_never_drawn`, `test_every_drawn_window_carries_the_full_truncation` |
| A minibatch holds `min(episodes, batch_size)` windows | `batch_valid`, and the loss divides by it | `test_the_minibatch_shrinks_to_the_episodes_there_are` |
| One `UpdateRandom()` per environment transition | `num_envs` other than one is refused | `test_more_than_one_environment_is_refused_rather_than_given_a_cadence` |
| The target pass reads the successor sequence from its own zero state | `Core._loss` unrolls the target over `bootstrap_inputs` | `test_the_target_reads_the_successor_sequence_from_its_own_zero_state` |
| Zero hidden state at the sampled window start | `Core._loss` opens on `q_function.reset` | `test_a_window_starts_from_a_zero_hidden_state`, `test_both_branches_open_their_window_on_the_same_zero_state` |
| One-step target-network Q-learning | `Core._loss` with `make_td0` | `test_the_target_is_one_step_and_greedy_under_the_target_network` |
| Not double Q-learning | `max` over the target network's own values | `test_the_greedy_action_is_the_target_networks_and_not_the_online_ones` |
| Hard target copy on a period | `periodic_incremental_update(..., 1.0)` | `test_the_copy_is_hard_and_not_an_average` |
| Linear Q head | `DiscreteQNetwork`, no head branch declared | `test_the_head_is_one_linear_map_from_the_recurrent_output` |
| Epsilon-greedy acting, annealed over solver iterations | `DRQN._epsilon` counts learner updates | `test_epsilon_anneals_on_learner_updates_and_not_on_environment_steps` |
| The rate is read once per episode | `DRQN._episode_epsilon`, held in `DRQNState.epsilon` | `test_exploration_holds_still_inside_an_episode` |
| Truncation 10 for the acceptance arm | `learning.truncated.length`, a manifest value | `test_the_truncation_is_the_window_and_full_bptt_is_the_episode` |
| Replay stores whole episodes | committed at the ending; the update reads replay as of before this transition | `test_an_update_cannot_draw_the_episode_it_is_still_finishing` |
| An episode being played changes nothing in replay | the ring reserves `max_episode_length` | `test_an_episode_being_played_evicts_nothing_that_could_be_drawn` |
| ADADELTA, lr 0.1, decay 0.95, gradient clip 10 | `optimizer.adadelta`, `grad_clip` | `test_the_published_solver_is_adadelta_over_a_clipped_gradient` |

R2D2's additions are absent rather than disabled: there is no priority
exponent, no importance-sampling exponent, no `n_step`, no value transform, no
burn-in and no dueling head anywhere in the parameter tree, so no manifest can
select one and no tuning trial can be spent discovering that it should not.
`test_the_declared_tree_offers_no_r2d2_enhancement` asserts the tree, and
`test_the_graph_holds_no_r2d2_machinery` asserts the built graph.

## Why replay has its own buffer

The sampling rows above are one clause of the paper, split up because each half
can be got wrong on its own. Drawing a stored *position* instead of an episode
lets a long episode collect probability in proportion to its length; letting a
window run over an ending leaves it cut short by the validity mask, so a
learner declaring TBPTT(64) is in places performing TBPTT(5). A window here
lies inside one completed episode and carries exactly `t` transitions, or it is
not drawn — an episode shorter than `t` contributes nothing rather than
contributing a short window.

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

Flashbax is storage here and nothing else; its own `can_sample` is not called.
It answers for its own trajectory sampler and will not report ready until a
whole `sample_sequence_length` has been written, which is the right contract
for drawing a fixed-length slice off the head and the wrong one for drawing an
episode. Left in, the warmup would be `max(min_length, t)` — a full-BPTT run
would start learning a whole horizon later than a `t = 4` one at the same
declared replay settings, the window length reaching out of the sampler and
into the schedule. A replay warmup is how much experience has been collected
before learning starts, so readiness here counts transitions and nothing else
(`test_the_warmup_is_the_declared_minimum_and_not_the_window_length`).

Replay is episode-atomic, and the half of that which is easy to miss is
eviction rather than visibility. Transitions of an episode still being played
are in the ring but no record describes them, so nothing can draw them and the
warmup does not count them — that much is the index doing its job. But their
writes still advance the ring's head, and left alone they would push the oldest
finished episodes out one at a time as the episode went on, so *which* episodes
an update could draw would depend on how far into the current episode it was.
That is not a difference in how much storage there is; it is a replay
distribution that moves under the learner, where an agent that commits whole
episodes has a still one.

So the ring is allocated a `max_episode_length` slack **on top of** the
capacity it was asked for, and presence is measured from each stream's last
commit rather than from its write head:

    start >= open_start[stream] - capacity

The threshold moves only when an episode commits, and it is strictly inside
physical presence because an open episode runs at most `max_episode_length`
past `open_start`. `replay.capacity` therefore means what it says — the
transitions of finished episodes replay keeps — and the slack is the buffer's
own cost, paid out of its own allocation rather than deducted from the caller's
number.

The bound is **strict**, which is the published `RememberEpisode`: it pushes the
new episode and then pops while `size >= capacity`, so what replay holds after a
commit is always *fewer* than `capacity` transitions. On a capacity of eight
with episodes of four, the second episode brings the total to exactly eight and
the first is dropped, leaving four — where a non-strict bound would have kept
both.

Readiness is measured the same way: against what replay is *currently holding*,
not against everything it has ever held. A lifetime count only rises, so it
would go on reporting ready against a buffer that had since evicted most of
what it counted. The comparison is strict too — `memory_size() > memory_threshold`
in the published loop — so a buffer sitting on the threshold is still warming
up. An episode straddling the oldest kept position is dropped whole, so what
replay holds sits at or above `capacity - max_episode_length` and never reaches
`capacity`; a warmup this buffer would not stay above is refused where it is
declared, because a threshold met and then unmet stops a run learning without
saying so.

Padding is zeroed rather than left as whatever the ring held. A step past the
end of a short episode does not enter the loss, but it is still unrolled
through the recurrent cell, so it is not nothing: unzeroed it would be the
*next* episode's transitions, or a slot never written at all
(`test_the_padding_past_an_episode_is_zeros_and_not_the_next_episode`).

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

## `full_bptt` is not the paper's sequential arm

The paper offers two ways of drawing and unrolling a minibatch, and this
implements one of them. `truncated` is *bootstrapped random updates*, which is
the published update and what the reproduction arm runs.

`full_bptt` is **not** *bootstrapped sequential updates*. The paper's
sequential scheme runs an episode as a succession of fixed-length unrolls,
carrying the hidden state from one to the next and taking an optimizer step at
each — truncated backpropagation with a stored state, nearer to R2D2's scheme
than to this one. `full_bptt` draws a completed episode and differentiates the
whole of it in a single step, with no state carried in and no boundary for the
gradient to stop at.

It is here because "the gradient crosses the entire episode" is the limit a
truncation is a truncation *of*, and the paper's sequential scheme, being
itself truncated, is not that limit. It is a deliberate addition, not a second
thing the paper published, and a write-up should call it full BPTT and never
DRQN's sequential arm.

## The replacement for the convolutional encoder

The paper's network is three convolutional layers over an 84x84 Atari frame,
then an LSTM in place of DQN's first fully-connected layer, then a linear Q
head. A low-dimensional environment hands over a short vector, so the encoder
has nothing left to do: it existed to turn an image into a feature vector, and
the observation already is one.

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
named. `adadelta` at the published constants is the published solver; anything
else, tuned or not, is this learner under a step rule of someone's choosing and
should not be written up as reproducing the paper's optimiser.

## Where this departs from the paper

One departure, not hidden behind a branch nothing selects.

**Truncation-boundary bootstrap.** Episodes are stored end to end in a stream
that resets itself, so the row after an ending holds the next episode's first
observation. Where an episode ended at its step limit rather than by failing,
the target reads the stored successor with the recurrence that reached it,
instead of valuing a state the transition never entered. The paper's Atari
episodes end by failing and the case does not arise there;
`test_a_cut_off_ending_bootstraps_from_the_state_that_was_reached` holds it,
and `test_a_terminal_ending_has_no_successor_at_all` holds the other ending.

## What is not a departure

**Reward clipping.** `clipped_reward` stores `sign(r)`, which is DQN's own
preprocessing and therefore the units the published agent's Q values are in. It
is applied on the way into replay and nowhere else: a run is scored on what the
environment paid, and clipping that would change the number being reported
rather than the number being learned from. It is not a parameter, because it is
not a choice the published agent offers. On a task paying in units or in
`±1` it is the identity; a task whose reward magnitudes carry information
beyond their sign is not one this learner can be run on as published.

**One learner update per environment transition.** An earlier version of this
document listed this as a deviation, on the grounds that the paper updates once
per four frames. That conflated two different things: Atari's frame skip is
action repeat in the environment's preprocessing, not a divisor on the update
rate, and DRQN's own `update_frequency` is 1 — one `UpdateRandom()` per
transition once the replay warmup has passed. One update per environment
transition is therefore what the published agent does, and what this does.

**The recurrent core.** The paper's LSTM is replaced by the structured diagonal
cell this repository's online learner carries exact recurrent sensitivity
through, so that the two can be run on one representation. It is an intentional
architecture adaptation rather than an approximation of the paper, and a
write-up should say "DRQN on a structured diagonal core", never "DRQN as
published" when it means the network.

## The one thing the batch size still decides

`published_loss` is Caffe's Euclidean loss over the recurrent net's target
blob: half the summed squared error divided by the **unroll length**, not by
the minibatch size. An ordinary mean over both axes would divide by the batch
size as well, and that factor is not absorbed into the learning rate, because
the published chain clips the global gradient norm at ten *before* ADADELTA —
scaling the gradient changes when the clip binds.

What this reproduces is the objective. Reproducing the published *gradient
scale* additionally needs the published `replay.batch_size` of 32, since the
factor the divisor leaves in is the number of windows in the batch. That is a
manifest value, so an arm at a different batch size is reproducing the
objective at a different step size, and a write-up comparing gradient
magnitudes to the paper's has to say which batch size it ran at.

One consequence worth stating because it is not obvious: this arm begins
updating as soon as **one** eligible episode exists, where the published
agent's 50,000-transition warmup guarantees a full batch before its first
update. So the factor is `min(eligible, batch_size)` here and a constant 32
there, until replay holds enough episodes. Requiring a full batch before the
first update would close that too, at the cost of a later learning start; it is
not done, because when learning may start is a manifest's business
(`replay.minimum_size`) and this is the layer that says what an update
computes.

## What this does not contain

This entry is the learner and nothing else. Which runs to launch, what counts
as one performing the same as another, and what number to report are decisions
about an experiment; they are not made here and this package holds none of
their vocabulary.

One thing does have to be said, because it is about the entry: **no image
carries `drqn` yet.** The entry is new, so a contract-10 rebuild has to happen
before anything can be launched, which is why the acceptance manifest names
`image: TBD`. Until then this entry supports diagnostic and smoke training
only.
