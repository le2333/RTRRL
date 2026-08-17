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
| Uniform replay | `make_episode_buffer`, no priority declared | `test_uniform_replay_keeps_no_priority_to_update` |
| Bootstrapped random updates from a random episode position | `any_position_starts` | `test_the_buffer_draws_from_inside_episodes_under_this_rule_and_not_the_other` |
| Zero hidden state at the sampled window start | `Core._loss` opens on `q_function.reset` | `test_a_window_starts_from_a_zero_hidden_state`, `test_both_branches_open_their_window_on_the_same_zero_state` |
| One-step target-network Q-learning | `Core._successor_values` with `make_td0` | `test_the_target_is_one_step_and_greedy_under_the_target_network` |
| Not double Q-learning | `max` over the target network's own values | `test_the_greedy_action_is_the_target_networks_and_not_the_online_ones` |
| Hard target copy on a period | `periodic_incremental_update(..., 1.0)` | `test_the_copy_is_hard_and_not_an_average` |
| Linear Q head | `DiscreteQNetwork`, no head branch declared | `test_the_head_is_one_linear_map_from_the_recurrent_output` |
| Epsilon-greedy acting, annealed | `Core.act`, `DRQN._epsilon` | `test_act_only_advances_recurrence` |
| Truncation 10 for the acceptance arm | `learning.truncated.length`, a manifest value | `test_the_truncation_is_the_window_and_full_bptt_is_the_episode` |

R2D2's additions are absent rather than disabled: there is no priority
exponent, no importance-sampling exponent, no `n_step`, no value transform, no
burn-in and no dueling head anywhere in the parameter tree, so no manifest can
select one and no tuning trial can be spent discovering that it should not.
`test_the_declared_tree_offers_no_r2d2_enhancement` asserts the tree, and
`test_the_graph_holds_no_r2d2_machinery` asserts the built graph.

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

## Where this departs from the paper

Three departures, none of them hidden behind a branch nothing selects.

**RMSProp becomes Adam.** The 2015 learner is written over RMSProp, which this
repository's shared step family does not offer; adding it would widen the
parameter tree of every algorithm that draws from that family. The replay arm
therefore declares `adam` only, as the other replay learner here does, and the
learning rate is tuned. This is a substitution in the optimiser, not in the
learning rule.

**One update per environment step, not one per four frames.** The paper's
cadence is tied to Atari frame skip. R1 compares against an online learner that
updates every step, and equal-budget comparison is what the truncation search
needs, so the replay arm updates whenever the buffer can be sampled. A run
therefore spends more gradient per environment step than the paper's would.

**Truncation-boundary bootstrap.** Episodes are stored end to end in a stream
that resets itself, so the row after an ending holds the next episode's first
observation. Where an episode ended at its step limit rather than by failing,
the target reads the stored successor with the recurrence that reached it,
instead of valuing a state the transition never entered. The paper's Atari
episodes end by failing and the case does not arise there;
`test_a_cut_off_ending_bootstraps_from_the_state_that_was_reached` holds it,
and `test_a_terminal_ending_has_no_successor_at_all` holds the other ending.

## What this does not contain

The truncation sweep and everything that would license a claim from it are not
here, and not because they were forgotten. Judging `t_eq` needs a fixed-episode
evaluation, an AUC over environment steps, and an archived separation of tuning
from formal seeds; all three are #46's, which is open. The equal-budget
bracketed search over `t = {1, 4, 16, 64, full}` selects *which* candidates run
and reads their scores, so it is written once there is a score to read.

Until then this entry supports diagnostic and smoke training only. A number
taken from it is not a paper number.
