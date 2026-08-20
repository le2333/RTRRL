"""RTRRL's step, wired as ``RTRRL-AAAI25/rtrrl.py`` wires it.

The arithmetic is copied from ``grads_step`` and ``step_fn``: the vjp taken once
and called twice, one combined ``td_loss`` rather than two, the TD error read
against a value carried from the previous step, the update taken along the trace
*before* this step's gradient enters it, and the trace cleared only after that.
What is not copied is how a forward is invoked -- the networks here are
memorax's, driven through ``Sequence.walk``, so that a mismatch against
``rtrrl_aaai.py`` can only come from the wiring and never from two different
implementations of an LRU.

Notably ``td_loss`` sums the two objectives and differentiates once, where
``rtrrl_aaai`` differentiates each head separately and adds the cotangents. They
agree by linearity; whether they agree in floating point is the point.
"""

import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import optax

from memorax.networks.sequence_models.lru import LRUStructuredRTRL
from memorax.utils import Timestep


def published():
    """``traces.py`` out of an RTRRL-AAAI25 checkout, or a reason there is none.

    Not vendored: that repository is somebody else's and this one does not carry
    it. CI points ``RTRRL_AAAI25`` at a clone; without it the tests that need
    this skip, which is the local case.
    """

    root = os.environ.get("RTRRL_AAAI25")
    if not root:
        raise RuntimeError(
            "set RTRRL_AAAI25 to an RTRRL-AAAI25 checkout to build this reference"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    import traces

    return traces


def make_reference(agent, sequence):
    """A step function over an explicitly supplied copy of the same topology."""

    traces = published()
    compute_updates = traces.compute_updates
    trace_update = traces.trace_update

    cfg = agent.cfg
    torso = agent.core.torso
    actor = agent.core.actor
    critic = agent.core.critic
    differentiation = LRUStructuredRTRL(sequence.core)

    def params_of(core, name):
        return core.torso.params if name == "torso" else getattr(core, name).params

    chain = []
    if cfg.torso_grad_clip:
        chain.append(optax.clip_by_global_norm(cfg.torso_grad_clip))
    chain.extend(
        (
            optax.scale_by_adam(
                b1=cfg.torso_optimizer.b1,
                b2=cfg.torso_optimizer.b2,
                eps=cfg.torso_optimizer.eps,
            ),
            optax.scale(cfg.torso_optimizer.lr),
        )
    )
    transforms = {
        "torso": optax.chain(*chain),
        # The published wiring steps both readouts under one transformation.
        # RTRRL now declares an optimizer per readout, and this file re-hosts
        # the published grouping rather than ours -- so it reads the actor's
        # and holds the two to being the same selection, which is what a
        # comparison against a one-optimizer implementation requires.
        "heads": optax.chain(
            optax.scale_by_adam(
                b1=cfg.actor_optimizer.b1,
                b2=cfg.actor_optimizer.b2,
                eps=cfg.actor_optimizer.eps,
            ),
            optax.scale(cfg.actor_optimizer.lr),
        ),
    }

    def forward(torso_params, timestep, recurrence):
        """The shared block, exactly as ``Torso.apply`` drives it."""

        _, done, _, _ = timestep
        (carry, differentiation_state), hidden = sequence.walk(
            torso_params,
            torso._input(timestep),
            done=done,
            carries=recurrence.carry,
            differentiation_state=recurrence.differentiation_state,
            differentiation=differentiation,
        )
        return hidden, (carry, differentiation_state)

    def grads_step(torso_params, head_params, timestep, recurrence, action_key):
        """``grads_step``: one vjp, called twice; one combined traced objective."""

        hidden, rnn_backwards, rnn_state = jax.vjp(
            lambda p: forward(p, timestep, recurrence), torso_params, has_aux=True
        )

        @partial(jax.grad, has_aux=True, argnums=[0, 1])
        def td_loss(_params, _hidden):
            v_hat = critic.apply(_params["critic"], _hidden, timestep)
            action_dist = actor.apply(_params["actor"], _hidden, timestep)
            # The published line is ``action_dist.sample(seed=action_key)``, and
            # distrax reparametrises, so the gradient there runs through the
            # draw as well as through the density. Cut here to test whether that
            # path is the whole of the disagreement.
            action = jax.lax.stop_gradient(action_dist.sample(seed=action_key))
            actor_loss = action_dist.log_prob(action)
            loss = actor_loss.mean() * cfg.eta_pi + v_hat.mean()
            return loss, (action, v_hat)

        (grads_next, hidden_grads), (action, v_hat) = td_loss(head_params, hidden)
        hidden_grads = rnn_backwards(hidden_grads)[0]

        @partial(jax.grad, has_aux=True, argnums=[0, 1])
        def non_td_loss(_params, _hidden):
            action_dist = actor.apply(_params["actor"], _hidden, timestep)
            ent = action_dist.entropy().mean()
            return ent * cfg.entropy_rate, ent

        (non_td_grad, hidden_non_td_grad), _ = non_td_loss(head_params, hidden)
        hidden_extra_grad = rnn_backwards(hidden_non_td_grad)[0]

        return (
            rnn_state,
            action[0][0],
            v_hat[0],
            {"torso": hidden_grads, **grads_next},
            {"torso": hidden_extra_grad, "actor": non_td_grad["actor"]},
        )

    def step(state, keys):
        """``step_fn``'s order, on the state ``rtrrl_aaai`` carries."""

        action_key, env_key, reset_key = keys
        core = state.core

        # Restart what ended, which is what the published wrapper does inside
        # its own env.step, and advance every stream's own key.
        state = agent._reset(reset_key, state)
        timestep = state.timestep.to_sequence()

        def one(tp, hp, ts, rec, k):
            ts = jax.tree.map(lambda leaf: leaf[None], ts)
            rec = jax.tree.map(lambda leaf: leaf[None], jax.lax.stop_gradient(rec))
            carry, act, value, traced, direct = grads_step(tp, hp, ts, rec, k)
            return jax.tree.map(lambda leaf: leaf[0], carry), act, value, traced, direct

        head_params = {"actor": core.actor.params, "critic": core.critic.params}
        carry, action, value, traced, direct = jax.vmap(
            one, in_axes=(None, None, 0, 0, 0)
        )(
            core.torso.slow_params,
            head_params,
            timestep,
            core.torso.recurrence,
            jax.random.split(action_key, cfg.num_envs),
        )

        # TD-ERROR -----------------------------------------------------------
        v_targ = state.timestep.reward + cfg.gamma * value * (1 - state.terminal)
        d = v_targ - core.value

        # Combine traces with td-error to compute the updates, from the trace as
        # it stood before this step's gradient.
        updates = {
            "actor": compute_updates(core.actor.traces, d=d),
            "critic": compute_updates(core.critic.traces, d=d),
            "torso": compute_updates(core.torso.traces, d=d * cfg.eta_f),
        }
        updates["actor"] = jax.tree.map(
            lambda a, b: a + b, updates["actor"], direct["actor"]
        )
        updates["torso"] = jax.tree.map(
            lambda a, b: a + b, updates["torso"], direct["torso"]
        )
        updates = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)

        # The published chain, rebuilt here rather than reached for through the
        # kernel's ``make_rules``: an optimiser under test is not a reference.
        # The published chain is the *group's* -- one transformation over both
        # readouts -- and the state it is carried in is the kernel's, which is
        # now a rule per block. Applying the same chain to each block on its
        # own is the same arithmetic: Adam's moments are per leaf, its count
        # steps once per call either way, and the only cross-block operation in
        # either chain is the torso's global-norm clip, whose group is one
        # block. So the grouping under comparison is unchanged and only the
        # bookkeeping follows the state.
        taken = {}
        stepped = {}
        for group, names in (("torso", ("torso",)), ("heads", ("actor", "critic"))):
            for name in names:
                tree = {name: updates[name]}
                held = {name: params_of(core, name)}
                moved, opt = transforms[group].update(tree, state.core.rule[name], held)
                taken[name] = opt
                stepped[name] = optax.apply_updates(held[name], moved[name])

        # I <- gamma I (1 - done) + done
        done = state.timestep.done
        emphasis = (cfg.gamma * core.emphasis * (1 - done) + done).astype(jnp.float32)

        # Reset trace if done, then update it.
        advanced = {}
        for name, decay in (
            ("torso", cfg.gamma * cfg.lambda_rnn),
            ("actor", cfg.gamma * cfg.lambda_pi),
            ("critic", cfg.gamma * cfg.lambda_v),
        ):
            held = core.torso.traces if name == "torso" else getattr(core, name).traces
            cleared = jax.tree.map(
                lambda old: jnp.where(
                    done[(slice(None),) + (None,) * (old.ndim - 1)],
                    jnp.zeros_like(old),
                    old,
                ),
                held,
            )
            advanced[name] = jax.vmap(
                lambda z, g, i, gl=decay: trace_update(g, z, gamma_lambda=gl, _I=i)
            )(cleared, traced[name], emphasis)

        slow = torso.followed(stepped["torso"], core.torso.slow_params)
        core = core.replace(
            torso=core.torso.replace(
                params=stepped["torso"],
                traces=advanced["torso"],
                slow_params=slow,
                recurrence=type(core.torso.recurrence)(
                    carry=carry[0], differentiation_state=carry[1]
                ),
            ),
            actor=core.actor.replace(params=stepped["actor"], traces=advanced["actor"]),
            critic=core.critic.replace(
                params=stepped["critic"], traces=advanced["critic"]
            ),
            rule=taken,
            value=value,
            emphasis=emphasis,
        )

        obs, env_state, reward, done_next, terminal, info = agent.environment.step(
            env_key, state.env_state, action
        )
        obs, reward, scales = agent.normalization.apply(
            state.scales, obs, reward, done_next
        )
        return state.replace(
            step=state.step + cfg.num_envs,
            update_step=state.update_step + 1,
            timestep=Timestep(obs=obs, action=action, reward=reward, done=done_next),
            terminal=terminal,
            env_state=env_state,
            scales=scales,
            core=core,
        )

    return step
