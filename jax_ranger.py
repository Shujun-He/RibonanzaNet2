"""JAX implementation of Ranger

Designed to match:
https://github.com/mpariente/Ranger-Deep-Learning-Optimizer/blob/master/pytorch_ranger/ranger.py
"""

import optax
import jax.tree


def ranger(
    lr: optax.ScalarOrSchedule = 1e-3,
    alpha: float = 0.5,
    k: int = 6,
    N_sma_threshold: float = 5,
    betas: tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-5,
    weight_decay: float = 0,
):
    return nnx_compatible_lookahead(
        fast_optimizer=optax.chain(
            optax.scale_by_radam(
                b1=betas[0],
                b2=betas[1],
                eps=eps,
                threshold=N_sma_threshold,
            ),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(lr),
        ),
        sync_period=k,
        slow_step_size=alpha,
        reset_state=True,
    )


def nnx_compatible_lookahead(
    *,
    fast_optimizer,
    sync_period: int,
    slow_step_size: float,
    reset_state: bool,
    **kwargs,
) -> optax.GradientTransformation:
    base = optax.lookahead(
        fast_optimizer=fast_optimizer,
        sync_period=sync_period,
        slow_step_size=slow_step_size,
        reset_state=reset_state,
        **kwargs,
    )
    base_init_fn = base.init
    base_update_fn = base.update

    def init_fn(params) -> optax.LookaheadState:
        return base_init_fn(optax.LookaheadParams(fast=params.fast, slow=params.slow))

    def update_fn(updates, state, params):
        updates, new_opt_state = base_update_fn(
            updates,
            state,
            params,
        )
        if not isinstance(params, optax.LookaheadParams):
            # `updates` has type `optax.LookaheadParams`, but when
            # using flax.nnx.Optimizer, `params` will have type
            # `flax.nnx.State`.  The subsequent `optax.apply_updates`
            # call requires that the updates match the structure of
            # `params`.
            updates = jax.tree.unflatten(
                jax.tree.structure(params), jax.tree.leaves(updates)
            )
        return updates, new_opt_state

    return optax.GradientTransformation(init_fn, update_fn)
