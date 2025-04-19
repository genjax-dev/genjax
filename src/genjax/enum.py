from dataclasses import dataclass
from functools import wraps

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import vmap
from jax.extend.core import Jaxpr
from jax.util import safe_map

from genjax.core import (
    Any,
    Callable,
    ElaboratedPrimitive,
    EnumDSL,
    Environment,
    InitialStylePrimitive,
    R,
    Score,
    X,
    assume_p,
    initial_style_bind,
    modular_vmap,
    stage,
    static_dim_length,
    style,
)

observe_p = InitialStylePrimitive(
    f"{style.BOLD}{style.GREEN}enum.observe{style.RESET}",
)


def observe_binder(
    log_density_impl: Callable[..., Any],
    name: str | None = None,
):
    def observe(*args, **kwargs):
        # TODO: really not sure if this is right if you
        # nest vmaps...
        def batch(vector_args, batch_axes, **params):
            n = static_dim_length(batch_axes, tuple(vector_args))
            num_consts = params["num_consts"]
            in_tree = jtu.tree_unflatten(params["in_tree"], vector_args[num_consts:])
            batch_tree = jtu.tree_unflatten(params["in_tree"], batch_axes[num_consts:])
            if params["yes_kwargs"]:
                args = in_tree[0]
                kwargs = in_tree[1]
                v = observe_binder(
                    vmap(
                        lambda args, kwargs: log_density_impl(*args, **kwargs),
                        in_axes=batch_tree,
                    ),
                    name=name,
                )(args, kwargs)
            else:
                v = observe_binder(
                    vmap(
                        log_density_impl,
                        in_axes=batch_tree,
                    ),
                    name=name,
                )(*in_tree)
            outvals = (v,)
            out_axes = (0 if n else None,)
            return outvals, out_axes

        return initial_style_bind(
            observe_p,
            batch=batch,
        )(log_density_impl, dist=name)(*args, **kwargs)

    return observe


@dataclass
class CollectingInterpreter:
    score: Score

    def eval_jaxpr_collect(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        safe_map(env.write, jaxpr.invars, args)
        safe_map(env.write, jaxpr.constvars, consts)

        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)

            if primitive == observe_p:
                outvals = ElaboratedPrimitive.rebind(
                    eqn.primitive, inner_params, params, *args
                )
                self.score += outvals[0]

            else:
                outvals = ElaboratedPrimitive.rebind(
                    eqn.primitive, inner_params, params, *args
                )

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        outvals = self.eval_jaxpr_collect(jaxpr, consts, flat_args)
        return self.score, jtu.tree_unflatten(out_tree(), outvals)


def collect(
    f: EnumDSL[tuple[X, R]],
) -> EnumDSL[tuple[Score, tuple[X, R]]]:
    @wraps(f)
    def wrapped(*args) -> tuple[Score, tuple[X, R]]:
        interpreter = CollectingInterpreter(jnp.array(0.0))
        score, retval = interpreter.run_interpreter(f, *args)
        return score, retval

    return wrapped


@dataclass
class EnumerationInterpreter:
    @staticmethod
    def eval_jaxpr_enum(
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()

        safe_map(env.write, jaxpr.constvars, consts)

        def eval_jaxpr_enum_loop(eqns, env, invars, args):
            env = env.copy()
            safe_map(env.write, invars, args)

            for idx, eqn in enumerate(eqns):
                invals = safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + invals
                primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)

                if primitive == assume_p:

                    def continuation(v):
                        return eval_jaxpr_enum_loop(
                            eqns[idx + 1 :], env, eqn.outvars, [v]
                        )

                    # Instantiate support.
                    num_consts = inner_params["num_consts"]
                    tree_args = jtu.tree_unflatten(
                        inner_params["in_tree"], args[num_consts:]
                    )
                    support_fn = inner_params["support"]
                    assert support_fn, (
                        f"{params['name']} doesn't have a support function.",
                        inner_params,
                    )
                    if inner_params["yes_kwargs"]:
                        support = inner_params["support"](*tree_args[0], **tree_args[1])
                    else:
                        support = inner_params["support"](*tree_args)

                    # Now, vmap the continuation.
                    return modular_vmap(continuation)(support)

                else:
                    outvals = ElaboratedPrimitive.rebind(
                        eqn.primitive, inner_params, params, *args
                    )

                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                safe_map(env.write, eqn.outvars, outvals)

            return safe_map(env.read, jaxpr.outvars)

        return eval_jaxpr_enum_loop(jaxpr.eqns, env, jaxpr.invars, args)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_enum(jaxpr, consts, flat_args)
        return jtu.tree_unflatten(out_tree(), flat_out)


def enum(
    f: EnumDSL[tuple[X, R]],
) -> Callable[..., tuple[Score, tuple[X, R]]]:
    @wraps(f)
    def wrapped(*args) -> tuple[Score, tuple[X, R]]:
        interpreter = EnumerationInterpreter()
        collected: EnumDSL[tuple[Score, tuple[X, R]]] = collect(f)
        return interpreter.run_interpreter(collected, *args)

    return wrapped
