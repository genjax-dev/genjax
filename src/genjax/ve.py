from dataclasses import dataclass
from functools import wraps

import jax.tree_util as jtu
from jax import vmap
from jax.extend.core import Jaxpr
from jax.util import safe_map

from .core import (
    Any,
    Callable,
    ElaboratedPrimitive,
    Environment,
    Pytree,
    assume_p,
    modular_vmap,
    observe_p,
    stage,
)


@Pytree.dataclass
class EnumeratedValue(Pytree):
    v: Any

    @classmethod
    def unwrap(cls, v):
        if isinstance(v, EnumeratedValue):
            return v.v
        else:
            return v


@dataclass
class VariableEliminationInterpreter:
    def eval_jaxpr_ve(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, args)
        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)

            if primitive == assume_p:
                support = inner_params["support"](*args)
                outvals = [EnumeratedValue(support)]

            elif primitive == observe_p and any(
                isinstance(arg, EnumeratedValue) for arg in args
            ):
                in_axes = tuple(
                    0 if isinstance(arg, EnumeratedValue) else None for arg in args
                )
                unwrapped_args = tuple(map(EnumeratedValue.unwrap, args))
                initial_outvals = modular_vmap(
                    lambda *args: ElaboratedPrimitive.rebind(
                        primitive,
                        *args,
                        **params,
                        **inner_params,
                    ),
                    in_axes=in_axes,
                )(*unwrapped_args)
                outvals = (
                    EnumeratedValue(initial_outvals)
                    if not eqn.primitive.multiple_results
                    else list(map(EnumeratedValue, initial_outvals))
                )
            else:
                if any(isinstance(arg, EnumeratedValue) for arg in args):
                    in_axes = tuple(
                        0 if isinstance(arg, EnumeratedValue) else None for arg in args
                    )
                    unwrapped_args = tuple(map(EnumeratedValue.unwrap, args))
                    initial_outvals = vmap(
                        lambda *args: ElaboratedPrimitive.rebind(
                            eqn.primitive,
                            *args,
                            **params,
                            **inner_params,
                        ),
                        in_axes=in_axes,
                    )(*unwrapped_args)
                    outvals = (
                        EnumeratedValue(initial_outvals)
                        if not eqn.primitive.multiple_results
                        else list(map(EnumeratedValue, initial_outvals))
                    )
                else:
                    outvals = eqn.primitive.bind(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(EnumeratedValue.unwrap, safe_map(env.read, jaxpr.outvars))

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_ve(jaxpr, consts, flat_args)
        return jtu.tree_unflatten(out_tree(), flat_out)


def ve(f: Callable[..., Any]):
    @wraps(f)
    def wrapped(*args):
        interpreter = VariableEliminationInterpreter()
        return interpreter.run_interpreter(f, *args)

    return wrapped
