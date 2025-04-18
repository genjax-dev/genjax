from dataclasses import dataclass
from functools import wraps

import jax.tree_util as jtu
from jax.extend.core import Jaxpr
from jax.util import safe_map

from .core import (
    Any,
    ElaboratedPrimitive,
    Environment,
    ReflectedMeasureProgram,
    assume_p,
    modular_vmap,
    stage,
)


@dataclass
class EnumerationInterpreter:
    @classmethod
    def eval_jaxpr_enum(
        cls,
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
                    tree_args = jtu.tree_unflatten(inner_params["in_tree"], args)
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


def enum(f: ReflectedMeasureProgram[Any]):
    @wraps(f)
    def wrapped(*args):
        interpreter = EnumerationInterpreter()
        return interpreter.run_interpreter(f, *args)

    return wrapped
