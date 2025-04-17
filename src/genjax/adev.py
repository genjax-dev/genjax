import itertools as it
from abc import abstractmethod
from functools import wraps

import jax
import jax._src.core
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype.typing import Annotated, Any, Callable
from beartype.vale import Is
from jax import util as jax_util
from jax.extend import source_info_util as src_util
from jax.extend.core import Jaxpr, Var, jaxpr_as_fun
from jax.interpreters import ad as jax_autodiff
from jaxtyping import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from .core import (
    Distribution,
    ElaboratedPrimitive,
    Environment,
    Pytree,
    initial_style_bind,
    modular_vmap,
    sample_p,
    stage,
)
from .distributions import bernoulli, categorical, geometric, normal

tfd = tfp.distributions

DualTree = Annotated[
    Any,
    Is[lambda v: Dual.static_check_dual_tree(v)],
]
"""
`DualTree` is the type of `Pytree` argument values with `Dual` leaves.
"""

###################
# ADEV primitives #
###################


class ADEVPrimitive(Pytree):
    """
    An `ADEVPrimitive` is a primitive sampler equipped with a JVP
    gradient estimator strategy. These objects support forward sampling,
    but also come equipped with a strategy that interacts with ADEV's
    AD transformation to return a JVP estimate.
    """

    @abstractmethod
    def sample(self, *args) -> Any:
        pass

    @abstractmethod
    def prim_jvp_estimate(
        self,
        dual_tree: tuple[DualTree, ...],
        konts: tuple[
            Callable[..., Any],
            Callable[..., Any],
        ],
    ) -> "Dual":
        pass

    def __call__(self, *args):
        return sample_primitive(self, *args)


####################
# Sample intrinsic #
####################


def sample_primitive(adev_prim: ADEVPrimitive, *args):
    def _adev_prim_call(adev_prim, *args):
        return adev_prim.sample(*args)

    return initial_style_bind(sample_p)(_adev_prim_call)(adev_prim, *args)


####################
# ADEV interpreter #
####################


@Pytree.dataclass
class Dual(Pytree):
    primal: Any
    tangent: Any

    @staticmethod
    def tree_pure(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v
            else:
                return Dual(v, jnp.zeros_like(v))

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def dual_tree(primals, tangents):
        return jtu.tree_map(lambda v1, v2: Dual(v1, v2), primals, tangents)

    @staticmethod
    def tree_primal(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v.primal
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_tangent(v):
        def _inner(v):
            if isinstance(v, Dual):
                return v.tangent
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_leaves(v):
        v = Dual.tree_pure(v)
        return jtu.tree_leaves(v, is_leaf=lambda v: isinstance(v, Dual))

    @staticmethod
    def tree_unzip(v):
        primals = jtu.tree_leaves(Dual.tree_primal(v))
        tangents = jtu.tree_leaves(Dual.tree_tangent(v))
        return tuple(primals), tuple(tangents)

    @staticmethod
    def static_check_is_dual(v) -> bool:
        return isinstance(v, Dual)

    @staticmethod
    def static_check_dual_tree(v) -> bool:
        return all(
            map(
                lambda v: isinstance(v, Dual),
                jtu.tree_leaves(v, is_leaf=Dual.static_check_is_dual),
            )
        )


@Pytree.dataclass
class ADEVInterpreter(Pytree):
    """The `ADEVInterpreter` takes a `Jaxpr`, propagates dual numbers
    through it, while also performing a CPS transformation,
    to compute forward mode AD.

    When this interpreter hits
    the `sample_p` primitive, it creates a pair of continuation closures
    which is passed to the gradient strategy which the primitive is using.
    """

    @staticmethod
    def flat_unzip(duals: list[Any]):
        primals, tangents = jax_util.unzip2((t.primal, t.tangent) for t in duals)
        return list(primals), list(tangents)

    @staticmethod
    def eval_jaxpr_adev(
        jaxpr: Jaxpr,
        consts: list[ArrayLike],
        flat_duals: list[Dual],
    ):
        dual_env = Environment()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, Dual.tree_pure(consts))
        jax_util.safe_map(dual_env.write, jaxpr.invars, flat_duals)

        # TODO: Pure evaluation.
        def eval_jaxpr_iterate_pure(eqns, pure_env, invars, flat_args):
            jax_util.safe_map(pure_env.write, invars, flat_args)
            for eqn in eqns:
                in_vals = jax_util.safe_map(pure_env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals
                if eqn.primitive is sample_p:
                    pass
                else:
                    outs = eqn.primitive.bind(*args, **params)
                    if not eqn.primitive.multiple_results:
                        outs = [outs]
                    jax_util.safe_map(pure_env.write, eqn.outvars, outs)

            return jax_util.safe_map(pure_env.read, jaxpr.outvars)

        # Dual evaluation.
        def eval_jaxpr_iterate_dual(
            eqns,
            dual_env: Environment,
            invars: list[Var],
            flat_duals: list[Dual],
        ):
            jax_util.safe_map(dual_env.write, invars, flat_duals)

            for eqn_idx, eqn in enumerate(eqns):
                with src_util.user_context(eqn.source_info.traceback):
                    in_vals = jax_util.safe_map(dual_env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    duals = subfuns + in_vals

                    primitive, inner_params = ElaboratedPrimitive.unwrap(eqn.primitive)
                    # Our sample_p primitive.
                    if primitive is sample_p:
                        dual_env = dual_env.copy()
                        pure_env = Dual.tree_primal(dual_env)

                        # Create pure continuation.
                        def _sample_pure_kont(*args):
                            return eval_jaxpr_iterate_pure(
                                eqns[eqn_idx + 1 :],
                                pure_env,
                                eqn.outvars,
                                [*args],
                            )

                        # Create dual continuation.
                        def _sample_dual_kont(*duals: Dual):
                            return eval_jaxpr_iterate_dual(
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                list(duals),
                            )

                        in_tree = inner_params["in_tree"]
                        num_consts = inner_params["num_consts"]

                        flat_primals, flat_tangents = ADEVInterpreter.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals[num_consts:]))
                        )
                        adev_prim, *primals = jtu.tree_unflatten(in_tree, flat_primals)
                        _, *tangents = jtu.tree_unflatten(in_tree, flat_tangents)
                        dual_tree = Dual.dual_tree(primals, tangents)

                        return adev_prim.prim_jvp_estimate(
                            tuple(dual_tree),
                            (_sample_pure_kont, _sample_dual_kont),
                        )

                    # Handle branching.
                    elif eqn.primitive is jax.lax.cond_p:
                        pure_env = Dual.tree_primal(dual_env)

                        # Create dual continuation for the computation after the cond_p.
                        def _cond_dual_kont(dual_tree: list[Any]):
                            dual_leaves = Dual.tree_pure(dual_tree)
                            return eval_jaxpr_iterate_dual(
                                eqns[eqn_idx + 1 :],
                                dual_env,
                                eqn.outvars,
                                dual_leaves,
                            )

                        branch_adev_functions = list(
                            map(
                                lambda fn: ADEVInterpreter.forward_mode(
                                    jaxpr_as_fun(fn),
                                    _cond_dual_kont,
                                ),
                                params["branches"],
                            )
                        )

                        # NOTE: the branches are stored in the params
                        # in reverse order, so we need to reverse them
                        # This could totally be something which breaks in the future...
                        return jax.lax.cond(
                            Dual.tree_primal(in_vals[0]),
                            *it.chain(reversed(branch_adev_functions), in_vals[1:]),
                        )

                    # Default JVP rule for other JAX primitives.
                    else:
                        flat_primals, flat_tangents = ADEVInterpreter.flat_unzip(
                            Dual.tree_leaves(Dual.tree_pure(duals))
                        )
                        if len(flat_primals) == 0:
                            primal_outs = eqn.primitive.bind(*flat_primals, **params)
                            tangent_outs = jtu.tree_map(jnp.zeros_like, primal_outs)
                        else:
                            jvp = jax_autodiff.primitive_jvps.get(eqn.primitive)
                            if not jvp:
                                msg = f"differentiation rule for '{eqn.primitive}' not implemented"
                                raise NotImplementedError(msg)
                            primal_outs, tangent_outs = jvp(
                                flat_primals, flat_tangents, **params
                            )

                if not eqn.primitive.multiple_results:
                    primal_outs = [primal_outs]
                    tangent_outs = [tangent_outs]

                jax_util.safe_map(
                    dual_env.write,
                    eqn.outvars,
                    Dual.dual_tree(primal_outs, tangent_outs),
                )
            (out_dual,) = jax_util.safe_map(dual_env.read, jaxpr.outvars)
            if not isinstance(out_dual, Dual):
                out_dual = Dual(out_dual, jnp.zeros_like(out_dual))
            return out_dual

        return eval_jaxpr_iterate_dual(jaxpr.eqns, dual_env, jaxpr.invars, flat_duals)

    @staticmethod
    def forward_mode(f, kont=lambda v: v):
        def _inner(*duals: DualTree):
            primals = Dual.tree_primal(duals)
            closed_jaxpr, (_, _, out_tree) = stage(f)(*primals)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            dual_leaves = Dual.tree_leaves(Dual.tree_pure(duals))
            out_duals = ADEVInterpreter.eval_jaxpr_adev(
                jaxpr,
                consts,
                dual_leaves,
            )
            out_tree_def = out_tree()
            tree_primals, tree_tangents = Dual.tree_unzip(out_duals)
            out_dual_tree = Dual.dual_tree(
                jtu.tree_unflatten(out_tree_def, tree_primals),
                jtu.tree_unflatten(out_tree_def, tree_tangents),
            )
            vs = kont(out_dual_tree)
            return vs

        # Force coercion to JAX arrays.
        def maybe_array(v):
            return jnp.array(v, copy=False)

        def _dual(*duals: DualTree):
            duals = jtu.tree_map(maybe_array, duals)
            return _inner(*duals)

        return _dual


#################
# ADEV programs #
#################


@Pytree.dataclass
class ADEVProgram(Pytree):
    source: Callable[..., Any] = Pytree.static()

    def jvp_estimate(
        self,
        duals: tuple[DualTree, ...],  # Pytree with Dual leaves.
        dual_kont: Callable[..., Any],
    ) -> Dual:
        def adev_jvp(f):
            @wraps(f)
            def wrapped(*duals: DualTree):
                return ADEVInterpreter.forward_mode(self.source, dual_kont)(*duals)

            return wrapped

        return adev_jvp(self.source)(*duals)


###############
# Expectation #
###############


@Pytree.dataclass
class Expectation(Pytree):
    prog: ADEVProgram

    def jvp_estimate(self, *duals: DualTree):
        # Trivial continuation.
        def _identity(v):
            return v

        return self.prog.jvp_estimate(duals, _identity)

    # The JVP rules here are registered below.
    # (c.f. Register custom forward mode with JAX)
    def grad_estimate(self, *primals):
        def _invoke_closed_over(primals):
            return invoke_closed_over(self, primals)

        return jax.grad(_invoke_closed_over)(primals)

    def estimate(self, args):
        tangents = jtu.tree_map(lambda _: 0.0, args)
        return self.jvp_estimate(tangents).primal


def expectation(source: Callable[..., Any]):
    prog = ADEVProgram(source)
    return Expectation(prog)


#########################################
# Register custom forward mode with JAX #
#########################################


# These two functions are defined to external to `Expectation`
# to ignore complexities with defining custom JVP rules for Pytree classes.
@jax.custom_jvp
def invoke_closed_over(instance, args):
    return instance.estimate(*args)


def invoke_closed_over_jvp(primals: tuple, tangents: tuple):
    (instance, primals) = primals
    (_, tangents) = tangents
    duals = Dual.dual_tree(primals, tangents)
    out_dual = instance.jvp_estimate(*duals)
    (v,), (tangent,) = Dual.tree_unzip(out_dual)
    return v, tangent


invoke_closed_over.defjvp(invoke_closed_over_jvp, symbolic_zeros=False)

################################
# Gradient strategy primitives #
################################


@Pytree.dataclass
class REINFORCE(ADEVPrimitive):
    sample_function: Callable[..., Any] = Pytree.static()
    differentiable_logpdf: Callable[..., Any] = Pytree.static()

    def sample(self, *args):
        return self.sample_function(*args)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        primals = Dual.tree_primal(dual_tree)
        tangents = Dual.tree_tangent(dual_tree)
        v = self.sample(*primals)
        dual_tree = Dual.tree_pure(v)
        out_dual = kdual(dual_tree)
        (out_primal,), (out_tangent,) = Dual.tree_unzip(out_dual)
        _, lp_tangent = jax.jvp(
            self.differentiable_logpdf,
            (v, *primals),
            (jnp.zeros_like(v), *tangents),
        )
        return Dual(out_primal, out_tangent + (out_primal * lp_tangent))


def reinforce(sample_func, logpdf_func):
    return REINFORCE(sample_func, logpdf_func)


###########################
# Distribution primitives #
###########################


@Pytree.dataclass
class FlipEnum(ADEVPrimitive):
    def sample(self, *args):
        (probs,) = args
        return 1 == bernoulli.sample(probs)

    def prim_jvp_estimate(
        self,
        dual_tree: tuple[DualTree, ...],
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        true_dual = kdual(Dual(jnp.array(True), jnp.zeros_like(jnp.array(True))))
        false_dual = kdual(Dual(jnp.array(False), jnp.zeros_like(jnp.array(False))))
        (true_primal,), (true_tangent,) = Dual.tree_unzip(true_dual)
        (false_primal,), (false_tangent,) = Dual.tree_unzip(false_dual)

        def _inner(p, tl, fl):
            return p * tl + (1 - p) * fl

        out_primal, out_tangent = jax.jvp(
            _inner,
            (p_primal, true_primal, false_primal),
            (p_tangent, true_tangent, false_tangent),
        )
        return Dual(out_primal, out_tangent)


flip_enum = FlipEnum()


@Pytree.dataclass
class FlipMVD(ADEVPrimitive):
    def sample(self, *args):
        p = (args,)
        return 1 == bernoulli.sample(probs=p)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (kpure, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_primal(dual_tree)
        v = bernoulli.sample(probs=p_primal)
        b = v == 1
        b_primal, b_tangent = kdual((b,), (jnp.zeros_like(b),))
        other = kpure(jnp.logical_not(b))
        est = ((-1) ** v) * (other - b_primal)
        return Dual(b_primal, b_tangent + est * p_tangent)


flip_mvd = FlipMVD()


@Pytree.dataclass
class FlipEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (p,) = args
        return 1 == bernoulli.sample(probs=p)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (p_primal,) = Dual.tree_primal(dual_tree)
        (p_tangent,) = Dual.tree_tangent(dual_tree)
        ret_primals, ret_tangents = modular_vmap(kdual)(
            (jnp.array([True, False]),),
            (jnp.zeros_like(jnp.array([True, False]))),
        )

        def _inner(p, ret):
            return jnp.sum(jnp.array([p, 1 - p]) * ret)

        return Dual(
            *jax.jvp(
                _inner,
                (p_primal, ret_primals),
                (p_tangent, ret_tangents),
            )
        )


flip_enum_parallel = FlipEnumParallel()


@Pytree.dataclass
class CategoricalEnumParallel(ADEVPrimitive):
    def sample(self, *args):
        (probs,) = args
        return categorical.sample(probs)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (_, kdual) = konts
        (probs_primal,) = Dual.tree_primal(dual_tree)
        (probs_tangent,) = Dual.tree_tangent(dual_tree)
        idxs = jnp.arange(len(probs_primal))
        ret_primals, ret_tangents = modular_vmap(kdual)(
            (idxs,), (jnp.zeros_like(idxs),)
        )

        def _inner(probs, primals):
            return jnp.sum(jax.nn.softmax(probs) * primals)

        return Dual(
            *jax.jvp(
                _inner,
                (probs_primal, ret_primals),
                (probs_tangent, ret_tangents),
            )
        )


categorical_enum_parallel = CategoricalEnumParallel()

flip_reinforce = Distribution(
    reinforce(
        bernoulli.sample,
        bernoulli.logpdf,
    ),
    bernoulli.logpdf,
)

geometric_reinforce = Distribution(
    reinforce(
        geometric.sample,
        geometric.logpdf,
    ),
    geometric.logpdf,
)

normal_reinforce = Distribution(
    reinforce(
        normal.sample,
        normal.logpdf,
    ),
    normal.logpdf,
)


@Pytree.dataclass
class NormalREPARAM(ADEVPrimitive):
    def sample(self, *args):
        loc, scale_diag = args
        return normal.sample(loc, scale_diag)

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        _, kdual = konts
        (mu_primal, sigma_primal) = Dual.tree_primal(dual_tree)
        (mu_tangent, sigma_tangent) = Dual.tree_tangent(dual_tree)
        eps = normal.sample(0.0, 1.0)

        def _inner(mu, sigma):
            return mu + sigma * eps

        primal_out, tangent_out = jax.jvp(
            _inner,
            (mu_primal, sigma_primal),
            (mu_tangent, sigma_tangent),
        )
        return kdual(Dual(primal_out, tangent_out))


normal_reparam = Distribution(
    NormalREPARAM(),
    normal.logpdf,
)


@Pytree.dataclass
class Baseline(ADEVPrimitive):
    prim: ADEVPrimitive

    def sample(self, *args):
        return self.prim.sample(*args[1:])

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ):
        (kpure, kdual) = konts
        (b_primal, *prim_primals) = Dual.tree_primal(dual_tree)
        (b_tangent, *prim_tangents) = Dual.tree_tangent(dual_tree)

        def new_kdual(key, dual: Dual):
            ret_dual = kdual(key, dual)

            def _inner(ret, b):
                return ret - b

            primal, tangent = jax.jvp(
                _inner,
                (ret_dual.primal, b_primal),
                (ret_dual.tangent, b_tangent),
            )
            return Dual(primal, tangent)

        l_dual = self.prim.prim_jvp_estimate(
            Dual.dual_tree(prim_primals, prim_tangents),
            (kpure, new_kdual),
        )

        def _inner(left, right):
            return left + right

        primal, tangent = jax.jvp(
            _inner,
            (l_dual.primal, b_primal),
            (l_dual.tangent, b_tangent),
        )
        return Dual(primal, tangent)


def baseline(prim):
    return Baseline(prim)


##################
# Loss primitive #
##################


@Pytree.dataclass
class AddCost(ADEVPrimitive):
    def sample(self, *args):
        (w,) = args
        return w

    def prim_jvp_estimate(
        self,
        dual_tree: DualTree,
        konts: tuple[Any, ...],
    ) -> Dual:
        (_, kdual) = konts
        (w,) = Dual.tree_primal(dual_tree)
        (w_tangent,) = Dual.tree_tangent(dual_tree)
        l_dual = kdual(Dual(None, None))
        return Dual(w + l_dual.primal, w_tangent + l_dual.tangent)


def add_cost(w):
    prim = AddCost()
    prim(w)


###########
# Exports #
###########

__all__ = [
    "Dual",
    "expectation",
    "flip_enum",
]
