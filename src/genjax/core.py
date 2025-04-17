import itertools as it
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import overload

import beartype.typing as btyping
import jax
import jax.core as jc
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import jaxtyping as jtyping
import penzai.pz as pz
from jax import api_util, tree_util
from jax._src.interpreters.batching import AxisData  # pyright: ignore
from jax.core import eval_jaxpr
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Primitive, Var
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe
from jax.lax import cond_p, scan, scan_p, switch
from jax.scipy.special import logsumexp
from jax.util import safe_map, split_list
from numpy import dtype
from tensorflow_probability.substrates import jax as tfp
from typing_extensions import dataclass_transform

tfd = tfp.distributions

##########
# Typing #
##########

Any = btyping.Any
PRNGKey = jtyping.PRNGKeyArray
Array = jtyping.Array
ArrayLike = jtyping.ArrayLike
IntArray = jtyping.Int[jtyping.Array, "..."]
FloatArray = jtyping.Float[jtyping.Array, "..."]
BoolArray = jtyping.Bool[jtyping.Array, "..."]
Callable = btyping.Callable
TypeAlias = btyping.TypeAlias
Sequence = btyping.Sequence
Iterable = btyping.Iterable
Final = btyping.Final
Generator = btyping.Generator
Generic = btyping.Generic
InAxes = int | Sequence[Any] | None
Flag = bool | BoolArray
TypeVar = btyping.TypeVar

A = TypeVar("A")
R = TypeVar("R")
X = TypeVar("X")

#######################
# Probabilistic types #
#######################

Weight = FloatArray
Score = FloatArray

##########
# Pytree #
##########


class Pytree(pz.Struct):
    """`Pytree` is an abstract base class which registers a class with JAX's `Pytree`
    system. JAX's `Pytree` system tracks how data classes should behave across
    JAX-transformed function boundaries, like `jax.jit` or `jax.vmap`.

    Inheriting this class provides the implementor with the freedom to
    declare how the subfields of a class should behave:

    * `Pytree.static(...)`: the value of the field cannot
    be a JAX traced value, it must be a Python literal, or a constant).
    The values of static fields are embedded in the `PyTreeDef` of any
    instance of the class.
    * `Pytree.field(...)` or no annotation: the value may be a JAX traced
    value, and JAX will attempt to convert it to tracer values inside of
    its transformations.

    If a field _points to another `Pytree`_, it should not be declared as
    `Pytree.static()`, as the `Pytree` interface will automatically handle
    the `Pytree` fields as dynamic fields.
    """

    @staticmethod
    @overload
    def dataclass(
        incoming: None = None,
        /,
        **kwargs,
    ) -> Callable[[type[R]], type[R]]: ...

    @staticmethod
    @overload
    def dataclass(
        incoming: type[R],
        /,
        **kwargs,
    ) -> type[R]: ...

    @dataclass_transform(
        frozen_default=True,
    )
    @staticmethod
    def dataclass(
        incoming: type[R] | None = None,
        /,
        **kwargs,
    ) -> type[R] | Callable[[type[R]], type[R]]:
        """
        Denote that a class (which is inheriting `Pytree`) should be treated
        as a dataclass, meaning it can hold data in fields which are
        declared as part of the class.

        A dataclass is to be distinguished from a "methods only"
        `Pytree` class, which does not have fields, but may define methods.
        The latter cannot be instantiated, but can be inherited from,
        while the former can be instantiated:
        the `Pytree.dataclass` declaration informs the system _how
        to instantiate_ the class as a dataclass,
        and how to automatically define JAX's `Pytree` interfaces
        (`tree_flatten`, `tree_unflatten`, etc.) for the dataclass, based
        on the fields declared in the class, and possibly `Pytree.static(...)`
        or `Pytree.field(...)` annotations (or lack thereof, the default is
        that all fields are `Pytree.field(...)`).

        All `Pytree` dataclasses support pretty printing, as well as rendering
        to HTML.

        Examples
        --------

        ```{python}
        from genjax import Pytree
        from jaxtyping import ArrayLike
        import jax.numpy as jnp


        @Pytree.dataclass
        # Enforces type annotations on instantiation.
        class MyClass(Pytree):
            my_static_field: int = Pytree.static()
            my_dynamic_field: ArrayLike


        MyClass(10, jnp.array(5.0))
        ```
        """

        return pz.pytree_dataclass(
            incoming,
            overwrite_parent_init=True,
            **kwargs,
        )

    @staticmethod
    def static(**kwargs):
        """Declare a field of a `Pytree` dataclass to be static.
        Users can provide additional keyword argument options,
        like `default` or `default_factory`, to customize how the field is
        instantiated when an instance of
        the dataclass is instantiated.` Fields which are provided with default
        values must come after required fields in the dataclass declaration.

        Examples
        --------

        ```{python}
        @Pytree.dataclass
        # Enforces type annotations on instantiation.
        class MyClass(Pytree):
            my_dynamic_field: ArrayLike
            my_static_field: int = Pytree.static(default=0)


        MyClass(jnp.array(5.0))
        ```

        """
        return field(metadata={"pytree_node": False}, **kwargs)

    @staticmethod
    def field(**kwargs):
        """Declare a field of a `Pytree` dataclass to be dynamic.
        Alternatively, one can leave the annotation off in the declaration."""
        return field(**kwargs)


############################################
# Staging utilities for Jaxpr interpreters #
############################################


def get_shaped_aval(x):
    return jc.get_aval(x)


@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


def stage(f, **params):
    """Returns a function that stages a function to a ClosedJaxpr."""

    @wraps(f)
    def wrapped(
        *args, **kwargs
    ) -> tuple[ClosedJaxpr, tuple[list[Any], Any, Callable[..., Any]]]:
        debug_info = api_util.debug_info("genjax.stage", f, args, kwargs)
        fun = lu.wrap_init(f, params, debug_info=debug_info)
        if kwargs:
            flat_args, in_tree = jtu.tree_flatten((args, kwargs))
            flat_fun, out_tree = api_util.flatten_fun(fun, in_tree)
        else:
            flat_args, in_tree = jtu.tree_flatten(args)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        closed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return closed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


#########################
# Custom JAX primitives #
#########################


class InitialStylePrimitive(Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(InitialStylePrimitive, self).__init__(name)
        self.multiple_results = True

        def impl(*flat_args, **params):
            return params["impl"](*flat_args, **params)

        def abstract(*flat_avals, **params):
            return params["abstract"](*flat_avals, **params)

        def jvp(
            flat_primals: tuple[Any, ...] | list[Any],
            flat_tangents: tuple[Any, ...] | list[Any],
            **params,
        ) -> tuple[list[Any], list[Any]]:
            return params["jvp"](flat_primals, flat_tangents, **params)

        def batch(flat_vector_args: tuple[Any, ...] | list[Any], dim, **params):
            return params["batch"](flat_vector_args, dim, **params)

        def lowering(ctx: mlir.LoweringRuleContext, *mlir_args, **params):
            if "lowering_warning" in params and lowering_warning:
                warnings.warn(params["lowering_warning"])
            elif "lowering_exception" in params and enforce_lowering_exception:
                raise params["lowering_exception"]
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(ctx, *mlir_args, **params)

        # Store for elaboration.
        self.impl = impl
        self.jvp = jvp
        self.abstract = abstract
        self.batch = batch
        self.lowering = lowering

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)


class ElaboratedPrimitive(Primitive):
    """
    An `ElaboratedPrimitive` is a primitive which wraps an underlying
    primitive, but is elaborated with additional parameters that only show up
    in the pretty printing of a `Jaxpr`.
    In addition, `ElaboratedPrimitive` instances hide excess metadata in
    pretty printing.
    """

    def __init__(self, prim: InitialStylePrimitive, **params):
        super(ElaboratedPrimitive, self).__init__(prim.name)
        self.prim = prim
        self.multiple_results = self.prim.multiple_results
        self.params = params

        def impl(*args, **params):
            return self.prim.impl(*args, **self.params, **params)

        def abstract(*args, **params):
            return self.prim.abstract(*args, **self.params, **params)

        def jvp(*args, **params):
            return self.prim.jvp(*args, **self.params, **params)

        def batch(*args, **params):
            return self.prim.batch(*args, **self.params, **params)

        def lowering(*args, **params):
            return self.prim.lowering(*args, **self.params, **params)

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)

    @classmethod
    def unwrap(cls, v):
        return (v.prim, v.params) if isinstance(v, ElaboratedPrimitive) else (v, {})

    @classmethod
    def check(cls, primitive, other):
        if isinstance(primitive, ElaboratedPrimitive):
            return primitive.prim == other
        else:
            return primitive == other

    @classmethod
    def rebind(cls, primitive: Primitive, *args, **params):
        return primitive.bind(*args, **params)


def batch_fun(fun: lu.WrappedFun, axis_data, in_dims):
    tag = jc.TraceTag()
    in_dims = in_dims() if callable(in_dims) else in_dims
    batched, out_dims = batching.batch_subtrace(fun, tag, axis_data, in_dims)
    return batched, out_dims


def initial_style_bind(
    prim,
    modular_vmap_aware=True,
    **params,
):
    """Binds a primitive to a function call."""

    def bind(f, **elaboration_kwargs):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to an `ElaboratedPrimitive`
            primitive, hiding the implementation details of the eval
            (impl) rule, abstract rule, batch rule, and jvp rule."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            debug_info = jaxpr.jaxpr.debug_info

            def impl(*flat_args, **params) -> list[Any]:
                consts, flat_args = split_list(flat_args, [params["num_consts"]])
                return jc.eval_jaxpr(jaxpr.jaxpr, consts, *flat_args)

            def abstract(*flat_avals, **params):
                if modular_vmap_aware:
                    if "ctx" in params and params["ctx"] == "modular_vmap":
                        flat_avals = flat_avals[1:]  # ignore dummy
                return pe.abstract_eval_fun(
                    impl,
                    *flat_avals,
                    debug_info=debug_info,
                    **params,
                )

            def batch(flat_vector_args: tuple[Any, ...] | list[Any], dims, **params):
                axis_data = AxisData(None, None, None, None)
                batched, out_dims = batch_fun(
                    lu.wrap_init(impl, params, debug_info=debug_info),
                    axis_data,
                    dims,
                )
                return batched.call_wrapped(*flat_vector_args), out_dims()

            def jvp(
                flat_primals: tuple[Any, ...] | list[Any],
                flat_tangents: tuple[Any, ...] | list[Any],
                **params,
            ) -> tuple[list[Any], list[Any]]:
                primals_out, tangents_out = ad.jvp(
                    lu.wrap_init(impl, params, debug_info=debug_info)
                ).call_wrapped(flat_primals, flat_tangents)

                # We always normalize back to list.
                return list(primals_out), list(tangents_out)

            if "impl" in params:
                impl = params["impl"]
                params.pop("impl")

            if "abstract" in params:
                abstract = params["abstract"]
                params.pop("abstract")

            if "batch" in params:
                batch = params["batch"]
                params.pop("batch")

            if "jvp" in params:
                jvp = params["jvp"]
                params.pop("jvp")

            elaborated_prim = ElaboratedPrimitive(
                prim,
                impl=impl,
                abstract=abstract,
                batch=batch,
                jvp=jvp,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                **params,
            )
            outs = elaborated_prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                **elaboration_kwargs,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind


########
# PJAX #
########


def static_dim_length(in_axes, args: tuple[Any, ...]) -> int | None:
    # perform the in_axes massaging that vmap performs internally:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)
    elif isinstance(in_axes, list):
        in_axes = tuple(in_axes)

    def find_axis_size(axis: int | None, x: Any) -> int | None:
        """Find the size of the axis specified by `axis` for the argument `x`."""
        if axis is not None:
            leaf = jtu.tree_leaves(x)[0]
            return leaf.shape[axis]

    # tree_map uses in_axes as a template. To have passed vmap validation, Any non-None entry
    # must bottom out in an array-shaped leaf, and all such leafs must have the same size for
    # the specified dimension. Fetching the first is sufficient.
    axis_sizes = jtu.tree_leaves(
        jtu.tree_map(
            find_axis_size,
            in_axes,
            args,
            is_leaf=lambda x: x is None,
        )
    )
    return axis_sizes[0] if axis_sizes else None


######################
# Sampling primitive #
######################


class style:
    CYAN = "\033[36m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


assume_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.assume{style.RESET}",
)
observe_p = InitialStylePrimitive(
    f"{style.BOLD}{style.CYAN}pjax.observe{style.RESET}",
)


# Zero-cost, just staging.
def make_flat(f):
    @wraps(f)
    def _make_flat(*args, **kwargs):
        debug_info = api_util.debug_info("_make_flat", f, args, kwargs)
        jaxpr, *_ = stage(f)(*args, **kwargs)

        def flat(*flat_args, **params):
            consts, args = split_list(flat_args, [params["num_consts"]])
            return eval_jaxpr(jaxpr.jaxpr, consts, *args)

        return flat, debug_info

    return _make_flat


class LoweringSamplePrimitiveToMLIRException(Exception):
    pass


# This is very cheeky.
@dataclass
class GlobalKeyCounter:
    count: int = 0


# Very large source of unique keys.
global_counter = GlobalKeyCounter()

_fake_key = jrand.key(1)


def assume_binder(
    keyful_sampler: Callable[..., Any],
    name: str | None = None,
    support: Callable[..., Any] | None = None,
    batch_shape: tuple[int, ...] = (),
):
    keyful_with_batch_shape = partial(keyful_sampler, sample_shape=batch_shape)

    def sampler(*args, **kwargs):
        # We're playing a trick here by allowing users to invoke sample_p
        # without a key. So we hide it inside, and we pass this as the
        # impl of `sample_p`.
        #
        # This is problematic for JIT, which will cache the statically
        # generated key. But it's obvious to the user - their returned
        # random choices won't change!
        #
        # The `seed` transformation below solves the JIT problem directly.
        def keyless(*args, **kwargs):
            global_counter.count += 1
            return keyful_with_batch_shape(
                jrand.key(global_counter.count),
                *args,
                **kwargs,
            )

        # Zero-cost, just staging.
        flat_keyful_sampler, _ = make_flat(keyful_with_batch_shape)(
            _fake_key, *args, **kwargs
        )

        # Overload batching so that the primitive is retained
        # in the Jaxpr under vmap.
        # Holy smokes recursion.
        def batch(vector_args, batch_axes, **params):
            if "ctx" in params and params["ctx"] == "modular_vmap":
                axis_size = params["axis_size"]
                vector_args = tuple(vector_args[1:])  # ignore dummy
                batch_axes = tuple(batch_axes[1:])  # ignore dummy
                n = static_dim_length(batch_axes, vector_args)
                outer_batch_dim = (
                    () if n is not None else (axis_size,) if axis_size else ()
                )
                assert isinstance(outer_batch_dim, tuple)
                new_batch_shape = outer_batch_dim + batch_shape
                v = assume_binder(
                    keyful_sampler,
                    name=name,
                    batch_shape=new_batch_shape,
                )(*vector_args)
                return (v,), (0 if n or axis_size else None,)
            else:
                raise NotImplementedError()

        lowering_msg = (
            "JAX is attempting to lower the `pjax.sample_p` primitive to MLIR. "
            "This will bake a PRNG key into the MLIR code, resulting in deterministic behavior. "
            "Instead, use `pjax.seed` to transform your function into one which allows keys to be passed in. "
            "You can do this at any level of your computation above the `sample_p` invocation."
        )
        lowering_exception = LoweringSamplePrimitiveToMLIRException(
            lowering_msg,
        )

        return initial_style_bind(
            assume_p,
            keyful_sampler=keyful_sampler,
            flat_keyful_sampler=flat_keyful_sampler,
            batch=batch,
            support=support,
            lowering_warning=lowering_msg,
            lowering_exception=lowering_exception,
        )(keyless, dist=name)(*args, **kwargs)

    return sampler


def observe_binder(
    log_density_impl: Callable[..., Any],
    name: str | None = None,
):
    def observe(*args, **kwargs):
        # TODO: really not sure if this is right if you
        # nest vmaps...
        def batch(vector_args, batch_axes, **params):
            n = static_dim_length(batch_axes, tuple(vector_args))
            v = observe_binder(
                jax.vmap(log_density_impl, in_axes=batch_axes), name=name
            )(*vector_args)
            outvals = (v,)
            out_axes = (0 if n else None,)
            return outvals, out_axes

        return initial_style_bind(
            observe_p,
            batch=batch,
        )(log_density_impl, dist=name)(*args, **kwargs)

    return observe


VarOrLiteral = Var | Literal


@Pytree.dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during interpretation."""

    env: dict[int, Any] = Pytree.field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        """
        Read a value from a variable in the environment.
        """
        v = self.get(var)
        if v is None:
            assert isinstance(var, Var)
            raise ValueError(
                f"Unbound variable in interpreter environment at count {var.count}:\nEnvironment keys (count): {list(self.env.keys())}"
            )
        return v

    def get(self, var: VarOrLiteral) -> Any:
        if isinstance(var, Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        """
        Write a value to a variable in the environment.
        """
        if isinstance(var, Literal):
            return cell
        cur_cell = self.get(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        """
        Check if a variable is in the environment.
        """
        if isinstance(var, Literal):
            return True
        return var.count in self.env

    def copy(self):
        """
        `Environment.copy` is sometimes used to create a new environment with
        the same variables and values as the original, especially in CPS
        interpreters (where a continuation closes over the application of an
        interpreter to a `Jaxpr`).
        """
        keys = list(self.env.keys())
        return Environment({k: self.env[k] for k in keys})


####################
# Seed Interpreter #
####################


@dataclass
class SeedInterpreter:
    key: PRNGKey

    def eval_jaxpr_seed(
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
                invals = safe_map(env.read, eqn.invars)
                args = subfuns + invals
                flat_keyful_sampler = inner_params["flat_keyful_sampler"]
                self.key, sub_key = jrand.split(self.key)
                outvals = flat_keyful_sampler(sub_key, *args, **inner_params)

            elif primitive == cond_p:
                invals = safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                self.key, sub_key = jrand.split(self.key)
                branches = tuple(
                    seed(jex.core.jaxpr_as_fun(branch))
                    for branch in branch_closed_jaxprs
                )
                index_val, ops_vals = invals[0], invals[1:]
                outvals = switch(
                    index_val,
                    branches,
                    sub_key,
                    *ops_vals,
                )

            # We replace the original scan with a new scan
            # that calls the interpreter on the scan body,
            # carries the key through and evolves it.
            elif primitive == scan_p:
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = split_list(
                    invals, [num_consts, num_carry]
                )

                body_fun = jex.core.jaxpr_as_fun(body_jaxpr)

                def new_body(carry, scanned_in):
                    (key, in_carry) = carry
                    (idx, in_scan) = scanned_in
                    all_values = const_vals + jtu.tree_leaves((in_carry, in_scan))
                    sub_key = jrand.fold_in(key, idx)
                    outs = seed(body_fun)(sub_key, *all_values)
                    out_carry, out_scan = split_list(outs, [num_carry])
                    return (key, out_carry), out_scan

                self.key, sub_key = jrand.split(self.key)
                fold_idxs = jnp.arange(length)
                (_, flat_carry_out), scanned_out = scan(
                    new_body,
                    (sub_key, carry_vals),
                    (fold_idxs, xs_vals),
                    length=length,
                    reverse=reverse,
                )
                outvals = jtu.tree_leaves(
                    (flat_carry_out, scanned_out),
                )

            else:
                outvals = eqn.primitive.bind(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(fn)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_seed(
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def seed(
    f: Callable[..., Any],
):
    @wraps(f)
    def wrapped(key: PRNGKey, *args):
        interpreter = SeedInterpreter(key)
        return interpreter.run_interpreter(
            f,
            *args,
        )

    return wrapped


############################################
# Interpreter which modularly extends vmap #
############################################


@dataclass
class ModularVmapInterpreter:
    """
    `ModularVmapInterpreter` is an interpreter which piggybacks off of
    `jax.vmap`,
    but is aware of probability distributions as first class citizens.
    """

    @classmethod
    def eval_jaxpr_modular_vmap(
        cls,
        axis_size: int,
        jaxpr: Jaxpr,
        consts: list[Any],
        flat_args: list[Any],
        dummy_arg: Array,
    ):
        env = Environment()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, flat_args)
        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals

            # Probabilistic.
            if ElaboratedPrimitive.check(eqn.primitive, assume_p):
                outvals = eqn.primitive.bind(
                    dummy_arg,
                    *args,
                    axis_size=axis_size,
                    ctx="modular_vmap",
                    **params,
                )

            elif eqn.primitive == cond_p:
                invals = safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                branch_closed_jaxprs = params["branches"]
                branches = tuple(
                    partial(
                        ModularVmapInterpreter.stage_and_run,
                        axis_size,
                        jex.core.jaxpr_as_fun(branch),
                    )
                    for branch in branch_closed_jaxprs
                )
                index_val, ops_vals = invals[0], invals[1:]
                outvals = switch(
                    index_val,
                    branches,
                    dummy_arg,
                    ops_vals,
                )

            elif eqn.primitive == scan_p:
                body_jaxpr = params["jaxpr"]
                length = params["length"]
                reverse = params["reverse"]
                unroll = params["unroll"]
                num_consts = params["num_consts"]
                num_carry = params["num_carry"]
                const_vals, carry_vals, xs_vals = split_list(
                    invals, [num_consts, num_carry]
                )

                body_fun = partial(
                    ModularVmapInterpreter.stage_and_run,
                    axis_size,
                    jex.core.jaxpr_as_fun(body_jaxpr),
                )

                def new_body(carry, x):
                    (dummy, *in_carry) = carry
                    all_out = body_fun(
                        dummy,
                        (*const_vals, *in_carry, *x),
                    )
                    out_carry, out_scan = split_list(all_out, [num_carry])
                    return (dummy, *out_carry), out_scan

                (_, *out_carry), out_scan = scan(
                    new_body,
                    (dummy_arg, *carry_vals),
                    xs=xs_vals,
                    length=length,
                    reverse=reverse,
                    unroll=unroll,
                )
                outvals = jtu.tree_leaves(
                    (out_carry, out_scan),
                )

            # Deterministic and not control flow.
            else:
                outvals = eqn.primitive.bind(*args, **params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]

            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    @classmethod
    def stage_and_run(cls, axis_size, fn, dummy_arg, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(
            fn,
        )(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        assert axis_size is not None
        flat_out = ModularVmapInterpreter.eval_jaxpr_modular_vmap(
            axis_size,
            jaxpr,
            consts,
            flat_args,
            dummy_arg,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)

    def run_interpreter(
        self,
        in_axes: int | tuple[int | None, ...] | Sequence[Any] | None,
        axis_size: int | None,
        axis_name: str | None,
        spmd_axis_name: str | None,
        fn,
        *args,
    ):
        axis_size = static_dim_length(in_axes, args) if axis_size is None else axis_size
        assert axis_size is not None

        dummy_arg = jnp.array(
            [1 for _ in range(axis_size)],
            dtype=dtype("i1"),
        )
        return jax.vmap(
            partial(
                ModularVmapInterpreter.stage_and_run,
                axis_size,
                fn,
            ),
            in_axes=(0, in_axes),
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )(dummy_arg, args)


########
# APIs #
########


def modular_vmap(
    f: Callable[..., Any],
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
):
    """
    `modular_vmap` extends `jax.vmap` to be "aware of probability distributions"
    as a first class citizen.
    """

    @wraps(f)
    def wrapped(*args):
        # Quickly throw if "normal" vmap would fail.
        jax.vmap(
            lambda *_: None,
            in_axes=in_axes,
            axis_size=axis_size,
            axis_name=axis_name,
            spmd_axis_name=spmd_axis_name,
        )(*args)

        interpreter = ModularVmapInterpreter()
        return interpreter.run_interpreter(
            in_axes,
            axis_size,
            axis_name,
            spmd_axis_name,
            f,
            *args,
        )

    return wrapped


# Control the behavior of lowering `sample_p`
# to MLIR.
# `sample_p` should be _eliminated_ by `seed`,
# otherwise, keys are baked in, yielding deterministic results.
# These flags send warnings / raise exceptions in case this happens.
enforce_lowering_exception = True
lowering_warning = True

########################
# Generative functions #
########################


@Pytree.dataclass
class Trace(Generic[X, R], Pytree):
    _gen_fn: "GFI[X, R]"
    _args: Any
    _choices: X
    _retval: R
    _score: Score

    def get_gen_fn(self) -> "GFI[X, R]":
        return self._gen_fn

    def get_choices(self) -> X:
        return self._choices

    def get_args(self) -> Any:
        return self._args

    def get_retval(self) -> R:
        return self._retval

    def get_score(self) -> Score:
        if jnp.shape(self._score):
            return jnp.sum(self._score)
        else:
            return self._score

    def update(self, args, x: X):
        gen_fn = self.get_gen_fn()
        return gen_fn.update(args, self, x)

    def __getitem__(self, addr):
        choices = self.get_choices()
        return get_choices(choices[addr])  # pyright: ignore


def get_choices(x: Trace[X, R] | X) -> X:
    if isinstance(x, Trace):
        return x.get_choices()
    return x


class GFI(Generic[X, R], Pytree):
    def __call__(self, *args) -> "Thunk[X, R] | R":
        if handler_stack:
            return Thunk(self, args)
        else:
            tr = self.simulate(args)
            return tr.get_retval()

    def T(self, *args) -> "Thunk[X, R]":
        return Thunk(self, args)

    @abstractmethod
    def simulate(
        self,
        args,
    ) -> Trace[X, R]:
        pass

    @abstractmethod
    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Weight, R]:
        pass

    @abstractmethod
    def update(
        self,
        args_,
        tr: Trace[X, R],
        x_: X,
    ) -> tuple[Trace[X, R], Weight, R, X]:
        pass

    def vmap(
        self,
        in_axes: int | tuple[int | None, ...] | Sequence[Any] | None = 0,
        axis_size=None,
        axis_name: str | None = None,
        spmd_axis_name: str | None = None,
    ):
        return Vmap(
            self,
            in_axes,
            axis_size,
            axis_name,
            spmd_axis_name,
        )

    def repeat(self, n: int):
        return self.vmap(in_axes=None, axis_size=n)


@Pytree.dataclass
class Thunk(Generic[X, R], Pytree):
    gen_fn: GFI[X, R]
    args: tuple

    def __matmul__(self, other: str):
        return trace(other, self.gen_fn, self.args)


@Pytree.dataclass
class Vmap(Generic[X, R], GFI[X, R]):
    gen_fn: GFI[X, R]
    in_axes: int | tuple[int | None, ...] | Sequence[Any] | None
    axis_size: int | None
    axis_name: str | None
    spmd_axis_name: str | None

    def simulate(
        self,
        args,
    ) -> Trace[X, R]:
        tr = modular_vmap(
            self.gen_fn.simulate,
            in_axes=(self.in_axes,),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.axis_name,
        )(args)
        return tr

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Weight, R]:
        w, r = modular_vmap(
            self.gen_fn.assess,
            in_axes=(self.in_axes, 0),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.axis_name,
        )(args, x)
        return jnp.sum(w), r

    def update(
        self,
        args_,
        tr: Trace[X, R],
        x_: X,
    ) -> tuple[Trace[X, R], Weight, R, X]:
        new_tr, w, r, discard = modular_vmap(
            self.gen_fn.update,
            in_axes=(self.in_axes, 0, 0),
            axis_size=self.axis_size,
            axis_name=self.axis_name,
            spmd_axis_name=self.axis_name,
        )(args_, tr, x_)
        return new_tr, jnp.sum(w), r, discard


#################
# Distributions #
#################


@Pytree.dataclass
class Distribution(Generic[X], GFI[X, X]):
    assume: Callable[..., X] = Pytree.static()
    observe: Callable[..., Weight] = Pytree.static()

    def simulate(
        self,
        args,
    ) -> Trace[X, X]:
        x = self.assume(*args)
        log_density = self.observe(x, *args)
        return Trace(self, args, x, x, -log_density)

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Weight, X]:
        log_density = self.observe(x, *args)
        return log_density, x

    def update(
        self,
        args_,
        tr: Trace[X, X],
        x_: X,
    ) -> tuple[Trace[X, X], Weight, X, X]:
        log_density_ = self.observe(x_, *args_)
        return (
            Trace(self, args_, x_, x_, -log_density_),
            log_density_ + tr.get_score(),
            x_,
            tr.get_retval(),
        )


# Mostly, just use TFP.
# This wraps PJAX's `sample_p` correctly.
def tfp_distribution[X](
    dist: Callable[..., "tfd.Distribution"],
    support: Callable[..., Any] | None = None,
    name: str | None = None,
) -> Distribution[Any]:
    """
    Creates a generative function from a TensorFlow Probability distribution.

    Args:
        dist: A callable that returns a TensorFlow Probability distribution.

    Returns:
        A generative function wrapping the TensorFlow Probability distribution.

    This function creates a generative function that encapsulates the sampling
    and log probability
    computation of a TensorFlow Probability distribution. It uses the
    distribution's `sample` and
    `log_prob` methods to define the generative function's behavior.
    """

    def assume(*args, **kwargs):
        def keyful_sampler(key, *args, sample_shape=(), **kwargs):
            d = dist(*args, **kwargs)
            return d.sample(seed=key, sample_shape=sample_shape)

        batch_shape = kwargs.get("shape", ())
        if "shape" in kwargs:
            kwargs.pop("shape")
        return assume_binder(
            keyful_sampler,
            name=name,
            support=support,
            batch_shape=batch_shape,
        )(
            *args,
            **kwargs,
        )

    def observe(v, *args, **kwargs):
        def _log_density(v, *args, **kwargs):
            d = dist(*args, **kwargs)
            return d.log_prob(v)

        return observe_binder(_log_density, name=name)(v, *args, **kwargs)

    return Distribution(
        assume,
        observe,
    )


######
# Fn #
######


@dataclass
class Simulate:
    score: Weight
    trace_map: dict[str, Any]

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
    ) -> R:
        tr = gen_fn.simulate(args)
        self.score += tr.get_score()
        self.trace_map[addr] = tr
        return tr.get_retval()


@dataclass
class Assess:
    choice_map: dict[str, Any]
    weight: Weight

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args,
    ) -> R:
        x = self.choice_map[addr]
        x = get_choices(x)
        w, r = gen_fn.assess(args, x)
        self.weight += w
        return r


@dataclass
class Update(Generic[R]):
    trace: Trace[dict[str, Any], R]
    choice_map: dict[str, Any]
    trace_map: dict[str, Any]
    discard: dict[str, Any]
    score: Score
    weight: Weight

    def __call__(
        self,
        addr: str,
        gen_fn: GFI[X, R],
        args_,
    ) -> R:
        subtrace = self.trace.get_choices()[addr]
        x = self.choice_map[addr] if addr in self.choice_map else self.trace[addr]
        x = get_choices(x)
        tr, w, r, discard = gen_fn.update(args_, subtrace, x)
        self.trace_map[addr] = tr
        self.discard[addr] = discard
        self.score += tr.get_score()
        self.weight += w
        return r


handler_stack: list[Simulate | Assess | Update] = []


# Generative function "FFI" invocation.
def trace(
    addr: str,
    gen_fn: GFI[X, R],
    args,
) -> R:
    handler = handler_stack[-1]
    retval = handler(addr, gen_fn, args)
    return retval


@Pytree.dataclass
class Fn(
    Generic[R],
    GFI[dict[str, Any], R],
):
    source: Callable[..., R] = Pytree.static()

    def simulate(
        self,
        args,
    ) -> Trace[dict[str, Any], R]:
        handler_stack.append(Simulate(jnp.array(0.0), {}))
        r = self.source(*args)
        handler = handler_stack.pop()
        assert isinstance(handler, Simulate)
        score, trace_map = handler.score, handler.trace_map
        return Trace(self, args, trace_map, r, score)

    def assess(
        self,
        args,
        x: dict[str, Any],
    ) -> tuple[Weight, R]:
        handler_stack.append(Assess(x, jnp.array(0.0)))
        r = self.source(*args)
        handler = handler_stack.pop()
        assert isinstance(handler, Assess)
        w = handler.weight
        return w, r

    def update(
        self,
        args_,
        tr: Trace[dict[str, Any], R],
        x_: dict[str, Any],
    ) -> tuple[Trace[dict[str, Any], R], Weight, R, dict[str, Any]]:
        handler_stack.append(Update(tr, x_, {}, {}, jnp.array(0.0), jnp.array(0.0)))
        r = self.source(*args_)
        handler = handler_stack.pop()
        assert isinstance(handler, Update)
        trace_map, score, w, discard = (
            handler.trace_map,
            handler.score,
            handler.weight,
            handler.discard,
        )
        return Trace(self, args_, trace_map, r, score), w, r, discard


def gen(fn: Callable[..., R]) -> Fn[R]:
    return Fn(source=fn)


############
# Marginal #
############


class Algorithm(Generic[X], GFI[X, X]):
    pass


@Pytree.dataclass
class Importance(Generic[R], Algorithm[dict[str, Any]]):
    proposal: Fn[R]
    K: int = Pytree.static(default=2)

    def simulate(self, args) -> Trace[dict[str, Any], dict[str, Any]]:
        from genjax import categorical

        (gen_fn, addr, constraint), *args = args
        tr = modular_vmap(self.proposal.simulate, axis_size=self.K)(args)
        choices = tr.get_choices()

        def _assess(choices):
            choices[addr] = constraint
            p, _ = gen_fn.assess(args, choices)
            return p

        ps = modular_vmap(_assess)(choices)
        ws = ps - modular_vmap(lambda tr: tr.get_score())(tr)
        idx = categorical.assume(ws)
        v = jtu.tree_map(lambda x: x[idx], choices)
        Z = logsumexp(ws) - jnp.log(self.K)
        return Trace(self, args, v, v, Z)

    def assess(self, args, x: X) -> tuple[Weight, X]:
        raise NotImplementedError

    def update(
        self,
        args_,
        tr: Trace[X, X],
        x_: X,
    ) -> tuple[Trace[X, X], Weight, X, X]:
        raise NotImplementedError


@Pytree.dataclass
class Marginal(Generic[R, X], GFI[X, X]):
    gen_fn: Fn[R]
    alg: Algorithm[dict[str, Any]]
    addr: str = Pytree.static()

    def simulate(
        self,
        args,
    ) -> Trace[X, X]:
        tr = self.gen_fn.simulate(args)
        choices = tr.get_choices()
        marginalized = get_choices(choices.pop(self.addr))
        weight, _ = self.alg.assess(
            ((self.gen_fn, self.addr, marginalized), *args),
            choices,
        )
        return Trace(
            self,
            args,
            marginalized,
            marginalized,
            tr.get_score() - weight,
        )

    def assess(
        self,
        args,
        x: X,
    ) -> tuple[Weight, X]:
        tr = self.alg.simulate(
            ((self.gen_fn, self.addr, x), *args),
        )
        choices = tr.get_choices()
        choices[self.addr] = x
        weight, _ = self.gen_fn.assess(args, choices)
        return weight - tr.get_score(), x

    def update(
        self,
        args_,
        tr: Trace[X, X],
        x_: X,
    ) -> tuple[Trace[X, X], Weight, X, X]:
        raise NotImplementedError


def marginal(
    gen_fn: Fn[R],
    alg: Algorithm[dict[str, Any]],
    addr: str,
) -> Marginal[R, Any]:
    return Marginal(gen_fn, alg, addr)
