from beartype import BeartypeConf
from beartype.claw import beartype_this_package

conf = BeartypeConf(
    is_color=True,
    is_debug=False,
    is_pep484_tower=True,
    violation_type=TypeError,
)

beartype_this_package(conf=conf)

from .adev import (
    Dual,
    add_cost,
    categorical_enum_parallel,
    expectation,
    flip_enum,
    flip_enum_parallel,
    flip_mvd,
    flip_reinforce,
    geometric_reinforce,
    normal_reinforce,
    normal_reparam,
)
from .core import (
    GFI,
    Distribution,
    Fn,
    Pytree,
    Vmap,
    gen,
    get_choices,
    modular_vmap,
    seed,
    sel,
    tfp_distribution,
    trace,
)
from .distributions import (
    attach_discretization,
    bernoulli,
    beta,
    categorical,
    flip,
    labeled_cat,
    normal,
    normal_grid_around_mean,
)
from .enum import enum
from .sp import (
    Algorithm,
    Importance,
    Marginal,
    marginal,
)

__all__ = [
    "GFI",
    "Distribution",
    "Dual",
    "Fn",
    "flip_mvd",
    "labeled_cat",
    "flip_enum_parallel",
    "flip_reinforce",
    "categorical_enum_parallel",
    "geometric_reinforce",
    "add_cost",
    "Algorithm",
    "Importance",
    "Vmap",
    "bernoulli",
    "Pytree",
    "sel",
    "Marginal",
    "beta",
    "enum",
    "seed",
    "categorical",
    "expectation",
    "flip",
    "flip_enum",
    "normal_reparam",
    "normal_reinforce",
    "attach_discretization",
    "gen",
    "get_choices",
    "marginal",
    "modular_vmap",
    "normal_grid_around_mean",
    "normal",
    "tfp_distribution",
    "trace",
]
