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
    expectation,
    flip_enum,
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
    tfp_distribution,
    trace,
)
from .distributions import (
    bernoulli,
    beta,
    categorical,
    flip,
    normal,
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
    "Algorithm",
    "Importance",
    "Vmap",
    "bernoulli",
    "Pytree",
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
    "gen",
    "get_choices",
    "marginal",
    "modular_vmap",
    "normal",
    "tfp_distribution",
    "trace",
]
