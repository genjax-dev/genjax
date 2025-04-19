import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates import jax as tfp

from .core import (
    GFI,
    Any,
    Fn,
    Generic,
    Pytree,
    R,
    Trace,
    Weight,
    X,
    get_choices,
    modular_vmap,
)

tfd = tfp.distributions

############
# Marginal #
############


class Algorithm(Generic[X], GFI[X, X]):
    def update(
        self,
        args_,
        tr: Trace[X, X],
        x_: X,
    ) -> tuple[Trace[X, X], Weight, X, X]:
        log_density_, _ = self.assess(args_, x_)
        return (
            Trace(self, args_, x_, x_, -log_density_),
            log_density_ + tr.get_score(),
            x_,
            tr.get_retval(),
        )


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
        idx = categorical.sample(ws)
        v = jtu.tree_map(lambda x: x[idx], choices)
        Z = logsumexp(ws) - jnp.log(self.K)
        return Trace(self, args, v, v, Z)

    def assess(self, args, x: X) -> tuple[Weight, X]:
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
            tr.get_score() + weight,
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
        return weight + tr.get_score(), x

    def update(
        self,
        args_,
        tr: Trace[X, X],
        x_: X,
    ) -> tuple[Trace[X, X], Weight, X, X]:
        log_density_, _ = self.assess(args_, x_)
        return (
            Trace(self, args_, x_, x_, -log_density_),
            log_density_ + tr.get_score(),
            x_,
            tr.get_retval(),
        )


def marginal(
    gen_fn: Fn[R],
    alg: Algorithm[dict[str, Any]],
    addr: str,
) -> Marginal[R, Any]:
    return Marginal(gen_fn, alg, addr)
