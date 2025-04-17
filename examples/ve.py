import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.lax import cond

from genjax import bernoulli, enum


def prog():
    x = bernoulli.assume(0.8)
    y = bernoulli.assume(0.3)
    p = cond(x, lambda: 0.3, lambda: 0.6)
    return bernoulli.observe(x, probs=p)


print(enum(prog)())
