import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax.lax import cond

from genjax import bernoulli, ve


def prog():
    x = bernoulli.assume(0.8)
    y = bernoulli.assume(0.2)
    return x + y


print(ve(prog)())
