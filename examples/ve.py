import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import make_jaxpr
from jax.lax import cond

from genjax import bernoulli, enum, modular_vmap


def prog():
    x = bernoulli.rv(probs=0.5)
    y = bernoulli.rv(probs=0.3)
    p = cond(x, lambda: 0.3, lambda: 0.6)
    return p > 0.5


x, score = enum(prog)()
print(jax.scipy.special.logsumexp(score))
