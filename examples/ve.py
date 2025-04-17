import jax
from jax.lax import cond

from genjax import bernoulli, enum


def prog():
    x = bernoulli.rv(probs=0.5)
    y = bernoulli.rv(probs=0.3)
    p = cond(x, lambda: 0.3, lambda: 0.6)
    return p > 0.5


x, score = enum(prog)()
print(jax.scipy.special.logsumexp(score))
