import jax
from jax.lax import cond

from genjax import bernoulli, enum


def prog():
    x = bernoulli.rv(probs=0.5)
    y = bernoulli.rv(probs=0.3)
    p1 = cond(x, lambda: 0.3, lambda: 0.6)
    p2 = cond(y, lambda: 0.1, lambda: 0.8)
    return p1 + p2 > 0.5


x, score = enum(prog)()
print(jax.scipy.special.logsumexp(score))
