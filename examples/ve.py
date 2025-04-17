# %%
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import make_jaxpr
from jax.lax import cond

from genjax import bernoulli, enum, modular_vmap


def prog():
    x = bernoulli.assume(0.8)
    p = cond(x, lambda: 0.3, lambda: 0.6)
    return x, bernoulli.observe(x, probs=p)


# %%
v = jnp.array([True, False])
probs = jnp.array([0.3, 0.6])
print(jax.vmap(bernoulli.observe)(v, probs))
