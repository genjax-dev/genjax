import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from .core import tfp_distribution

tfd = tfp.distributions

bernoulli = tfp_distribution(
    tfd.Bernoulli,
    name="Bernoulli",
)

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    name="Flip",
)

beta = tfp_distribution(
    tfd.Beta,
    name="Beta",
)

normal = tfp_distribution(
    tfd.Normal,
    name="Normal",
)

categorical = tfp_distribution(
    tfd.Categorical,
    name="Categorical",
)

geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)
