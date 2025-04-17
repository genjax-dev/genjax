from functools import partial

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from .core import tfp_distribution

tfd = tfp.distributions

bernoulli = tfp_distribution(
    partial(tfd.Bernoulli, dtype=jnp.bool_),
    name="Bernoulli",
    support=lambda *args: jnp.array([True, False]),
)

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    name="Flip",
    support=lambda *args: jnp.array([True, False]),
)

categorical = tfp_distribution(
    tfd.Categorical,
    name="Categorical",
    support=lambda *args, **kwargs: jnp.arange(len(kwargs["probs"])),
)

beta = tfp_distribution(
    tfd.Beta,
    name="Beta",
)

normal = tfp_distribution(
    tfd.Normal,
    name="Normal",
)


geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)
