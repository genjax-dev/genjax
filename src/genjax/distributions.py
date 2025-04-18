import jax.numpy as jnp
from jax import vmap
from tensorflow_probability.substrates import jax as tfp

from .core import distribution, tfp_distribution

tfd = tfp.distributions

bernoulli = tfp_distribution(
    tfd.Bernoulli,
    name="Bernoulli",
)

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    discretization=lambda p, size: (p, flip),
    name="Flip",
)

beta = tfp_distribution(
    tfd.Beta,
    name="Beta",
)

categorical = tfp_distribution(
    tfd.Categorical,
    name="Categorical",
)

geometric = tfp_distribution(
    tfd.Geometric,
    name="Geometric",
)


def normal_discretization(args, size):
    (mu, sigma) = args
    x_range = jnp.arange(-sigma, sigma, 1 / size)
    logits = vmap(normal.logpdf, in_axes=(0, None, None))(x_range + mu, mu, sigma)
    return (logits, x_range + mu), labeled_cat


normal = tfp_distribution(
    tfd.Normal,
    discretization=normal_discretization,
    name="Normal",
)

#######################
# Labeled categorical #
#######################


def labeled_categorical_sampler(key, logits, vs, sample_shape=()):
    idx = tfd.Categorical(logits=logits).sample(seed=key, sample_shape=sample_shape)
    return jnp.array(vs)[idx]


def labeled_categorical_logpdf(v, logits, vs):
    comparisons = v == vs
    idx = jnp.argmax(comparisons)
    return categorical.logpdf(idx, logits=logits)


def labeled_categorical_support(logits, vs):
    return vs


labeled_cat = distribution(
    labeled_categorical_sampler,
    labeled_categorical_logpdf,
    support=labeled_categorical_support,
    name="LabeledCategorical",
)
