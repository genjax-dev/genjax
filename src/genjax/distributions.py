import jax.numpy as jnp
from jax import vmap
from tensorflow_probability.substrates import jax as tfp

from genjax.core import (
    BB,
    RR,
    Callable,
    Distribution,
    Finite,
    Shaped,
    X,
    distribution,
    tfp_distribution,
    wrap_logpdf,
    wrap_sampler,
)

tfd = tfp.distributions

bernoulli = tfp_distribution(
    tfd.Bernoulli,
    discretization=lambda logits: bernoulli,
    tryper=lambda sample_shape: BB(sample_shape),
    name="Bernoulli",
)

flip = tfp_distribution(
    lambda p: tfd.Bernoulli(probs=p, dtype=jnp.bool_),
    discretization=lambda p: flip,
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

###################
# Discretizations #
###################


def attach_discretization(
    d: Distribution[X],
    strategy: Callable[..., Distribution[X]],
):
    return distribution(
        d.sample,
        d.logpdf,
        discretization=strategy,
        tryper=d.tryper,
        name=d.name,
    )


def normal_grid_around_mean(radius, num_points):
    def strategy(mu, sigma):
        def keyful_sampler(key, mu, sigma, sample_shape=(), **kwargs):
            uniform_x_range = jnp.arange(-radius, radius, 1 / num_points)
            x_range = uniform_x_range * sigma
            logits = vmap(normal.logpdf, in_axes=(0, None, None))(
                x_range + mu, mu, sigma
            )
            idx = tfd.Categorical(logits=logits).sample(
                seed=key, sample_shape=sample_shape
            )
            return x_range[idx]

        def logpdf(v, mu, sigma):
            uniform_x_range = jnp.arange(-radius, radius, 1 / num_points)
            x_range = uniform_x_range * sigma
            logits = vmap(normal.logpdf, in_axes=(0, None, None))(
                x_range + mu, mu, sigma
            )
            comparisons = v == x_range
            idx = jnp.argmax(comparisons)
            return categorical.logpdf(idx, logits=logits)

        def support(mu, sigma):
            x_range = jnp.arange(-radius, radius, 1 / num_points)
            return x_range + mu

        def tryper(mu, sigma):
            x_range = jnp.arange(-radius, radius, 1 / num_points)
            return Shaped((), Finite(len(x_range)))

        return distribution(
            wrap_sampler(
                keyful_sampler,
                name="DiscretizedNormal",
                support=support,
            ),
            wrap_logpdf(logpdf),
            support=support,
            tryper=tryper,
            name="DiscretizedNormal",
        )

    return strategy


normal = tfp_distribution(
    tfd.Normal,
    discretization=normal_grid_around_mean(2, 500),
    tryper=lambda mu, sigma: Shaped(jnp.shape(mu), RR()),
    name="Normal",
)
