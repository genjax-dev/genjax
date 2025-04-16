# %% [markdown]
#
# <br>
# <p align="center">
# <img width="500px" src="./assets/logo.png"/>
# </p>
#
# (**Probabilistic programming language**) GenJAX is a probabilistic programming
# language (PPL): a system which provides automation for writing programs
# which perform computations on probability distributions, including sampling,
# variational approximation, gradient estimation for expected values, and more.
#
# (**With programmable inference**) The design of GenJAX is centered
# on _programmable inference_: automation which allows users to express and
# customize Bayesian inference algorithms (algorithms for computing with
# posterior distributions: "_x_ affects _y_, and I observe _y_, what are my
# new beliefs about _x_?"). Programmable inference includes advanced forms
# of Monte Carlo and variational inference methods.
#
# GenJAX's automation is based on two key concepts: _parallel generative functions_ (GenJAX's version of probabilistic programs) and _traces_ (samples from probabilistic programs). GenJAX provides:
#
# * Modeling language automation for constructing complex probability distributions from pieces
#
# * Inference automation for constructing Monte Carlo samplers using convenient idioms (programs expressed by creating and modifying traces), and [variational inference automation](https://dl.acm.org/doi/10.1145/3656463)([artifact](https://github.com/femtomc/programmable-vi-pldi-2024)) using [new extensions to automatic differentation for expected values](https://dl.acm.org/doi/10.1145/3571198).
#
# (**Fully vectorized & compatible with JAX**) All of GenJAX's automation is
# compatible with JAX, implying that any program written in GenJAX can
# be `vmap`'d and `jit` compiled.

# %%
# | code-summary: Prelude
# | code-fold: true
import genstudio.plot as Plot
import jax.numpy as jnp
import jax.random as jrand
import treescope
from jax import jit, make_jaxpr
from jax.lax import cond
from jax.numpy import array, sum, zeros

from genjax import Importance, gen, marginal, normal, seed, trace
from genjax import modular_vmap as pjax_vmap
from genjax import modular_vmap as vmap
from genjax.adev import Dual, expectation, flip_enum

treescope.basic_interactive_setup()


def dot_plot(x, y):
    points = list(zip(x, y))
    return (
        Plot.dot(points, fill="cyan", opacity=0.7, r=2.0)
        + Plot.frame()
        + Plot.aspectRatio(1.5)
    )


# %% [markdown]
# ## Modeling & inference with GenJAX

# %% [markdown]
# GenJAX supports convenient syntax to express programs
# that denote probability distributions.
# The program below defines a polynomial regression model with
# a prior over coefficients (`"alpha"`).
#
# **(Example: polynomial regression model)**


# %%
# | column: margin
# | fig-cap: "Samples from a probabilistic program defining a distribution which can be used as a regression model. Points sampled noisily near polynomial curves."
# A regression model.
@gen
def regression(x):
    # Addresses like "alpha" denote random variables.
    coefficients = normal.repeat(n=3)(0.0, 1.0) @ "alpha"

    # The `@gen` decorator creates a probabilistic program
    # from JAX-compatible Python source code.
    @gen
    def generate_y(x):
        basis_value = array([1.0, x, x**2])
        polynomial_value = sum(
            basis_value * coefficients,
        )
        y = normal(polynomial_value, 0.2) @ "v"
        return y

    # Probabilistic programs can be transformed
    # into new ones: here, `generate_y.vmap` creates
    # a new probabilistic program which applies itself
    # independently to the elements of `x`.
    return generate_y.vmap(in_axes=0)(x) @ "y"


# Sample a curve.
xrange = jnp.linspace(-1, 1, 100)
y_samples = regression.simulate((xrange,)).get_retval()
dot_plot(xrange, y_samples)

# %% [markdown]
# The `@gen` decorator creates a _parallel generative function_, a datatype which
# implements a probabilistic interface which exposes sampling and density
# computation called _the generative function interface_, or GFI for short.
# GenJAX's GFI consists of 3 methods (`simulate`, `assess`, and `update`), which are shown below.

# %% [markdown]
# ### `GFI.simulate`

# %%
# Sample a trace.
sample_curve = regression.simulate((xrange,))
sample_curve

# %% [markdown]
# ### `GFI.assess`

# %%
# Evaluate the density of random choices, and
# the return value given those choices.
choices = sample_curve.get_choices()
density, retval = regression.assess((xrange,), choices)
density

# %% [markdown]
# ### `GFI.update`

# %%
# Change a trace by changing the probabilistic program
# which generated it, or the values of random choices,
# and compute a density ratio for the change.
new_choices = {"alpha": jnp.array([0.3, 1.0, 2.0])}
new_trace, w, _, _ = sample_curve.update((xrange,), new_choices)
new_trace["alpha"]

# %% [markdown]
# ## Marginalization of random choices

# %% [markdown]
# Marginalization provides a way to hide random choices. Exact marginalization involves computing integrals, which is intractable. GenJAX supports _pseudo_-marginalization [via stochastic probabilities](https://dl.acm.org/doi/abs/10.1145/3591290). To support _pseudo_-marginalization, constructing a marginal requires that you provide a proposal:


# %%
@gen
def model():
    x = normal(0.0, 10.0) @ "x"
    y = normal(0.0, 10.0) @ "y"
    rs = x**2 + y**2
    z = normal(rs, 0.1 + (rs / 100.0)) @ "z"


@gen
def proposal(*args):
    x = normal(0.0, 10.0) @ "x"
    y = normal(0.0, 10.0) @ "y"


# Tell me the model, what address to marginalize to,
# and a proposal for the other addresses given the
# remaining one.
# vmap(marginal(model, Importance(proposal, 5), "z").simulate, axis_size=1000)(())["z"]

# %% [markdown]
# ## Automatic differentiation of expected values

# %% [markdown]
# GenJAX also exposes functionality to support _unbiased gradient estimation_ of expected value objectives.


# %%
@expectation
def flip_exact_loss(p):
    b = flip_enum(p)
    return cond(
        b,
        lambda _: 0.0,
        lambda p: -p / 2.0,
        p,
    )


flip_exact_loss

# %% [markdown]
# Using the `@expectation` decorator creates an `Expectation` object, which supports `jvp_estimate` and `grad_estimate` methods.
#
# For the above `@expectation`-decorated program, the meaning corresponds to the following expectation:
# $$\mathcal{L}(p) := \mathbb{E}_{v \sim Ber(\cdot; p)}[\textbf{if}~v~\textbf{then}~0.0~\textbf{else}~\frac{-p}{2}]$$
# which we can evaluate analytically:
# $$\mathcal{L}(p) = (1 - p) * \frac{-p}{2} = \frac{p^2 - p}{2}$$
# and whose $\nabla_p$ we can also evaluate analytically:
# $$\nabla_p\mathcal{L}(p) = p - \frac{1}{2}$$
#
# The methods `jvp_estimate` and `grad_estimate` provide access to _gradient estimators_ for the expected value objective $\mathcal{L}(p)$.
#
# In the `@expectation`-decorated program, users can inform the automation _what gradient estimator they'd like to construct_ by using samplers equipped with estimation strategies (`flip_enum`, is a Bernoulli sampler with an annotation which directs ADEV's automation to use enumeration, exactly evaluating the expectation, to construct a gradient estimator).

# %%
# Compare ADEV's derived derivatives with the exact value.
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    p_dual = flip_exact_loss.jvp_estimate(Dual(p, 1.0))
    treescope.show(p_dual.tangent - (p - 0.5))

# %% [markdown]
# ## Programmable variational inference
#
# ## Case study: probabilistic atavising the _Game of Life_
#
# ## Under the hood: PJAX

# %% [markdown]
# All of GenJAX's functionality is constructed on top of a vectorizable probabilistic intermediate representation called PJAX.
#
# PJAX is a modular extension to JAX that explicitly represents operations on probability distributions as first class primitives.


# %%
def sampler():
    v = normal.sample(0.0, 1.0)
    return jnp.exp(v)


make_jaxpr(sampler)()

# %% [markdown]
# PJAX supports a `vmap`-like transformation, which is an extension of `jax.vmap` to natively work with operations on probability distributions.


# %%
make_jaxpr(pjax_vmap(sampler, axis_size=10))()

# %% [markdown]
# GenJAX's GFI is implemented in terms of PJAX:

# %%
make_jaxpr(regression.simulate)((xrange,))
