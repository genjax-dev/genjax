# %% [markdown]
# ---
# title: "The Generative Cookbook"
# subtitle: "A wondrous guide designed to shepherd the (human or robotic) enthusiast in the usage of the magnanimous"
# bibliography: references.bib
# copyright: "Copyright McCoy Reynolds Becker & MIT Probabilistic Computing Project. All Rights Reserved."
# ---

# %% [markdown]
# <p align="center">
# <img width="500px" src="./assets/logo.png"/>
# </p>
#
# (**Probabilistic programming language**) GenJAX is a probabilistic programming
# language (PPL): a system which provides automation for writing programs
# which perform computations on probability distributions, including sampling,
# variational approximation, gradient estimation, and more.
#
# (**With programmable inference**) The design of GenJAX is centered
# on _programmable inference_ [@mansinghka_probabilistic_2018]: automation which allows users to express and
# customize Bayesian inference algorithms (algorithms for computing with
# posterior distributions: "_x_ affects _y_, and I observe _y_, what are my
# new beliefs about _x_?"). Programmable inference supports advanced forms
# of Monte Carlo and variational inference methods.
#
# Following [@cusumano-towner_gen_2019], GenJAX's automation is based on two key concepts: _parallel generative functions_ (GenJAX's version of probabilistic programs) and _traces_ (samples from probabilistic programs). GenJAX provides:
#
# * Modeling language automation for constructing complex probability distributions.
#
# * Inference automation for constructing Monte Carlo samplers
# and variational inference algorithms, including advanced
# algorithms which utilize marginalization, or complex
# variational objectives and gradient estimation strategies.
#
# (**Fully vectorized & compatible with JAX**) All of GenJAX's automation is
# compatible with JAX, implying that any program written in GenJAX can
# be `vmap`'d and `jit` compiled.

# %%
# | code-summary: Prelude
# | code-fold: true
from functools import partial

import genstudio.plot as Plot
import jax.numpy as jnp
import jax.random as jrand
import treescope
from jax import jit, make_jaxpr
from jax.lax import cond, scan
from jax.numpy import array, sum, zeros

from genjax import (
    GFI,
    Importance,
    gen,
    marginal,
    normal,
    normal_reinforce,
    normal_reparam,
    seed,
    trace,
)
from genjax import modular_vmap as pjax_vmap
from genjax import modular_vmap as vmap
from genjax.adev import Dual, expectation, flip_enum

treescope.basic_interactive_setup(autovisualize_arrays=False)


def dot_plot(x, y, aspect_ratio=None):
    points = list(zip(x, y))
    plot = (
        Plot.dot(
            points, fill="black", stroke="white", opacity=1.0, strokeWidth=0.5, r=2.5
        )
        + Plot.frame()
    )
    plot = plot + Plot.aspectRatio(aspect_ratio) if aspect_ratio else plot
    return plot


# %% [markdown]
# ## Modeling & inference with GenJAX
#
# Writing a probabilistic model involves _telling a story
# about how the data might have been generated_. Inference
# is the process of _inverting that story_, to attempt
# to construct a representation of the _elements of the story_
# which coheres with the data.
#
# Another way to think about it: you're authoring a world whose
# behavior can give rise to the data, and we're exploring
# queries like "I see the data, now what was the behavior,
# probably?"
#
# Even a regression model follows this pattern.
# GenJAX supports convenient syntax to express programs
# that denote probability distributions (over these worlds!).
# The program below defines a polynomial regression model with
# a prior over coefficients (`"alpha"`).


# %%
# | column: margin
# | fig-cap: "Samples from a probabilistic program defining a distribution which can be used as a regression model. Points sampled noisily near polynomial curves."
# "Authoring a world" in GenJAX:
# * convenient syntax for denoting random variables
# * modeling constructs to build larger distributions from
#   small ones
# * compatibility with JAX computations
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
x_range = jnp.linspace(-1, 1, 100)
y_samples = regression.simulate((x_range,)).get_retval()
dot_plot(x_range, y_samples)

# %% [markdown]
# The `@gen` decorator creates a _parallel generative function_,
# a probabilistic program datatype which
# implements an interface that provides automation
# for sampling and density computation.
# The interface is called _the generative function interface_, or GFI for short.

# %%
isinstance(regression, GFI)

# %% [markdown]
# GenJAX's GFI consists of 3 methods (`simulate`, `assess`, and `update`), which are shown below.

# %% [markdown]
# ### `GFI.simulate`

# %%
# Sample a trace.
trace = regression.simulate((x_range,))
trace

# %% [markdown]
# A _trace_ is a recording of the execution of a
# parallel generative function. It contains the random choices
# which were sampled during the execution, as well as other
# data associated with the execution (the arguments, the
# return value).
#
# Importantly, a trace also contains a quantity called _the score_, which is a recording of $1 / P(\text{random choices})$.

# %%
trace.get_score()

# %% [markdown]
# ### `GFI.assess`

# %%
# Evaluate the density of random choices, and
# the return value given those choices.
choices = trace.get_choices()
density, retval = regression.assess((x_range,), choices)
density

# %% [markdown]
# The `assess` method gives you access to
# $P(\text{random choices})$. `simulate` and `assess`
# pair together to allow you to implement _importance samplers_,
# which we'll see in a moment.

# %% [markdown]
# ### `GFI.update`

# %%
# Reweight a trace by changing the arguments to the
# probabilistic program which generated it, or
# change the values of random choices (or both!),
# and compute a density ratio for the change.
new_choices = {"alpha": jnp.array([0.3, 1.0, 2.0])}
new_trace, w, _, _ = trace.update((x_range,), new_choices)
new_trace["alpha"]

# %% [markdown]
# The `update` method allows you to modify or reweight an
# existing trace. This method is useful when you wish
# to implement algorithms whose logic involves making change
# to samples, like Markov chain Monte Carlo (MCMC), or
# variants of sequential Monte Carlo (SMC).

# %% [markdown]
# ### Why the GFI?
#
# The methods of the GFI are focused on the expression
# of approximations
# to inference problems by using Monte Carlo sampling and
# properly-weighted approximations. This sentence is strongly
# already an "if you know you know" description: the gist is,
# inference problems are often intractable for analytical
# methods, and Monte Carlo is a broad class of approximation
# methods. The GFI focuses on automation support for a subclass
# of Monte Carlo that the creators of GenJAX have found to be
# particularly useful in their own work.

# %% [markdown]
# ## Marginalization of random choices

# %% [markdown]
# Marginalization provides a way to hide random choices. Exact marginalization involves computing integrals, which is often intractable for complex distributions. GenJAX supports _pseudo_-marginalization via stochastic probabilities [@lew_probabilistic_2023]. To support _pseudo_-marginalization, constructing a marginal requires that you provide a proposal:


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
# GenJAX also exposes functionality to support _unbiased gradient estimation_ of expected value objectives [@lew_adev_2023].


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
# In the `@expectation`-decorated program, users can inform the automation _what gradient estimator they'd like to construct_ by using samplers equipped with estimation strategies (`flip_enum` is a Bernoulli sampler with an annotation which directs ADEV's automation to use enumeration, exactly evaluating the expectation, to construct a gradient estimator).

# %%
# Compare ADEV's derived derivatives with the exact value.
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    p_dual = flip_exact_loss.jvp_estimate(Dual(p, 1.0))
    treescope.show(p_dual.tangent - (p - 0.5))


# %% [markdown]
# ## Programmable variational inference
#
# ADEV provides access to unbiased gradient estimators of expected
# value objectives, and expected value objectives occur often in
# _variational inference_, where users define optimization problems
# over spaces of distributions, often using some notion of closeness
# defined via an expected value.
#
# For instance, given the density of an unnormalized measure $P$ and parametric variational approximation $Q(\cdot; \theta)$, the _evidence lower bound_ objective
# $$\mathbb{E}_{x \sim Q(\cdot; \theta)}[\log P(x) - \log Q(x; \theta)]$$
# is often used to define inference problems as optimization, where $\theta \mapsto \theta'$ involves maximizing the objective, squeezing a KL divergence between the normalized version of $P$ and $Q$.


# %%
# | column: margin
# | fig-cap: "The theta parameter over the course of training with a REINFORCE gradient estimator."
@gen
def variational_model():
    x = normal(0.0, 1.0) @ "x"
    y = normal(x, 0.3) @ "y"


@gen
def variational_family(theta):
    # Use distribution with a gradient strategy!
    x = normal_reinforce(theta, 1.0) @ "x"


@expectation
def elbo(family, data: dict, theta):
    # Use GFI methods to structure the objective function!
    tr = family.simulate((theta,))
    q = tr.get_score()
    p, _ = variational_model.assess((), {**data, **tr.get_choices()})
    return p - q


def optimize(family, data, init_theta):
    def update(theta, _):
        _, _, theta_grad = elbo.grad_estimate(family, data, theta)
        theta += 1e-3 * theta_grad
        return theta, theta

    final_theta, intermediate_thetas = scan(
        update,
        init_theta,
        length=500,
    )
    return final_theta, intermediate_thetas


# `seed`: seed any sampling with fresh random keys.
# (GenJAX will send you a warning if you need to do this)
_, thetas = seed(optimize)(
    jrand.key(1),
    variational_family,
    {"y": 3.0},
    0.01,
)
dot_plot(jnp.arange(500), thetas)

# %% [markdown]
# ### What's programmable about it?
#
# In programmable variational inference
# [@becker_probabilistic_2024], users are allowed to change
# their objective function (by writing programs which
# denote objectives), and they are also allowed to change
# the unbiased gradient estimator strategy for the objective.
#
# For instance, instead of using `normal_reinforce`, we could use `normal_reparam`.


# %%
# | column: margin
# | fig-cap: "The theta parameter over the course of training with a reparametrization gradient estimator."
@gen
def reparam_variational_family(theta):
    # Use distribution with a gradient strategy!
    x = normal_reparam(theta, 1.0) @ "x"


_, thetas = seed(optimize)(
    jrand.key(1),
    reparam_variational_family,
    {"y": 3.0},
    0.01,
)
dot_plot(jnp.arange(500), thetas)

# %% [markdown]
# which leads to a significantly less noisy training process.
# Trying out different objective functions, and
# unbiased gradient estimators is an important part of
# designing variational inference algorithms, and
# programmable variational inference tries to make this
# more convenient.

# %% [markdown]
# ## Programmable Monte Carlo
#
# GenJAX's GFI is designed to provide users with the ability
# to construct customized Monte Carlo algorithms,
# producing better quality approximations in difficult inference
# settings. Paired with expressive modeling syntax, GenJAX
# allows users to construct complex distributions concisely,
# and develop effective sampling (and variational)
# approximations.

# %% [markdown]
# ### Case study: inferencing the _Game of Life_
#
# The _Game of Life_ is a computational system which gives rise
# to a bewildering array of interesting phenomenon. One
# interesting theoretical question: given a fixed Life
# configuration, is it possible to find a configuration that
# precedes it? This is known as _atavising_. Exact atavisation
# is a complex, high-dimensional discrete search problem.
# By relaxing the problem to _noisily_ atavise (by adding a
# bit of probability), we can construct algorithms that
# build approximate predecessor states almost instantaneously
# on modern hardware.

# %% [markdown]
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
@gen
def model():
    x = normal(0.0, 1.0) @ "x"
    y = normal(0.0, 1.0) @ "y"
    return x + y


make_jaxpr(model.simulate)(())

# %% [markdown]
# Even ADEV's unbiased gradient estimator programs are implemented in terms of PJAX:


# %%
@expectation
def loss(mu):
    x = normal_reparam.sample(mu, 1.0)
    return x**2


make_jaxpr(loss.grad_estimate)(0.1)

# %% [markdown]
# ## Advanced topics: more on the GFI

# %% [markdown]
# ## Future work & sharp edges
#
# The creators of GenJAX have intended for GenJAX to be a useful
# and somewhat general design for GPU-accelerated
# probabilistic programming! Of course, that's not always
# possible: there are known sharp edges when using GenJAX's
# automation, which we tabulate below.
#
# ### Known incompatibilities between features
#
# #### `vmap` within ADEV programs
#
# The semantics of `vmap` _within_ ADEV programs is a
# direction of future research. `vmap` is a
# vectorization transformation that converts primitives
# into batched versions of themselves. For ADEV's samplers,
# the primitives come equipped with gradient estimation
# strategies. Therefore, using `vmap` on code which contains
# ADEV samplers requires care. Not all primitives support
# "batched" gradient estimation strategies.
#
# #### `scan` within ADEV programs
#
# `scan` is a second-order control flow primitive which supports
# iteration behavior within JAX programs. ADEV's automation
# has not yet been extended to work with `scan`. There are
# levels to `scan` compatibility, which we briefly mention below:
#
# * (Not done, but should be straightforward) "deterministic" `scan` (no ADEV samplers _within_ `scan`)
#
# * (Possibly a bit of work, but also seemingly straightforward) `scan` with ADEV samplers _which invoke the continuation in tail position_
#
# * (Unknown, possibly very hard) `scan` with ADEV samplers that invoke the continuation multiple times, or not in tail position.
