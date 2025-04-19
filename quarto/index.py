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

import genstudio.plot as Plot
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import treescope
from jax import make_jaxpr
from jax.lax import cond, scan
from jax.numpy import array, sum

from genjax import (
    GFI,
    attach_discretization,
    flip,
    gen,
    normal,
    normal_grid_around_mean,
    normal_reinforce,
    normal_reparam,
    seed,
    sel,
    trace,
)
from genjax import modular_vmap as pjax_vmap
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
# ## Advanced topics: more on the GFI

# %% [markdown]
# ## Experimental: the reflective measure interface
#
# The GFI is a design for probabilistic computation based on "black box"
# objects that satisfy a interface. By "black box", we mean that
# the only thing that one object can know about another is
# that it satisfies the interface (meaning
# it provides access to interface methods).
# This allows Gen [@cusumano-towner_gen_2019]
# to support modular extension, including defining
# new functionality (like usage of neural networks, or external probabilistic
# modules) as "black box" objects that satisfy the interface.
#
# But often the "black box" nature of the GFI is a restricted viewpoint. Several observations inspire a strengthening:
#
# * In GenJAX, probabilistic computation carries
# the strong assumption of _JAX compatibility_.
#
# * Additionally, in GenJAX, _probabilistic computations
#   can share the same intermediate representation_.
#
# * JAX supports a compositional model of staged
#   metaprogramming.
#
# These observations beg the question: what about _stronger
# automation_ which brings these capabilities into the
# foreground?
#
# As motivation, note that the design of the GFI
# says nothing about several strong forms of automation,
# including interval interpretations of probabilistic programs
# for posterior analysis [@beutner_guaranteed_2022; @zaiser_guaranteed_2025],
# type-driven soundness analysis of inference [@lew_trace_2019],
# or inference functionality which requires program analysis
# and transformation [@li_compiling_2024].
#
# These forms of automation are highly important and useful!
# They work by providing language-based algorithms
# which utilize domain-specific languages and program
# transformations or interpretation.
#
# ### Case study: exact enumeration
#
# Consider the _burglar model_ (a classic!) below:

# %%
alarm_likelihood = jnp.array(
    [
        [0.001, 0.29],
        [0.94, 0.95],
    ]
)
john_likelihood = jnp.array([0.05, 0.90])
mary_likelihood = jnp.array([0.01, 0.7])


def int_(v):
    return v.astype(int)


@gen
def burglary():
    bg = int_(flip(0.005) @ "bg")
    eqk = int_(flip(0.001) @ "eqk")
    alarm = int_(flip(alarm_likelihood[bg, eqk]) @ "alarm")
    john = flip(john_likelihood[alarm]) @ "john"
    mary = flip(mary_likelihood[alarm]) @ "mary"


# %% [markdown]
#
# There is no reason to use Monte Carlo in this setting:
# when expressed as a Bayesian network, exact posterior queries
# on `burglary` can be computed using several algorithms,
# including exact enumeration (and, more efficiently, variable elimination).
#
# Let's assume that John and Mary both give us a call,
# we gain access to the unnormalized measure
# induced by conditioning `burglary` using the `RMI.project`
# method:

# %%
meas = burglary.project((), {"john": True, "mary": True})
meas

# %%
log_posterior, (choices, _) = meas.enum(())
log_posterior


# %%
def exact_MAP(*args):
    log_posterior, (choices, _) = args
    idx = jnp.unravel_index(
        jnp.argmax(log_posterior),
        shape=jnp.shape(log_posterior),
    )
    return log_posterior[idx], jtu.tree_map(lambda v: v[idx], choices)


exact_MAP(log_posterior, (choices, _))

# %% [markdown]
# Quick and painless, but how does this work? The trick is
# in the RMI: objects which satisfy the RMI
# participate in interfaces which expose their probabilistic logic
# -- for _enumeration_, the statement is that the logic can be lowered
# to an _enumeration language_, with automation for enumeration.

# %%
make_jaxpr(meas.lower_enum(()))()

# %% [markdown]
# The `enum` language extends JAX with new primitives
# for _assumptions_ (assuming a random variable) and
# _observations_ (observing that a random variable takes a
# value, and accumulating probability mass accordingly).
#
# ### Intuition: compositional lowering interfaces
#
# Another way to understand the RMI is that objects that
# satisfy the RMI participate in compositional
# _lowering_ interfaces: an RMI object may ask another RMI object
# for a representation of its logic. For `enum`, this interface
# has the following (Haskell-like) signature:
#
# ```haskell
# -- LHS: `X` is the type of sample, `R` is the return value
# lower_enum :: RMI[X, R] -> EnumDSL[tuple[X, R]]
# -- RHS: `tuple[X, R]` is the return value from the DSL function.
# ```
#
# where `EnumDSL[...]` is a representation of the logic of the
# `RMI[...]` in the `enum` language.
#
# Combined with JAX's staged metaprogramming, this allows
# zero-cost implementation of brute force exact enumeration
# (provided as automation via the `enum` language).

# %% [markdown]
# ### Discretization
#
# Discretization is a powerful technique, especially when paired with `enum`. The `RMI.discretize` interface
# provides support for a form of _programmable discretization_:

# %%
normal = attach_discretization(
    normal,
    normal_grid_around_mean(2, 50),
)


@gen
def trivial_model():
    x = normal(0.0, 1.0) @ "x"
    v = normal(x, 0.3) @ "v"


# Target specific addresses via `sel("x")`
discretized = trivial_model.discretize((), sel("x"))

# %% [markdown]
# The `attach_discretization` function allows tagging
# distributions with discretization strategies.
# The strategy `normal_grid_around_mean(radius, num_points)` constructs
# a uniform grid around the mean of the `normal`,
# with `radius` and `num_points = radius / eps`.

# %%
# Available for `Fn` (but not for other programs).
discretized.make_jaxpr()

# %% [markdown]
# Shown above, we've _changed_ the source logic of the `RMI`
# object: we swapped `Normal` for `DiscretizedNormal`,
# a discrete (and truncated) representation of `Normal`.
#
# Now, solutions to inference problems for this new model won't
# apply directly to our original model: by introducing
# our discretization -- we've changed the distribution!
# We'll cover _translation_ later, and we'll see how to convert
# solutions to discretized problems back to the original model.
#
# First, a few lines to construct a MAP using the RMI & `enum`.

# %%
meas = discretized.project((), {"v": 2.0})
exact_MAP(*meas.enum(()))

# %% [markdown]
# ### Interactive inference programming

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
