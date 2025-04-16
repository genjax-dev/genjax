import jax.numpy as jnp
from jax import make_jaxpr
from jax.numpy import array, sum, zeros

from genjax import gen, normal, trace


@gen
def generate_y(x):
    coefficients = normal.repeat(n=3)(0.0, 1.0) @ "alpha"
    basis_value = array([1.0, x, x**2])
    polynomial_value = sum(
        basis_value * coefficients,
    )
    y = normal(polynomial_value, 0.3) @ "v"
    return y


# A regression model.
regression = generate_y.vmap()

print(
    make_jaxpr(regression.assess)(
        (jnp.ones(3),),
        {"alpha": zeros((3, 3)), "v": jnp.ones(3)},
    )
)
