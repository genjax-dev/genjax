import jax.numpy as jnp
from jax import make_jaxpr

from genjax import (
    attach_discretization,
    enum,
    flip,
    gen,
    normal,
    normal_grid_around_mean,
    trace,
)
from genjax import modular_vmap as vmap

normal = attach_discretization(
    normal,
    normal_grid_around_mean(3, 1000),
)


@gen
def model():
    v = normal(0.3, 3.0) @ "v"
    x = normal(v, 0.3) @ "y"
    return v


measure_program = model.project((), {"y": 3.0})
discretized = measure_program.discretize((), "v")
scores, (v, _) = discretized.enum(())
print(v["v"][jnp.argmax(scores)])
