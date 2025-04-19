import jax.numpy as jnp

from genjax import (
    attach_discretization,
    gen,
    normal,
    normal_grid_around_mean,
)

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
