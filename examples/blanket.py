import jax.numpy as jnp

from genjax import (
    attach_discretization,
    gen,
    normal,
    normal_grid_around_mean,
    sel,
)

normal = attach_discretization(
    normal,
    normal_grid_around_mean(1, 20),
)


@gen
def model():
    m = normal(0.1, 2.0) @ "m"
    v = normal(0.3, 3.0) @ "v"
    x = normal(jnp.exp(v), m) @ "y"
    q = normal(jnp.exp(x), 3.0) @ "q"
    return v


model = model.discretize((), sel("v"))
measure_program = model.project((), {"y": 3.0})
tr, w = measure_program.generate(())
fn = tr.blanket(sel("v"))
print("v" in (~sel("v") ^ fn.addresses(())))
print("y" in (~sel("v") ^ fn.addresses(())))
