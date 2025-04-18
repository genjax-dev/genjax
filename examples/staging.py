import jax.numpy as jnp
from jax import make_jaxpr

from genjax import enum, flip, gen, normal, trace
from genjax import modular_vmap as vmap


@gen
def model():
    v = normal(0.3, 1.0) @ "v"
    x = normal(v, 1.0) @ "y"
    return v


print(model.make_jaxpr())
new_args, new_model = model.discretize((), 500)
print(new_model.make_jaxpr(*new_args))
measure_program = new_model.project(new_args, {"y": 3.0})
print(make_jaxpr(measure_program)())
scores, (x, _) = enum(measure_program)()
print(x["v"][jnp.argmax(scores)])
