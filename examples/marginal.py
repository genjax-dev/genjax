from jax import make_jaxpr

from genjax import Importance, gen, marginal, normal


@gen
def model():
    @gen
    def submodel():
        x = normal(0.0, 1.0) @ "x"
        y = normal(2.0, 1.0) @ "y"
        return x, y

    x, y = submodel.T() @ "inner"
    z = normal(x**2 + y**2, 0.3) @ "z"


@gen
def proposal(*args):
    x = normal(0.0, 1.0) @ "z"


print(
    make_jaxpr(
        marginal(
            model,
            Importance(proposal, 20),
            "inner",
        ).assess
    )((), {"x": 3.0, "y": 2.0})
)
