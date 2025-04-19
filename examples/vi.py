import jax.random as jrand
from jax.lax import scan

from genjax import expectation, gen, normal, normal_reinforce, seed


@gen
def variational_model():
    x = normal(0.0, 1.0) @ "x"
    y = normal(x, 0.3) @ "y"


@gen
def variational_family(theta):
    # Use distribution with a gradient strategy!
    x = normal_reinforce(theta, 1.0) @ "x"


@expectation
def elbo(data: dict, theta):
    # Use GFI methods to structure the objective function!
    tr = variational_family.simulate((theta,))
    q = tr.get_score()
    p, _ = variational_model.assess((), {**data, **tr.get_choices()})
    return p - q


def optimize(data, init_theta):
    def update(theta, _):
        _, theta_grad = elbo.grad_estimate(data, theta)
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
_, thetas = seed(optimize)(jrand.key(1), {"y": 3.0}, 0.01)
print(thetas)
