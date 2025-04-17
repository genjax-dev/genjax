from jax.lax import cond

from genjax.adev import Dual, expectation, flip_enum


@expectation
def flip_exact_loss(p):
    b = flip_enum(p)
    return cond(
        b,
        lambda _: 0.0,
        lambda p: -p / 2.0,
        p,
    )


for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    p_dual = flip_exact_loss.jvp_estimate(Dual(p, 1.0))
    print((p_dual.tangent, p - 0.5))
