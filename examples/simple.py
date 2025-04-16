from jax import make_jaxpr

from genjax import gen, get_choices, modular_vmap, normal


@gen
def model(mu):
    v = normal(mu, 1.0) @ "x"
    x = normal(v, 1.0) @ "y"
    return x


tr = model.simulate((1.0,))
chm = get_choices(tr)
w_, _ = model.assess((1.0,), chm)
print(w_ + tr.get_score())

tr_, w, r, _ = model.update(
    (2.0,),
    tr,
    {"x": get_choices(tr)["x"], "y": 3.0},
)
tr, w_, r, _ = model.update(
    (1.0,),
    tr_,
    {"x": get_choices(tr)["x"], "y": get_choices(tr)["y"]},
)
print(w_ + w)

print(
    make_jaxpr(model.update)(
        (1.0,),
        tr_,
        {"x": get_choices(tr)["x"], "y": get_choices(tr)["y"]},
    )
)


tr = modular_vmap(model.simulate, in_axes=(None,), axis_size=50)((1.0,))
print(tr.get_score())
