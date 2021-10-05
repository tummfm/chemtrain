
from jax_md import space
from jax import ops
import jax.numpy as jnp

from typing import Union

from jax_md.util import Array
Box = Union[float, Array]


def rectangular_boxtensor(box):
    spacial_dim = box.shape[0]
    return ops.index_update(jnp.eye(spacial_dim), jnp.diag_indices(spacial_dim), box)


def init_fractional_coordinates(box):
    if box.ndim != 2:  # we need to transform to box tensor
        box_tensor = rectangular_boxtensor(box)
    else:
        box_tensor = box
    inv_box_tensor = space.inverse(box_tensor)
    scale_fn = lambda R: jnp.dot(R, inv_box_tensor)  # to scale to hypercube
    return box_tensor, scale_fn

# wrapped=False: Particles are not mapped back to original box after each step --> saves compute effort and
#                allows easier time integration as space can be handled without periodic BCs:
#                We can integrate positions unconstrained; In force computation: displacement function ensures that
#                particle distances are still computed correctly
#                For sampled configurations to be used: We can easily remap it back to the original box
#                For wrapped=False: shift function is only adding displacement, not handling any periodicity
def differentiable_periodic(side: Box, wrapped=True):
    # here needs to be 1 instead of box_size due to reparametrization below
    displacement, shift = space.periodic(1., wrapped=wrapped)
    def reparameterized_displacement(Ra, Rb, **kwargs):
        box = side
        if 'box' in kwargs:
            box = kwargs['box']
            if box.ndim == 2:
                box = jnp.diag(box)
        Ra = Ra / box
        Rb = Rb / box
        return displacement(Ra, Rb, **kwargs) * box
    def reparameterized_shift(R, dR, **kwargs):
        box = side
        if 'box' in kwargs:
            box = kwargs['box']
            if box.ndim == 2:
                box = jnp.diag(box)
        return shift(R / box, dR / box) * box
    return reparameterized_displacement, reparameterized_shift
