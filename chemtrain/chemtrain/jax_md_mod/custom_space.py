"""Custom functions simplifying the handling of fractional coordinates."""
from typing import Union, Tuple, Callable

from jax_md import space, util
import jax.numpy as jnp

Box = Union[float, util.Array]


def _rectangular_boxtensor(box: Box) -> Box:
    """Transforms a 1-dimensional box to a 2D box tensor."""
    spatial_dim = box.shape[0]
    return jnp.eye(spatial_dim).at[jnp.diag_indices(spatial_dim)].set(box)


def init_fractional_coordinates(box: Box) -> Tuple[Box, Callable]:
    """Returns a 2D box tensor and a scale function that projects positions
    within a box in real space to the unit-hypercube as required by fractional
    coordinates.

    Args:
        box: A 1 or 2-dimensional box

    Returns:
        A tuple (box, scale_fn) of a 2D box tensor and a scale_fn that scales
        positions in real-space to the unit hypercube.
    """
    if box.ndim != 2:  # we need to transform to box tensor
        box_tensor = _rectangular_boxtensor(box)
    else:
        box_tensor = box
    inv_box_tensor = space.inverse(box_tensor)
    scale_fn = lambda positions: jnp.dot(positions, inv_box_tensor)
    return box_tensor, scale_fn
