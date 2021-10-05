"""Functions for io: Loading data to and from Jax M.D."""

import jax.numpy as jnp
import mdtraj


def load_box(filename):
    """
    Loads initial configuration using the file loader from MDTraj.

    Args:
        filename: String providing the location of the file to load

    Returns:
        Arrays of Coordinates, Velocities and box dimensions.
    """
    traj = mdtraj.load(filename)
    coordinates = traj.xyz[0]
    box = traj.unitcell_lengths[0]
    return jnp.array(coordinates), jnp.array(box)
