"""Functions for io: Loading data to and from Jax M.D."""

import MDAnalysis
import jax.numpy as jnp
import mdtraj


def load_configuration(file):
    """
    Loads initial configuration using the file loader from MDAnalysis.

    Args:
        file: String providing the location of the file to load

    Returns:
        Arrays of Coordinates, Velocities and box dimensions
    """
    universe = MDAnalysis.Universe(file, file)  # reads in A; we convert it to nm
    coordinates = universe.atoms.positions * 0.1
    if hasattr(universe.atoms, 'velocities'):
        velocities = universe.atoms.velocities * 0.1
    else:
        velocities = jnp.zeros_like(coordinates)
    box = universe.dimensions[:3] * 0.1
    return jnp.array(coordinates), jnp.array(velocities), jnp.array(box)


def load_box(filename):
    traj = mdtraj.load(filename)
