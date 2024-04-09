"""Pre-defined simulator templates."""

from jax_md import simulate
from jax_md.partition import NeighborList

from chemtrain.trajectory import traj_util


def init_nvt_langevin_simulator_template(shift_fn,
                                         nbrs: NeighborList = None,
                                         dt: float = 0.1,
                                         kT: float = 2.56,
                                         gamma: float = 100.):
    """Initializes a NVT Langevin simulator template."""
    extra_kwargs = dict(dt=dt, kT=kT, gamma=gamma)
    return traj_util.initialize_simulator_template(
        init_simulator_fn=simulate.nvt_langevin,
        shift_fn=shift_fn, nbrs=nbrs, init_with_PRNGKey=True,
        extra_simulator_kwargs=extra_kwargs
    )
