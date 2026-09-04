# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions helpful in designing new trainers."""
from contextlib import contextmanager
from functools import cache, partial
import importlib
from typing import Any

import chex
import cloudpickle as pickle

import jax
from jax import tree_util, device_count, numpy as jnp, lax
from jax.tree_util import tree_map

from jax_md import simulate
import numpy as onp

_MPI4JAX_COMM = None


@cache
def _import_optional(module):
    """Import an optional package once and preserve nested import errors."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module.partition(".")[0]:
            raise
        return None


def has_mpi4py():
    """Whether the optional mpi4py dependency is available."""
    return _import_optional("mpi4py.MPI") is not None


def use_mpi():
    """Whether this process belongs to an initialized multi-rank MPI run."""
    return get_mpi() is not None


def get_mpi():
    """Return mpi4py for an active multi-rank run."""
    mpi = _import_optional("mpi4py.MPI")
    if mpi is None or not mpi.Is_initialized():
        return None
    if mpi.COMM_WORLD.Get_size() <= 1:
        return None
    return mpi


def get_mpi4jax():
    """Return mpi4jax when compiled MPI collectives can be used."""
    if not use_mpi():
        return None
    return _import_optional("mpi4jax")


def has_mpi4jax():
    """Whether compiled MPI collectives are available."""
    return get_mpi4jax() is not None


def is_root():
    """Whether this is the process responsible for user-visible side effects."""
    if jax.process_count() > 1:
        return jax.process_index() == 0
    if not use_mpi():
        return True
    mpi = get_mpi()
    assert mpi is not None
    return mpi.COMM_WORLD.Get_rank() == 0


def get_communicator():
    """Returns the MPI communicator."""
    mpi = get_mpi()
    if mpi is None:
        return None
    return mpi.COMM_WORLD


@contextmanager
def mpi_guard():
    """Let all MPI processes continue only after every process succeeds.

    Every process must enter guarded sections in the same order. If one or
    more processes fail, all processes raise the same rank-ordered error.
    """
    local_exception = None
    local_error = None
    try:
        yield
    except Exception as exception:  # Synchronize before leaving this process.
        local_exception = exception
        local_error = f"{type(exception).__name__}: {exception}"

    if use_mpi():
        comm = get_communicator()
        assert comm is not None
        errors = comm.allgather(local_error)
    else:
        errors = [local_error]

    failures = [
        f"rank {rank}: {message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if failures:
        error = RuntimeError(
            "Parallel operation failed; " + "; ".join(failures)
        )
        if local_exception is not None:
            raise error from local_exception
        raise error


def _get_mpi4jax_communicator():
    """Return a communicator reserved for compiled mpi4jax collectives."""
    global _MPI4JAX_COMM
    mpi = get_mpi()
    if mpi is None:
        return None
    if _MPI4JAX_COMM is None:
        _MPI4JAX_COMM = mpi.COMM_WORLD.Dup()
    return _MPI4JAX_COMM


def mpi_any(x):
    """Elementwise logical OR of a boolean array across all MPI processes."""
    x = jnp.asarray(x, dtype=bool)

    if not use_mpi():
        return x

    comm = get_communicator()
    mpi = get_mpi()
    assert comm is not None and mpi is not None

    x_host = onp.asarray(x)
    out = onp.empty_like(x_host, dtype=bool)

    comm.Allreduce(x_host, out, op=mpi.LOR)

    return jnp.asarray(out)


def mpi_tree_slice(tree, dim=None):
    """Slices a pytree with disjoint subsets across MPI processes."""
    if not use_mpi():
        return tree, None

    leaves = tree_util.tree_leaves(tree)
    if dim is None:
        dim = leaves[0].shape[0]

    assert all(
        leaf.shape[0] == leaves[0].shape[0] for leaf in leaves
    ), 'Tree first dimension size is not equal.'

    comm = get_communicator()
    assert comm is not None
    rank = comm.Get_rank()
    size = comm.Get_size()
    return tree_util.tree_map(lambda x: x[rank::size], tree), dim


def mpi_tree_broadcast(tree, root: int = 0):
    """Broadcast a pytree from `root` to all MPI processes using mpi4jax."""
    if not use_mpi():
        return tree

    mpi4jax = get_mpi4jax()
    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_broadcast."
        )

    world = get_communicator()
    mpi = get_mpi()
    comm = _get_mpi4jax_communicator()
    assert world is not None and mpi is not None and comm is not None
    rank = world.Get_rank()

    def _bcast_leaf(x):
        x = jnp.asarray(x)
        if rank != root:
            x = jnp.zeros_like(x)
        return mpi4jax.allreduce(
            x, mpi.LOR if x.dtype == jnp.bool_ else mpi.SUM, comm=comm
        )

    return tree_util.tree_map(_bcast_leaf, tree)


def mpi_tree_gather(tree, dim=None):
    """Gathers a pytree from all MPI processes."""
    if not use_mpi():
        return tree

    mpi4jax = get_mpi4jax()
    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_gather/mpi_tree_mean inside jitted code."
        )
    world = get_communicator()
    mpi = get_mpi()
    comm = _get_mpi4jax_communicator()
    assert world is not None and mpi is not None and comm is not None
    rank = world.Get_rank()
    size = world.Get_size()

    gathered_tree = tree_util.tree_map(
        lambda x: jnp.zeros(
            (size * x.shape[0] if dim is None else dim, *x.shape[1:]), dtype=x.dtype
        ).at[rank::size].set(x), tree
    )

    def gather_leaf(x):
        """Reduce one gathered array."""
        return mpi4jax.allreduce(
            x, mpi.LOR if x.dtype == jnp.bool_ else mpi.SUM, comm=comm
        )

    tree = tree_util.tree_map(gather_leaf, gathered_tree)
    return tree


def mpi_tree_mean(tree, dim=None):
    """Mean of a pytree from all MPI processes."""
    if not use_mpi():
        return tree

    mpi4jax = get_mpi4jax()
    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_gather/mpi_tree_mean inside jitted code."
        )
    world = get_communicator()
    mpi = get_mpi()
    comm = _get_mpi4jax_communicator()
    assert world is not None and mpi is not None and comm is not None
    rank = world.Get_rank()
    size = world.Get_size()

    if dim is None:
        slice_size = 1
        dim = size
    else:
        slice_size = onp.arange(dim)[rank::size].size

    def mean_leaf(x):
        """Average one array."""
        return mpi4jax.allreduce(
            x * (slice_size / dim), mpi.SUM, comm=comm
        )

    tree = tree_util.tree_map(mean_leaf, tree)
    return tree


def mpi_tree_first_masked(tree, mask):
    """Return the globally first tree entry for which mask is nonzero.

    Assumes tree and mask are distributed using the same rank::size
    convention as `mpi_tree_slice`.

    Args:
        tree: Pytree with leaves of shape (N_local, ...).
        mask: Array of shape (N_local,).

    Returns:
        selected: Pytree containing the globally first matching entry.
            If no entry matches, returns zeros with the corresponding
            leaf shapes.
        found: Scalar boolean indicating whether a matching entry exists.
    """
    mask = jnp.asarray(mask, dtype=bool)

    # Non-MPI case.
    if not use_mpi():
        found = jnp.any(mask)

        if mask.shape[0] == 0:
            selected = tree_map(
                lambda x: jnp.zeros(x.shape[1:], dtype=x.dtype),
                tree,
            )
            return selected, found

        idx = jnp.argmax(mask)

        selected = tree_map(
            lambda x: jnp.where(
                found,
                x[idx],
                jnp.zeros_like(x[idx]),
            ),
            tree,
        )
        return selected, found

    mpi4jax = get_mpi4jax()
    if mpi4jax is None:
        raise RuntimeError("MPI is active but mpi4jax is unavailable.")

    mpi = get_mpi()
    comm = _get_mpi4jax_communicator()
    assert mpi is not None and comm is not None
    rank = comm.Get_rank()
    size = comm.Get_size()

    sentinel = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)

    # Find the first matching entry on this rank.
    if mask.shape[0] == 0:
        local_found = jnp.asarray(False)
        local_idx = jnp.asarray(0, dtype=jnp.int32)
        local_global_idx = sentinel
    else:
        local_found = jnp.any(mask)
        local_idx = jnp.argmax(mask).astype(jnp.int32)

        # mpi_tree_slice uses x[rank::size], so:
        #
        # local index i -> global index rank + size * i
        local_global_idx = jnp.where(
            local_found,
            rank + size * local_idx,
            sentinel,
        ).astype(jnp.int32)

    # Find the globally first matching index.
    first_global_idx = mpi4jax.allreduce(
        local_global_idx,
        op=mpi.MIN,
        comm=comm,
    )

    found = first_global_idx != sentinel

    # Only the rank owning first_global_idx contributes a nonzero value.
    local_is_selected = (
        local_found
        & (local_global_idx == first_global_idx)
    )

    leaves, treedef = tree_util.tree_flatten(tree)
    selected_leaves = []

    for x in leaves:
        x = jnp.asarray(x)

        if x.shape[0] == 0:
            candidate = jnp.zeros(x.shape[1:], dtype=x.dtype)
        else:
            candidate = x[local_idx]

        contribution = jnp.where(
            local_is_selected,
            candidate,
            jnp.zeros_like(candidate),
        )

        # Exactly one rank contributes, so SUM reproduces that rank's value.
        # For booleans use logical OR.
        op = mpi.LOR if jnp.issubdtype(x.dtype, jnp.bool_) else mpi.SUM

        selected = mpi4jax.allreduce(
            contribution,
            op=op,
            comm=comm,
        )

        selected_leaves.append(selected)

    return tree_util.tree_unflatten(treedef, selected_leaves), found


def mpi_tree_mean_packed(tree):
    """Average a tree with one MPI call for each dtype.

    Floating-point arrays with the same communication dtype are joined and
    restored to their original shapes afterwards. Half-precision arrays are
    sent as float32 because MPI does not support their JAX dtypes. Empty arrays
    are kept without communication. Non-array tree structure is preserved. The
    input is returned unchanged when MPI is inactive. Non-floating leaves raise
    ``ValueError`` because averaging them would not preserve their meaning.
    """
    if not use_mpi():
        return tree

    leaves, tree_def = tree_util.tree_flatten(tree)
    if not leaves:
        return tree

    dtype_groups = {}
    for index, leaf in enumerate(leaves):
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise ValueError("Packed MPI means require floating-point arrays.")
        if array.size == 0:
            continue
        communication_dtype = array.dtype
        if jnp.issubdtype(array.dtype, jnp.floating) and array.dtype.itemsize < 4:
            communication_dtype = jnp.dtype("float32")
        dtype_groups.setdefault(communication_dtype, []).append((index, array))

    reduced_leaves = list(leaves)
    for communication_dtype, group in dtype_groups.items():
        packed = jnp.concatenate([
            array.astype(communication_dtype).reshape(-1)
            for _, array in group
        ])
        packed = mpi_tree_mean(packed)
        offset = 0
        for index, array in group:
            size = array.size
            reduced_leaves[index] = packed[offset:offset + size].reshape(
                array.shape
            ).astype(array.dtype)
            offset += size

    return tree_util.tree_unflatten(tree_def, reduced_leaves)


# freezing seems to give slight performance improvement
@partial(chex.dataclass, frozen=True)
class TrainerState:
    """Each trainer at least contains the state of parameter and
    optimizer.
    """
    params: Any
    opt_state: Any


def _get_box_kwargs_if_npt(state):
    kwargs = {}
    if is_npt_ensemble(state):
        box = simulate.npt_box(state)
        kwargs['box'] = box
    return kwargs


def neighbor_update(neighbors, state, **kwargs):
    """Update neighbor lists irrespective of the ensemble.

    Fetches the box to the neighbor list update function in case of the
    NPT ensemble.

    Args:
        neighbors: Neighbor list to be updated
        state: Simulation state

    Returns:
        Updated neighbor list
    """
    kwargs.update(_get_box_kwargs_if_npt(state))
    nbrs = neighbors.update(state.position, **kwargs)
    return nbrs


def neighbor_allocate(neighbor_fn, state, extra_capacity=0):
    """Re-allocates neighbor lost irrespective of ensemble. Not jitable.

    Args:
        neighbor_fn: Neighbor function to re-allocate neighbor list
        state: Simulation state
        extra_capacity: Additional capacity of new neighbor list

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbor_fn.allocate(state.position, extra_capacity, **kwargs)
    return nbrs


def is_npt_ensemble(state):
    """Whether a state belongs to the NPT ensemble."""
    return hasattr(state, 'box_position')


def kl(p, q):
    """Returns Kullback-Leibler divergence D(P || Q) for discrete distributions.

    Args:
        p: Discrete probability density function values (array-like, shape=n).
        q: Discrete probability density function values (array-like, shape=n).
    """
    p = jnp.asarray(p, dtype=float)
    q = jnp.asarray(q, dtype=float)
    return -jnp.sum(jnp.where(p != 0, p * jnp.log(q / p), 0))


def jenson_shannon(p, q, distance=False):
    """Returns the Jensen-Shannon distance between two discrete distributions.

    Args:
        p: Discrete probability density function values (array-like, shape=n).
        q: Discrete probability density function values (array-like, shape=n).
        distance: Default False returns JS divergence. True returns JS distance
                  sqrt(JS).
    """
    p = onp.array(p)
    q = onp.array(q)
    m = 0.5 * (p + q)
    js = 0.5 * (kl(p, m) + kl(q, m))
    if distance:
        return onp.sqrt(js)
    else:
        return js


def tree_combine(tree):
    """Combines the first two axes of `tree`, e.g. after batching."""
    return tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), tree)


def tree_norm(tree):
    """Returns the Euclidean norm of a PyTree."""
    leaves, _ = tree_util.tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


def tree_get_single(tree, n=0):
    """Returns the n-th tree of a tree-replica, e.g. from pmap.
    By default, the first tree is returned.
    """
    single_tree = tree_map(lambda x: jnp.array(x[n]), tree)
    return single_tree


def tree_set(tree, new_data, end, start=0):
    """Overrides entries of a tree from index start:end along axis 0
    with new_data.
    """
    return tree_map(lambda leaf, new_data_leaf:
                    leaf.at[start:end, ...].set(new_data_leaf), tree, new_data)


def tree_get_slice(tree, idx_start, idx_stop, take_every=1, to_device=True):
    """Returns a slice of trees taken from a tree-replica along axis 0."""
    if to_device:
        return tree_map(lambda x: jnp.array(x[idx_start:idx_stop:take_every]),
                        tree)
    else:
        return tree_map(lambda x: x[idx_start:idx_stop:take_every], tree)


def tree_take(tree, indicies, axis=0, on_cpu=True):
    """Tree-wise application of numpy.take."""
    numpy = onp if on_cpu else jnp
    return tree_map(lambda x: numpy.take(x, indicies, axis), tree)


def tree_put(tree, indicies, values, axis=0, on_cpu=True):
    """Tree-wise application of numpy.put_along_axis."""
    if on_cpu:
        assert axis == 0, 'Only axis=0 is supported for numpy.'
        indicies = onp.asarray(indicies)

        def _put(x, y):
            x = onp.array(x, copy=True)
            x[indicies, ...] = y
            return x

        return tree_map(_put, tree, values)
    else:
        assert axis == 0, 'Only axis=0 is supported for jax.'
        return tree_map(
            lambda x, y: x.at[indicies, ...].set(y), tree, values)


def tree_delete(tree, indicies, axis=None, on_cpu=True):
    """Returns a tree, where entries at position indicies along axis are
    deleted from the original tree.
    """
    numpy = onp if on_cpu else jnp
    return tree_map(lambda leaf: numpy.delete(leaf, indicies, axis=axis), tree)


def tree_append(orig_tree, tree_to_append, axis=None, on_cpu=True):
    numpy = onp if on_cpu else jnp
    return tree_map(partial(numpy.append, axis=axis), orig_tree, tree_to_append)


def tree_replicate(tree, replicas: int = None):
    """Replicates a pytree along the first axis."""
    if replicas is None:
        replicas = device_count()

    return tree_map(
        lambda x: jnp.tile(jnp.expand_dims(x, 0), (replicas,) + (1,) * x.ndim),
        tree
    )


def tree_axis_swap(tree, axis1=0, axis2=1):
    """Swaps axes of all leaves of a pytree."""
    return tree_map(lambda x: jnp.swapaxes(x, axis1, axis2), tree)


def tree_concat(tree):
    """For output computed in parallel via pmap, restacks all leaves such that
    the parallel dimension is again along axis 0 and the leading pmap dimension
    vanishes.
    """
    return tree_map(partial(jnp.concatenate, axis=0), tree)


def tree_pmap_split(tree, n_devices):
    """Splits the first axis of `tree` evenly across the number of devices for
     pmap batching (size of first axis is n_devices).
     """
    assert tree_util.tree_leaves(tree)[0].shape[0] % n_devices == 0, \
        'First dimension needs to be multiple of number of devices.'
    return tree_map(lambda x: jnp.reshape(x, (n_devices, x.shape[0]//n_devices,
                                              *x.shape[1:])), tree)


def tree_vmap_split(tree, batch_size):
    """Splits the first axis of a 'tree' with leaf sizes (N, X)`into
    (n_batches, batch_size, X) to allow straightforward vmapping over axis0.
    """
    if len(tree_util.tree_leaves(tree)) == 0:
        return tree

    assert tree_util.tree_leaves(tree)[0].shape[0] % batch_size == 0, \
        'First dimension of tree needs to be splittable by batch_size' \
        ' without remainder.'
    return tree_map(lambda x: jnp.reshape(x, (x.shape[0] // batch_size,
                                              batch_size, *x.shape[1:])),
                    tree)


def tree_sum(*tree_list, axis=0):
    """Computes the sum of equal-shaped leafs of a pytree."""
    @partial(partial, tree_map)
    def leaf_add(*leafs):
        return jnp.sum(jnp.stack(leafs, axis=axis), axis=axis)
    return leaf_add(*tree_list)


def tree_mean(tree_list):
    """Computes the mean a list of equal-shaped pytrees."""
    @partial(partial, tree_map)
    def tree_add_imp(*leafs):
        return jnp.mean(jnp.stack(leafs), axis=0)

    return tree_add_imp(*tree_list)


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves, treedef = tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def tree_multiplicity(tree):
    """Returns the number of stacked trees along axis 0."""
    leaves, _ = tree_util.tree_flatten(tree)
    return leaves[0].shape[0]


def assert_distributable(total_samples, n_devies, vmap_per_device):
    assert total_samples % (n_devies * vmap_per_device) == 0, (
        'For parallelization, the samples need to be evenly distributed'
        'over the devices and vmap, i.e. be a multiple of n_devices * n_vmap.')


def load_trainer(file_path):
    """Returns the trainer saved via 'trainer.save_trainer'.

    Args:
        file_path: Path of pickle file containing trainer.

    """
    with open(file_path, 'rb') as pickle_file:
        trainer = pickle.load(pickle_file)
    trainer.move_to_device()
    return trainer


def format_not_recognized_error(file_format):
    raise ValueError(f'File format {file_format} not recognized. '
                     f'Expected ".hdf5" or ".pkl".')

def batch_map(f, xs, batch_size: int = 1):
    """Maps a function over an array in batches.

    Substitute for ``lax.map`` with batch size argument from later jax versions.

    Args:
        f: Function to map.
        xs: List of arguments to map over.
        batch_size: Size of each batch.

    Returns:
        Returns results of f evaluated at element entry of xs.

    """

    f_vmapped = jax.vmap(f)
    tree_leaves, tree_structure = tree_util.tree_flatten(xs)

    # Ensure that the batch size is not larger than the number of samples
    batch_size = onp.min([batch_size, tree_leaves[0].shape[0]])

    # First, we split the pytree into batch and remainder part
    batches = []
    remainders = []

    for leave in tree_leaves:
        n_batches = leave.shape[0] // batch_size
        remainder = leave.shape[0] % batch_size

        if n_batches > 0:
            batches.append(jnp.reshape(leave[:n_batches * batch_size], (n_batches, batch_size, *leave.shape[1:])))
        if remainder > 0:
            remainders.append(leave[-remainder:])

    # Then, we map over the batches and compute the remainder in a single pass
    batch_results = lax.map(
        f_vmapped, tree_util.tree_unflatten(tree_structure, batches))
    # We are done if we can split the data into batches without remainder
    if len(remainders) == 0:
        return tree_concat(batch_results)

    remainder_results = f_vmapped(
        tree_util.tree_unflatten(tree_structure, remainders))

    # Concatenate remainder results and batches
    results = tree_util.tree_map(
        lambda x, y: jnp.concat([x, y], axis=0),
        tree_concat(batch_results), remainder_results
    )

    return results
