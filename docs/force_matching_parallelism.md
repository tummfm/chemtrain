# Parallel force matching

`ForceMatching` can split one training batch between several devices. Import
the trainer from the public package and select the method with `parallelism`:

- `single` uses one Python process and one CPU or GPU device.
- `mpi` uses several MPI processes. Each process must see exactly one device.
- `jax` lets JAX split one batch between the selected devices and processes.
- `auto` selects JAX distributed training, MPI, local multi-device JAX, or
  single-device training in that order.

Only one method can be active. chemtrain reports an error when MPI and JAX
distributed training are mixed by accident.

## Batch size

`batch` is always the total number of structures used in one optimizer step.
For example, `batch=32` with four devices gives eight structures to each
device. The batch size must be divisible by the number of devices or MPI
processes.

`batch_per_device` remains available for older scripts but is deprecated. New
code should always set `batch`.

```python
from chemtrain.trainers import ForceMatching

trainer = ForceMatching(
    init_params,
    optimizer,
    energy_fn_template,
    nbrs_init,
    batch=32,
    parallelism="jax",
)
```

## HDF5 data

`HDF5ParallelDataLoader` selects the same sample indices on every process, but
reads only the samples needed by that process:

- In MPI mode, each process reads one continuous part of the batch.
- In JAX mode, each process reads the parts stored on its local devices.
- Padding masks are split in the same way as the samples.

The HDF5 file must contain datasets of equal length at its root. Pass a file
path for MPI or distributed JAX training. Each process then opens and closes
its own read-only file. An already open `h5py.File` can only be used for a
single-process run.

This is different from a generic in-memory loader. Such a loader cannot read
only the part needed by each rank, so MPI rank zero reads the cache and sends
each rank a contiguous CPU slice. HDF5 never uses this root-scatter path.

HDF5 reads can run in a background thread for every parallel method. In JAX
mode the worker returns NumPy arrays. The training thread then places those
arrays on the device mesh, because JAX device operations belong on that thread.

## MPI runs

MPI training needs both `mpi4py` and `mpi4jax>=0.9.0.post1`. Start one process
for each GPU and make exactly one GPU visible to every process. For example,
with Open MPI:

```bash
mpirun -np 2 bash -c \
  'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK python train.py'
```

MPI support is optional. Install it with `pip install 'chemtrain[mpi]'`;
importing chemtrain does not import either package. chemtrain loads mpi4py
when it first checks the MPI process state. It only loads mpi4jax after an
active multi-rank MPI run requests compiled collectives. An incompatible
mpi4jax installation therefore does not affect single-device or JAX-sharded
training.

At startup, rank 0 sends its parameters and optimizer state to all other
ranks. Each step then averages one packed gradient before every rank applies
the same optimizer update. Sending the complete model after every update is
not necessary and would be expensive. Long runs may add a periodic check of
parameter and optimizer-state summaries when debugging rank differences.

## JAX runs

A local JAX run uses all visible devices. An optional one-dimensional
`jax.sharding.Mesh` named `data` can be passed through the `mesh` argument.

Parameters and optimizer state are replicated. The batch and predictions are
sharded along their leading dimension. Every device evaluates its local loss
and gradient, then `lax.pmean` averages the gradients before every replica
applies the same optimizer update.

Temporary JAX workaround: cuEquivariance does not preserve JAX's manual-axis
information through segmented polynomials. chemtrain turns off that check,
marks parameters as local while differentiating, and then explicitly averages
the local gradients. Remove this once cuEquivariance preserves the information
in both differentiation directions.

For several JAX processes, call `jax.distributed.initialize` before creating a
trainer or making any other call that initializes JAX devices. All processes
must use the same global mesh and enter updates in the same order. Process zero
broadcasts the initial parameters and complete optimizer state before chemtrain
places them with replicated sharding.

The following checks distinguish sharded predictions from replicated state:

```python
parameter_leaf = jax.tree.leaves(trainer.params)[0]
assert parameter_leaf.sharding.is_fully_replicated
```

For detailed diagnostics, inspect each `jax.Array.sharding` and its
`addressable_shards`. A multi-host process sees only its locally addressable
pieces.

## MACE example and speed measurements

`examples/mace_jax/finetune_md22.py` fine-tunes MACE-MP on the MD22
double-walled nanotube. Each structure has 370 atoms, which makes this example
better suited to measuring graph-model throughput than very small molecules.
The script converts Angstrom and kcal/mol inputs to nm and kJ/mol, saves normal
chemtrain checkpoints, and plots test predictions in eV/atom and eV/Angstrom.

`examples/mace_jax/benchmark_md22.py` is kept separate so timing code does
not make the training example harder to read. It discards the first epochs,
which include JAX compilation, and records the remaining chemtrain training
times as JSON.

For a fair 1/2/4-device comparison:

1. Use the same dataset and global batch size for every run.
2. Keep the MACE model, numerical precision, and neighbor-list size unchanged.
3. Discard compilation epochs.
4. Measure MPI and JAX runs separately.
5. Use the slowest MPI rank when reporting an MPI time.

The most important performance choices are usually the number of padded atoms
and neighbor entries, the global batch size, HDF5 layout and cache size, JAX
compilation reuse, and how often validation and checkpoints run.

## Current limits

- MPI needs `mpi4jax`; `mpi4py` alone is not enough for compiled gradients.
- MPI requires one visible JAX device per rank.
- Systems with different atom counts must be padded to fixed shapes.
- Multi-process JAX must be initialized before the trainer is constructed.
- A padded loss divides by the fixed local shard capacity. All shards must
  therefore have equal capacity; chemtrain applies one global valid-sample
  correction after accumulating evaluation batches.
- `disable_shmap=True` is no longer supported. Use `parallelism="jax"`.
