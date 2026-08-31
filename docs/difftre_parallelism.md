# Distributed DiffTRe

`DifftreParallel` distributes a batch of independent state points across MPI
processes. It is intended for one visible JAX device per rank. Do not combine
this mode with a JAX distributed mesh in the same run. Set
`parallelism="mpi"` explicitly, or use `"auto"` to select MPI when more than
one rank is active. JAX sharding is not supported by this trainer.

Start one process per GPU and make one device visible to each rank. For
example, with Open MPI:

```bash
mpirun -np 2 bash -c \
  'CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK python train_difftre.py'
```

MPI runs require both `mpi4py` and `mpi4jax`. Every rank must construct the
trainer with the same state-point data, targets, and trajectory settings, and
must enter training operations in the same order. The global state-point batch
is split across ranks. Choose a batch size of at least the MPI size. If the
statepoint count has a remainder, chemtrain folds it into the preceding batch
so every statepoint is trained exactly once per epoch.

Parameters, optimizer state, and stored trajectories are replicated. Each rank
computes a disjoint set of statepoints. chemtrain averages the local gradients
over the true global batch size, then every rank applies the same optimizer
update. Use `sim_batch_size=-1` to update from all statepoints at once.

Trajectory initialization is explicit. Create or load trajectories before
training:

```python
from chemtrain.trainers import DifftreParallel

trainer = DifftreParallel(
    key,
    init_params,
    optimizer,
    energy_fn_template,
    simulator_template,
    neighbor_fn,
    timings,
    state_kwargs,
    quantities,
    targets,
    observables,
    reference_states=reference_states,
    sim_batch_size=4,
    parallelism="mpi",
)
trainer.initialize_trajstates(num_runs=1)
trainer.train(max_epochs=100)
```

The neighbor list keeps the positions from its last valid update. If it
overflows, chemtrain restarts from these reference positions and increases the
previous capacity by `overflow_capacity_multiplier`. It does not use a random
snapshot from the failed trajectory. Set `max_neighbor_retries` to limit
repeated attempts.

Use a shared filesystem for checkpoints and saved trajectory states. Call
`save_trajstates` and `load_trajstates` on every rank; rank zero performs the
file operation and reports failures to the other ranks. Resumable checkpoints
must come from a completed epoch and use the same MPI world size. Files named
`*_error_state.pkl` record a failed update for diagnosis and cannot be resumed.
Call `trainer.restore(path)` on every rank to resume a completed checkpoint.
