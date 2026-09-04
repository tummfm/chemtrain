# MACE-JAX examples

`finetune_md22.py` is the maintained force-matching example. It downloads the
MD22 double-walled nanotube dataset, converts its 370-atom structures to
chemtrain units, writes deterministic HDF5 splits, and fine-tunes a MACE-MP
foundation model through the public `ForceMatching` interface.

Install MACE-JAX from the revision supported by this branch together with the
optional data dependencies:

```bash
git clone https://github.com/ACEsuit/mace-jax /tmp/mace-jax
git -C /tmp/mace-jax fetch origin pull/21/head
git -C /tmp/mace-jax switch --detach 594563b322d6127f9b8903eec534dcde51fed83d
python -m pip install /tmp/mace-jax
python -m pip install h5py mace-torch
```

Run one normal training job with all visible JAX devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/mace_jax/finetune_md22.py \
  --parallelism jax --global-batch 16 --workdir output/md22-jax
```

The example writes regular chemtrain checkpoints and a prediction plot below
`--workdir`. The batch is global, so this command gives eight structures to
each of two devices.

Timing is kept in a separate program. Each invocation measures one backend and
uses chemtrain's public epoch timings after discarding compilation epochs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/mace_jax/benchmark_md22.py \
  --parallelism jax --global-batch 16 --epochs 4 --discard-epochs 1 \
  --workdir output/md22-data --output output/md22-jax.json
```

`run_md22_benchmarks.py` runs the single-GPU, two-device JAX, and two-rank MPI
cases with synchronous and asynchronous loading. Its MPI prefix must point to
a CUDA-aware installation when GPU buffers are passed directly:

```bash
python examples/mace_jax/run_md22_benchmarks.py \
  --gpus 0 1 --global-batch 16 \
  --output-directory output/md22-comparison \
  --mpi-prefix /opt/openmpi-4.1.8-cuda
```

Keep the global batch, model, dataset, precision, and discarded epochs fixed
when comparing backends. The report includes samples and atoms per second and
uses the slowest MPI rank for each epoch.

`train_difftre_example.py` is retained unchanged as a separate legacy DiffTRe
example. It is not part of the force-matching benchmark.
