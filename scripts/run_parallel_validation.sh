#!/usr/bin/env bash
# Run the repeatable CPU validation for chemtrain's parallel force matching.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${repository_root}/../../venv/bin/python"
mpi_executable="/opt/openmpi-4.1.8-cuda/bin/mpiexec"

cd "${repository_root}"

# Force two lightweight CPU devices in one process. This tests real JAX array
# sharding without depending on the host's currently unhealthy GPU P2P path.
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
MPI4PY_RC_INITIALIZE=0 \
"${python_executable}" -m pytest --tb=short -q \
    tests/test_compose/test_utils.py \
    tests/test_compose/test_mace_example.py \
    tests/test_data/test_data_loader.py \
    tests/test_learn/test_data_parallel.py \
    tests/test_utils/test_mpi_utils.py

# Run the ordinary suite separately so failures outside the parallel paths are
# visible while the marked multi-device tests keep their controlled topology.
JAX_PLATFORMS=cpu \
MPI4PY_RC_INITIALIZE=0 \
"${python_executable}" -m pytest --tb=short -q \
    -m "not jax_multidevice and not mpi" \
    -k "not bucketing" \
    --ignore=tests/test_compose/test_utils.py \
    --ignore=tests/test_compose/test_mace_example.py \
    --ignore=tests/test_data/test_data_loader.py \
    --ignore=tests/test_utils/test_mpi_utils.py

# Give each MPI rank one CPU device. The tests exercise compiled mpi4jax
# reductions and disjoint HDF5 reads through the CUDA-aware Open MPI build.
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=1 \
"${mpi_executable}" -n 2 \
    "${python_executable}" -m pytest --tb=short -q \
    tests/test_learn/test_mpi_data_parallel.py
