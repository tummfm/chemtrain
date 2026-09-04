(chemtrain-deploy-installation)=
# Installation

chemtrain-deploy consists of two compiled parts:

- `libconnector.so` and a PJRT plugin execute exported model bundles.
- The optional `chemtrain-deploy` LAMMPS package adapts LAMMPS atom and
  neighbor-list data to the connector API.

Install the Python package before exporting models. The ordinary installation
uses CPU JAX:

```bash
pip install 'chemtrain' --upgrade
```

For NVIDIA CUDA Python packages, install the matching JAX extra instead:

```bash
pip install 'chemtrain[cuda12]' --upgrade
# or
pip install 'chemtrain[cuda13]' --upgrade
```

The CUDA extras install the Python packages used to export CUDA models. Build
the connector and LAMMPS separately as described below.

The connector can be built for CPU or CUDA. Models must be exported for every
backend used at runtime. See {ref}`chemtrain-deploy-platforms`.

## Requirements

Building the connector requires Python 3.10 or newer and `patchelf`. By default,
`build.py` uses Clang from `PATH`. Pass `--nouse_clang` to let Bazel use its
configured C++17 toolchain. The script downloads a suitable Bazel binary if
needed. A CUDA build downloads the CUDA and cuDNN versions selected by its
options. Building LAMMPS needs CMake. The examples below enable MPI. Set
`BUILD_MPI=OFF` for a single-rank LAMMPS build without MPI.

The commands below use a CUDA build, which is the usual LAMMPS setup. A
separate command is provided for CPU-only installations.

## Build the Connector

### CUDA Connector

Run `build.py` from the `chemtrain-deploy` source directory:

```bash
python build.py \
  --enable_cuda \
  --build_gpu_pjrt_plugin \
  --cuda_version=12.9.1 \
  --cudnn_version=9.8.0 \
  --cuda_compute_capabilities=8.0 \
  --target_cpu_features=default \
  --output_path="$PWD/out" \
  --install_location="$PWD/lib"
```

Adjust the CUDA, cuDNN, and compute-capability values to the target system.
Multiple compute capabilities can be supplied as a comma-separated list.
`--output_path` selects the build directory and is a good place for fast
temporary storage. `--install_location` selects the runtime directory. Its
contents should remain together when the installation is moved.

The build enables NCCL by default. Pass `--noenable_nccl` when NCCL is
unavailable or should not be used. `python build.py --help` lists all
supported build options.

### CPU Connector

To package the CPU PJRT plugin, omit the CUDA options:

```bash
python build.py \
  --build_cpu_pjrt_plugin \
  --target_cpu_features=default \
  --output_path="$PWD/out" \
  --install_location="$PWD/lib"
```

The CPU and CUDA plugin flags may be combined when one installation needs both
backends. CUDA is required for the Kokkos pair style.

## Build LAMMPS

The LAMMPS integration is an in-tree optional package rather than a
runtime-loaded LAMMPS plugin. Configure the LAMMPS source tree with the package
enabled and link it to the connector built above:

### Host LAMMPS

```bash
cmake -S /path/to/lammps/cmake \
  -B /path/to/lammps/build-chemtrain \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_INSTALL_PREFIX=/path/to/lammps/install \
  -D BUILD_MPI=ON \
  -D PKG_CHEMTRAIN-DEPLOY=ON \
  -D CHEMTRAIN_DEPLOY_ROOT=/path/to/chemtrain/chemtrain-deploy \
  -D CHEMTRAIN_DEPLOY_LIBCONNECTOR=/path/to/chemtrain/chemtrain-deploy/lib/libconnector.so

cmake --build /path/to/lammps/build-chemtrain --parallel
cmake --install /path/to/lammps/build-chemtrain
```

### Kokkos LAMMPS

To build the Kokkos CUDA pair style, add the Kokkos CUDA preset and the
architecture corresponding to the target GPU:

```bash
cmake -S /path/to/lammps/cmake \
  -B /path/to/lammps/build-chemtrain-kokkos \
  -C /path/to/lammps/cmake/presets/kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_INSTALL_PREFIX=/path/to/lammps/install \
  -D BUILD_MPI=ON \
  -D PKG_CHEMTRAIN-DEPLOY=ON \
  -D CHEMTRAIN_DEPLOY_ROOT=/path/to/chemtrain/chemtrain-deploy \
  -D CHEMTRAIN_DEPLOY_LIBCONNECTOR=/path/to/chemtrain/chemtrain-deploy/lib/libconnector.so \
  -D Kokkos_ARCH_AMPERE80=ON

cmake --build /path/to/lammps/build-chemtrain-kokkos --parallel
cmake --install /path/to/lammps/build-chemtrain-kokkos
```

If an update changes the connector API version, rebuild both the connector and
LAMMPS.

> **Warning**
>
> Kokkos `/kk/device` communication strictly requires a CUDA-aware MPI build. Kokkos
> `/kk/host` and host-staged Kokkos communication are unsupported.

## Runtime Environment

At runtime, the dynamic loader must find `libconnector.so` and its
dependencies. Set the library path to the connector installation:

```bash
export PATH=/path/to/lammps/install/bin:$PATH
export LD_LIBRARY_PATH=/path/to/chemtrain/chemtrain-deploy/lib:${LD_LIBRARY_PATH:-}
```

Normally, no further setup is needed: the connector finds the runtime backend
and bundled extensions in its installation. Set `JCN_PJRT_PATH` only when the
runtime backends are installed in another directory. This setting replaces the
default location. The directory must contain one subdirectory per backend, for
example `cuda/pjrt_plugin.so`:

```bash
export JCN_PJRT_PATH=/opt/chemtrain/runtime/pjrt
```

Set `JCN_FFI_PATH` only when model extensions are installed outside the
connector installation. It replaces the default extension location and accepts
an ordered, colon-separated list of directories. Place extensions in a backend
subdirectory, such as `cuda/libjcn_ffi_example.so`. Include the
connector installation's extension directory in the list to retain its bundled
extensions:

```bash
export JCN_FFI_PATH=/opt/chemtrain/runtime/ffi:/shared/site-ffi
```

## Optional model extensions

An FFI extension is a separate shared library that supplies custom operations
used by a model. By default, no FFI extensions are required. Models that use
custom operations may require the extension named by their documentation. The
connector loads installed extensions at startup and prints their names. An
extension can be compiled with the connector or built separately with matching
settings.

For example, build and install the optional OpenEquivariance CUDA extension:

```bash
python build.py \
  --enable_cuda \
  --cuda_version=12.9.1 \
  --cudnn_version=9.8.0 \
  --cuda_compute_capabilities=8.0 \
  --target_cpu_features=default \
  --output_path="$PWD/out" \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so \
  --install_location=/path/to/chemtrain-deploy/lib
```

Separate extension builds must use the same CUDA and compiler settings as the
connector. Reusing `--output_path` also reuses cached build artifacts. To
build the connector and the extension together, add the target to the ordinary
CUDA build:

```bash
python build.py \
  --enable_cuda \
  --build_gpu_pjrt_plugin \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so \
  --install_location=/path/to/chemtrain-deploy/lib
```

Repeat `--ffi_provider_target` to include additional extensions. The build
script installs each extension in the selected connector installation, where
the connector will find it automatically.

Verify the package and Kokkos styles in the installed executable:

```bash
lmp -h
```

The output should list `chemtrain` under pair styles and, for a Kokkos build,
`chemtrain/kk`.

Continue with the {doc}`getting_started` guide for a minimal exported-model
simulation or the {doc}`lammps` reference for all package options.
