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

The CUDA extras install Python dependencies only. They do not build
chemtrain-deploy, `libconnector.so`, a PJRT plugin, or LAMMPS. Build those
components separately as described below.

The connector can be built for CPU or CUDA. CUDA is currently the only LAMMPS
execution path covered by the project's automated end-to-end checks. The CPU
host pair style has also been validated manually with single-rank and MPI runs.
A separate CPU regression tests model export, the CPU-only connector, the
public connector API, and host communication callbacks without depending on
LAMMPS. The model bundle must contain an executable for the selected backend.
See {ref}`chemtrain-deploy-platforms` for the distinction between executable
platforms and model variants.

Building the connector requires Python, Bazel, and a C++17 compiler. CUDA and
cuDNN are required only for a CUDA build. Building LAMMPS also requires CMake
and, for multi-rank simulations, an MPI implementation.

The connector build uses a hermetic Python 3.13 toolchain by default. The
compiled library does not embed that interpreter. Python 3.11, 3.12, 3.14, and
free-threaded Python 3.14 build toolchains can be selected with
`--python_version`.

The source tree pins the connector to the XLA revision shipped with JAX 0.11.0.
Update that revision together with the supported JAX range because PJRT and XLA
FFI are compiled interfaces, not Python-only dependencies.

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
`--output_path` selects the Bazel output base and may be placed on fast
temporary storage. `--install_location` is the persistent runtime directory.
After a CUDA build, the installation directory contains:

```text
lib/
├── libconnector.so
├── pjrt/
│   └── cuda/
│       ├── pjrt_plugin.so
│       └── deps/
└── ffi/
    └── cuda/
        ├── libjcn_ffi_openequivariance.so
        └── deps/
```

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

The CPU and CUDA plugin flags may be combined. The flags install
`pjrt/cpu/pjrt_plugin.so` and `pjrt/cuda/pjrt_plugin.so` beside one
`libconnector.so`. CPU supports ordinary and communication-enabled model
variants when the embedding application provides the connector's host
communication callbacks. CUDA is required for Kokkos execution. A build
without `--enable_cuda` omits CUDA-only OpenEquivariance implementations. The
CPU-only connector does not link CUDA implementations. Verify release
artifacts with `ldd`, or the platform equivalent, before distributing them.

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

LAMMPS and `libconnector.so` use a versioned C ABI. Rebuild both when that API
version changes.

> **Warning**
>
> Kokkos `/kk/device` communication strictly requires a CUDA-aware MPI build. Kokkos
> `/kk/host` and host-staged Kokkos communication are unsupported.

## Runtime Environment

At runtime, the dynamic loader must find `libconnector.so` and its
dependencies. By default the connector resolves its runtime files relative to
the loaded connector library: `pjrt/` and `ffi/` are sibling directories of
`libconnector.so`.

```bash
export PATH=/path/to/lammps/install/bin:$PATH
export LD_LIBRARY_PATH=/path/to/chemtrain/chemtrain-deploy/lib:${LD_LIBRARY_PATH:-}
```

`JCN_PJRT_PATH` is an optional override for the `pjrt/` directory itself. It
is not a search path:

```bash
export JCN_PJRT_PATH=/opt/chemtrain/runtime/pjrt
```

`JCN_FFI_PATH` optionally replaces the default FFI-provider directory with a
colon-separated, ordered list of directories:

```bash
export JCN_FFI_PATH=/opt/chemtrain/runtime/ffi:/shared/site-ffi
```

Every selected provider directory is scanned deterministically for provider
shared libraries. The connector retains loaded provider libraries for the
process lifetime because XLA retains their handler pointers. The LAMMPS
executable links `libconnector.so` at build time; it does not load the pair
style as a separate LAMMPS plugin.

### FFI providers

An FFI provider is a separate shared library, normally installed as
`ffi/<backend>/libjcn_ffi_<name>.so`. It is loaded after a PJRT backend is
available and must export this C entry point:

```cpp
extern "C" int RegisterFFi(const XLA_FFI_Api* api,
                           const char* platform_name);
```

The provider returns zero only after registering all of its typed XLA FFI
handler bundles for `platform_name`; a nonzero return aborts connector startup.
Use `api->XLA_FFI_Handler_Register` to register every supplied lifecycle stage.
Chemtrain resolves the plugin-local API from its generic
`XLA_FFI_GetPluginApi` anchor because the pinned XLA build keeps the public
`XLA_FFI_GetApi` symbol hidden. Runtime backends stay lowercase (`cuda` and
`cpu`); Chemtrain passes XLA's matching `CUDA` or `Host` lookup key to
providers. Providers
must not attempt to replace a target already registered by the connector or
another provider. Each provider must also print the target name and platform
after a successful registration, because the XLA FFI API does not expose a
general handler-enumeration API.

Providers use XLA's external FFI C API headers; they do not link XLA, PJRT, or
`libconnector.so`. The provider-only build mode
reuses `build.py`'s configured XLA revision, CUDA toolchain, and compiler flags
while building and installing only the requested Bazel provider target.
Updating a provider therefore does not rebuild `libconnector.so`, PJRT,
LAMMPS, or unrelated providers.

For example, rebuild and install only the packaged CUDA provider with the same
XLA and CUDA configuration used by the runtime:

```bash
python build.py \
  --enable_cuda \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so \
  --install_location=/path/to/chemtrain-deploy/lib
```

Pass the same target while building PJRT to compile and package the provider in
the same Bazel invocation:

```bash
python build.py \
  --enable_cuda \
  --build_gpu_pjrt_plugin \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so \
  --install_location=/path/to/chemtrain-deploy/lib
```

Repeat `--ffi_provider_target` to include additional providers for the same
backend. The build script installs providers under `ffi/cuda` when
`--enable_cuda` is set and under `ffi/cpu` otherwise. It does not select
OpenEquivariance implicitly.

The workspace uses its pinned OpenEquivariance revision by default. Pass
`--openequivariance_root=/path/to/OpenEquivariance` to use a local checkout
while developing the provider.

Verify the package and Kokkos styles in the installed executable:

```bash
lmp -h
```

The output should list `chemtrain` under pair styles and, for a Kokkos build,
`chemtrain/kk`.

Continue with the {doc}`getting_started` guide for a minimal exported-model
simulation or the {doc}`lammps` reference for all package options.
