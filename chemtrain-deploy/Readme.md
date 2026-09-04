# chemtrain-deploy

chemtrain-deploy exports JAX potential models as StableHLO bundles and evaluates
them from native molecular-simulation engines. The runtime installation
contains the versioned `libconnector.so` C API, PJRT plugins, and optional FFI
provider libraries. The supported LAMMPS integration is provided by the
optional in-tree `chemtrain-deploy` package.

User documentation:

- [Installation](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/installation.html)
- [Getting started](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/getting_started.html)
- [LAMMPS package](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/lammps.html)
- [Model inputs](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/model_inputs.html)

Install chemtrain with `pip install chemtrain` for CPU exports, or with
`pip install 'chemtrain[cuda12]'` or `pip install 'chemtrain[cuda13]'` for the
corresponding NVIDIA CUDA Python packages. These extras install Python
dependencies only. Build chemtrain-deploy separately with `build.py`.

Run `python build.py --help` for connector build options. A CUDA installation
uses separate build and runtime destinations:

```bash
python build.py \
  --enable_cuda \
  --build_gpu_pjrt_plugin \
  --cuda_version=12.9.1 \
  --cudnn_version=9.8.0 \
  --cuda_compute_capabilities=8.0 \
  --output_path="$PWD/out" \
  --install_location="$PWD/lib"
```

`out` is the Bazel output base. Runtime artifacts are copied to `lib`. The
default runtime layout puts `pjrt/` and `ffi/` next to `libconnector.so`; the
connector discovers both relative to the loaded connector library. Set the
single-root `JCN_PJRT_PATH` override only when PJRT plugins live elsewhere.
Set `JCN_FFI_PATH` to a colon-separated list of provider directories when the
default `ffi/` directory is not sufficient.

FFI providers are independent shared libraries, conventionally named
`libjcn_ffi_<name>.so`. Each exports:

```cpp
extern "C" int RegisterFFi(const XLA_FFI_Api* api,
                           const char* platform_name);
```

Chemtrain loads each provider explicitly, then asks the loaded PJRT plugin for
its plugin-local `XLA_FFI_Api`. Providers register complete handler bundles
through `api->XLA_FFI_Handler_Register` for the supplied platform and return
zero on success. This supports every XLA FFI lifecycle stage without an XLA
source patch. The pinned XLA build keeps `XLA_FFI_GetApi` hidden, so Chemtrain
exports the strictly equivalent generic `XLA_FFI_GetPluginApi` forwarding
anchor from each packaged PJRT plugin. The provider-only build mode reuses the
configured XLA FFI headers and compiler settings without rebuilding the
connector, PJRT plugins, or LAMMPS. See the installation guide for discovery
and build details.

Chemtrain communication registers a complete four-stage bundle for both
`Host` and `CUDA`. CUDA validates immutable buffer metadata during
instantiation; Host keeps that validation in execute because the pinned CPU
runtime supplies no buffers to its instantiate call. Prepare and initialize
are stateless and never retain execution buffers, streams, or callbacks.

The OpenEquivariance provider is an explicit CUDA target at
`@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so`.
The workspace downloads a pinned OpenEquivariance revision. It uses the
connector's pinned XLA and CUDA repositories and is not part of a standard
connector build. Pass `--openequivariance_root=/path/to/OpenEquivariance` to
build an uncommitted or locally modified checkout instead.

```bash
python build.py --enable_cuda \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so
```

## Full CPU regression

Merge-request pipelines run the standalone CPU connector regression only when
the merge request has the exact, case-sensitive `CI-ready` label. Add the
label before starting the pipeline, or retry the pipeline after adding it. The
job builds the CPU connector and PJRT plugin, exports its model fixtures, and
tests the public JCN API without LAMMPS, MPI, Kokkos, or a GPU.
