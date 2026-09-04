# chemtrain-deploy

chemtrain-deploy exports JAX potential models and evaluates them from native
molecular-simulation engines. It includes a connector library, runtime
backends, and an optional LAMMPS package.

User documentation:

- [Installation](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/installation.html)
- [Getting started](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/getting_started.html)
- [LAMMPS package](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/lammps.html)
- [Model inputs](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/model_inputs.html)

Install chemtrain with `pip install chemtrain` for CPU exports, or with
`pip install 'chemtrain[cuda12]'` or `pip install 'chemtrain[cuda13]'` for the
corresponding NVIDIA CUDA Python packages. Build chemtrain-deploy separately
with `build.py`.

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

`out` holds build files, while `lib` is the runtime installation. The contents
of `lib` should remain together when the installation is copied or deployed. The connector finds its
runtime backends and bundled extensions automatically. See the
[installation guide](https://chemtrain.readthedocs.io/en/latest/chemtrain-deploy/installation.html)
when they are installed elsewhere or when an extension is needed.

## Optional model extensions

An FFI extension adds custom operations needed by a model. Most models do not
need one. Build an extension together with the connector by adding its Bazel
target to the build command. For example, this command builds the optional
OpenEquivariance CUDA extension:

```bash
python build.py \
  --enable_cuda \
  --build_gpu_pjrt_plugin \
  --ffi_provider_target=@openequivariance_src//openequivariance_extjax:libjcn_ffi_openequivariance.so \
  --install_location="$PWD/lib"
```

The connector loads and reports each installed extensions when it starts.
The installation guide explains how to use an extension built or installed in
a separate location.
