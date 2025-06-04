#!/usr/bin/python
#
# Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Helper script for building chemtrain-deploy connector.
# Adapted from https://github.com/jax-ml/jax

__BAZELRC = """
# Load the JAX bazelrc
import %workspace%/jax.bazelrc
"""

import argparse
import collections
import hashlib
import logging
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import urllib.request

import pkgutil
import importlib
from typing import Optional

try:
    import jax_plugins
except ModuleNotFoundError:
    jax_plugins = None

logger = logging.getLogger(__name__)

def shell(cmd):
  try:
    logger.info("shell(): %s", cmd)
    output = subprocess.check_output(cmd)
  except subprocess.CalledProcessError as e:
    logger.info("subprocess raised: %s", e)
    if e.output: print(e.output)
    raise
  except Exception as e:
    logger.info("subprocess raised: %s", e)
    raise
  return output.decode("UTF-8").strip()


# Python

def load_pjrt_plugin_libraries(out) -> None:
    """Discovers plugins in the namespace package `jax_plugins` and loads
     the shared libraries.
    """
    plugin_modules = set()
    # Scan installed modules under |jax_plugins|. Note that not all packaging
    # scenarios are amenable to such scanning, so we also use the entry-point
    # method to seed the list.
    if jax_plugins:
        for _, name, _ in pkgutil.iter_modules(
            jax_plugins.__path__, jax_plugins.__name__ + '.'
        ):
            logger.debug("Discovered path based JAX plugin: %s", name)
            plugin_modules.add(name)
    else:
        raise ModuleNotFoundError("To load shared libraries, the jax_plugins "
                                  "namespace package must be available.")

    # Augment with advertised entrypoints.
    from importlib.metadata import entry_points

    for entry_point in entry_points(group="jax_plugins"):
        logger.debug("Discovered entry-point based JAX plugin: %s",
                     entry_point.value)
        plugin_modules.add(entry_point.value)

    # Now load and initialize them all.
    for plugin_module_name in plugin_modules:
        logger.debug("Loading plugin module %s", plugin_module_name)
        plugin_module = None
        try:
            plugin_module = importlib.import_module(plugin_module_name)
        except ModuleNotFoundError:
            logger.warning("Jax plugin configuration error: Plugin module %s "
                           "does not exist", plugin_module_name)
        except ImportError:
            logger.exception("Jax plugin configuration error: Plugin module %s "
                             "could not be loaded")

        if plugin_module:
            try:
                name = plugin_module_name.replace("jax_plugins", "pjrt_plugin")
                so_path = plugin_module._get_library_path()
                out_path = (out / f"{name}.so")

                out_path.write_bytes(pathlib.Path(so_path).read_bytes())
            except:
                raise RuntimeError(f"Failed to load plugin {plugin_module_name}")

def get_python_bin_path(python_bin_path_flag):
  """Returns the path to the Python interpreter to use."""
  path = python_bin_path_flag or sys.executable
  return path.replace(os.sep, "/")


def get_python_version(python_bin_path):
  version_output = shell(
    [python_bin_path, "-c",
     ("import sys; print(\"{}.{}\".format(sys.version_info[0], "
      "sys.version_info[1]))")])
  major, minor = map(int, version_output.split("."))
  return major, minor

def check_python_version(python_version):
  if python_version < (3, 10):
    print("ERROR: JAX requires Python 3.10 or newer, found ", python_version)
    sys.exit(-1)


def get_githash():
  try:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        encoding='utf-8',
        capture_output=True).stdout.strip()
  except OSError:
    return ""

# Bazel

BAZEL_BASE_URI = "https://github.com/bazelbuild/bazel/releases/download/6.5.0/"
BazelPackage = collections.namedtuple("BazelPackage",
                                      ["base_uri", "file", "sha256"])
bazel_packages = {
    ("Linux", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.5.0-linux-x86_64",
            sha256=
            "a40ac69263440761199fcb8da47ad4e3f328cbe79ffbf4ecc14e5ba252857307"),
    ("Linux", "aarch64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.5.0-linux-arm64",
            sha256=
            "5afe973cadc036496cac66f1414ca9be36881423f576db363d83afc9084c0c2f"),
    ("Darwin", "x86_64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.5.0-darwin-x86_64",
            sha256=
            "bbf9c2c03bac48e0514f46db0295027935535d91f6d8dcd960c53393559eab29"),
    ("Darwin", "arm64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.5.0-darwin-arm64",
            sha256=
            "c6b6dc17efcdf13fba484c6fe0b6c3361b888ae7b9573bc25a2dbe8c502448eb"),
    ("Windows", "AMD64"):
        BazelPackage(
            base_uri=None,
            file="bazel-6.5.0-windows-x86_64.exe",
            sha256=
            "6eae8e7f28e1b68b833503d1a58caf139c11e52de19df0d787d974653a0ea4c6"),
}


def download_and_verify_bazel():
  """Downloads a bazel binary from GitHub, verifying its SHA256 hash."""
  package = bazel_packages.get((platform.system(), platform.machine()))
  if package is None:
    return None

  if not os.access(package.file, os.X_OK):
    uri = (package.base_uri or BAZEL_BASE_URI) + package.file
    sys.stdout.write(f"Downloading bazel from: {uri}\n")

    def progress(block_count, block_size, total_size):
      if total_size <= 0:
        total_size = 170**6
      progress = (block_count * block_size) / total_size
      num_chars = 40
      progress_chars = int(num_chars * progress)
      sys.stdout.write("{} [{}{}] {}%\r".format(
          package.file, "#" * progress_chars,
          "." * (num_chars - progress_chars), int(progress * 100.0)))

    tmp_path, _ = urllib.request.urlretrieve(
      uri, None, progress if sys.stdout.isatty() else None
    )
    sys.stdout.write("\n")

    # Verify that the downloaded Bazel binary has the expected SHA256.
    with open(tmp_path, "rb") as downloaded_file:
      contents = downloaded_file.read()

    digest = hashlib.sha256(contents).hexdigest()
    if digest != package.sha256:
      print(
          "Checksum mismatch for downloaded bazel binary (expected {}; got {})."
          .format(package.sha256, digest))
      sys.exit(-1)

    # Write the file as the bazel file name.
    with open(package.file, "wb") as out_file:
      out_file.write(contents)

    # Mark the file as executable.
    st = os.stat(package.file)
    os.chmod(package.file,
             st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  return os.path.join(".", package.file)


def get_bazel_paths(bazel_path_flag):
  """Yields a sequence of guesses about bazel path. Some of sequence elements
  can be None. The resulting iterator is lazy and potentially has a side
  effects."""
  yield bazel_path_flag
  yield shutil.which("bazel")
  yield download_and_verify_bazel()


def get_bazel_path(bazel_path_flag):
  """Returns the path to a Bazel binary, downloading Bazel if not found. Also,
  checks Bazel's version is at least newer than 6.5.0

  A manual version check is needed only for really old bazel versions.
  Newer bazel releases perform their own version check against .bazelversion
  (see for details
  https://blog.bazel.build/2019/12/19/bazel-2.0.html#other-important-changes).
  """
  for path in filter(None, get_bazel_paths(bazel_path_flag)):
    version = get_bazel_version(path)
    if version is not None and version >= (6, 5, 0):
      return path, ".".join(map(str, version))

  print("Cannot find or download a suitable version of bazel."
        "Please install bazel >= 6.5.0.")
  sys.exit(-1)


def get_bazel_version(bazel_path):
  try:
    version_output = shell([bazel_path, "--version"])
  except (subprocess.CalledProcessError, OSError):
    return None
  match = re.search(r"bazel *([0-9\\.]+)", version_output)
  if match is None:
    return None
  return tuple(int(x) for x in match.group(1).split("."))


def get_clang_path_or_exit():
  which_clang_output = shutil.which("clang")
  if which_clang_output:
    # If we've found a clang on the path, need to get the fully resolved path
    # to ensure that system headers are found.
    return str(pathlib.Path(which_clang_output).resolve())
  else:
    print(
        "--use_clang set, but --clang_path is unset and clang cannot be found"
        " on the PATH. Please pass --clang_path directly."
    )
    sys.exit(-1)

def get_clang_major_version(clang_path):
  clang_version_proc = subprocess.run(
      [clang_path, "-E", "-P", "-"],
      input="__clang_major__",
      check=True,
      capture_output=True,
      text=True,
  )
  major_version = int(clang_version_proc.stdout)

  return major_version


def _find_executable(executable: str) -> Optional[str]:
  logging.info("Trying to find path to %s...", executable)
  # Resolving the symlink is necessary for finding system headers.
  if unresolved_path := shutil.which(executable):
    return str(pathlib.Path(unresolved_path).resolve())
  return None


def _find_executable_or_die(
    executable_name: str, executable_path: Optional[str] = None
) -> str:
  """Finds executable and resolves symlinks or raises RuntimeError.

  Resolving symlinks is sometimes necessary for finding system headers.

  Args:
    executable_name: The name of the executable that we want to find.
    executable_path: If not None, the path to the executable.

  Returns:
    The path to the executable we are looking for, after symlinks are resolved.
  Raises:
    RuntimeError: if path to the executable cannot be found.
  """
  if executable_path:
    return str(pathlib.Path(executable_path).resolve(strict=True))
  resolved_path_to_exe = _find_executable(executable_name)
  if resolved_path_to_exe is None:
    raise RuntimeError(
        f"Could not find executable `{executable_name}`! "
        "Please change your $PATH or pass the path directly like"
        f"`--{executable_name}_path=path/to/executable."
    )
  logging.info("Found path to %s at %s", executable_name, resolved_path_to_exe)

  return resolved_path_to_exe


def _get_cuda_compute_capabilities_or_die() -> list[str]:
  """Finds compute capabilities via nvidia-smi or rasies exception.

  Returns:
    list of unique, sorted strings representing compute capabilities:
  Raises:
    RuntimeError: if path to nvidia-smi couldn't be found.
    subprocess.CalledProcessError: if nvidia-smi process failed.
  """
  try:
    nvidia_smi = _find_executable_or_die("nvidia-smi")
    nvidia_smi_proc = subprocess.run(
        [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    # Command above returns a newline separated list of compute capabilities
    # with possible repeats. So we should unique them and sort the final result.
    capabilities = sorted(set(nvidia_smi_proc.stdout.strip().split("\n")))
    logging.info("Found CUDA compute capabilities: %s", capabilities)
    return capabilities
  except (RuntimeError, subprocess.CalledProcessError) as e:
    logging.info(
        "Could not find nvidia-smi, or nvidia-smi command failed. Please pass"
        " capabilities directly using --cuda_compute_capabilities."
    )
    raise e

cBANNER = r"""
CHEMSIM
"""

EPILOG = """
THANKS FOR USING CHEMSIM
"""


def _parse_string_as_bool(s):
  """Parses a string as a boolean argument."""
  lower = s.lower()
  if lower == "true":
    return True
  elif lower == "false":
    return False
  else:
    raise ValueError(f"Expected either 'true' or 'false'; got {s}")


def add_boolean_argument(parser, name, default=False, help_str=None):
  """Creates a boolean flag."""
  group = parser.add_mutually_exclusive_group()
  group.add_argument(
      "--" + name,
      nargs="?",
      default=default,
      const=True,
      type=_parse_string_as_bool,
      help=help_str)
  group.add_argument("--no" + name, dest=name, action="store_false")


def main():
  cwd = os.getcwd()
  parser = argparse.ArgumentParser(
      description="Builds jax-connector from source.", epilog=EPILOG)
  add_boolean_argument(
      parser,
      "verbose",
      default=False,
      help_str="Should we produce verbose debugging output?")
  parser.add_argument(
      "--bazel_path",
      help="Path to the Bazel binary to use. The default is to find bazel via "
      "the PATH; if none is found, downloads a fresh copy of bazel from "
      "GitHub.")
  add_boolean_argument(
      parser,
      "use_clang",
      default = "true",
      help_str=(
          "Should we build using clang as the host compiler? Requires "
          "clang to be findable via the PATH, or a path to be given via "
          "--clang_path."
      ),
  )
  parser.add_argument(
      "--clang_path",
      help=(
          "Path to clang binary to use if --use_clang is set. The default is "
          "to find clang via the PATH."
      ),
  )
  add_boolean_argument(
      parser,
      "enable_cuda",
      help_str="Should we build with CUDA enabled? Requires CUDA and CuDNN."
  )
  add_boolean_argument(
      parser,
      "build_gpu_pjrt_plugin",
      default=False,
      help_str=(
          "Are we building the cuda pjrt plugin?"
      ),
  )
  add_boolean_argument(
      parser,
      "build_cpu_pjrt_plugin",
      default=False,
      help_str=(
          "Are we building the cpu pjrt plugin?"
      ),
  )
  add_boolean_argument(
      parser,
      "load_gpu_pjrt_plugin",
      default=False,
      help_str=(
          "Load the GPU PJRT plugin from the jax wheels."
      ),
  )
  add_boolean_argument(
      parser,
      "use_cuda_nvcc",
      default=True,
      help_str=(
          "Should we build CUDA code using NVCC compiler driver? The default value "
          "is true. If --nouse_cuda_nvcc flag is used then CUDA code is built "
          "by clang compiler."
      ),
  )
  add_boolean_argument(
      parser,
      "enable_nccl",
      default=True,
      help_str="Should we build with NCCL enabled? Has no effect for non-CUDA "
               "builds.")
  parser.add_argument(
      "--cuda_version",
      default=None,
      help="CUDA toolkit version, e.g., 12.3.2")
  parser.add_argument(
      "--cudnn_version",
      default=None,
      help="CUDNN version, e.g., 8.9.7.29")
  parser.add_argument(
      "--build_cuda_with_clang",
      action="store_true",
      help="""
          Should CUDA code be compiled using Clang? The default behavior is to
          compile CUDA with NVCC.
          """,
  )


  # Caution: if changing the default list of CUDA capabilities, you should also
  # update the list in .bazelrc, which is used for wheel builds.
  parser.add_argument(
      "--cuda_compute_capabilities",
      default=None,
      help="A comma-separated list of CUDA compute capabilities to support.")

  parser.add_argument(
      "--bazel_startup_options",
      action="append", default=[],
      help="Additional startup options to pass to bazel.")
  parser.add_argument(
      "--bazel_options",
      action="append", default=[],
      help="Additional options to pass to the main Bazel command to be "
           "executed, e.g. `run`.")

  parser.add_argument(
      "--output_path",
      default=os.path.join(cwd, "out"),
      help="Directory to which the compilation outputs should be written")

  parser.add_argument(
      "--target_cpu",
      default=None,
      help="CPU platform to target. Default is the same as the host machine. "
           "Currently supported values are 'darwin_arm64' and 'darwin_x86_64'.")
  parser.add_argument(
      "--target_cpu_features",
      choices=["release", "native", "default"],
      default="native",
      help="""
        What CPU features should we target? Release enables CPU features that
        should be enabled for a release build, which on x86-64 architectures
        enables AVX. Native enables -march=native, which generates code targeted
        to use all features of the current machine. Default means don't opt-in
        to any architectural features and use whatever the C compiler generates
        by default.
        """,
  )

  parser.add_argument(
    "--configure_only",
    action="store_true",
    help="""
      If true, writes the Bazel options to the .jax_configure.bazelrc file but
      does not build the artifacts.
      """,
  )

  args = parser.parse_args()

  logging.basicConfig()
  if args.verbose:
    logger.setLevel(logging.DEBUG)

  arch = platform.machine()
  os_name = platform.system().lower()

  cpus = {
      "darwin_arm64": "arm64",
      "darwin_x86_64": "x86_64",
      # "ppc": "ppc64le",
      # "aarch64": "aarch64",
  }

  build_options = args.bazel_options

  output_path = os.path.abspath(args.output_path)
  config_path = os.path.abspath("config.bazelrc")
  os.chdir(os.path.dirname(__file__ or args.prog) or '.')

  # Find a working Bazel.
  bazel_path, bazel_version = get_bazel_path(args.bazel_path)
  print(f"Bazel binary path: {bazel_path}")
  print(f"Bazel version: {bazel_version}")


  # Enable cross-compilation
  target_cpu = (
      cpus[args.target_cpu] if args.target_cpu is not None else arch
  )

  clang_path = ""
  if args.use_clang:
      clang_path = args.clang_path or get_clang_path_or_exit()
      clang_major_version = get_clang_major_version(clang_path)
      logging.debug(
          "Using Clang as the compiler, clang path: %s, clang version: %s",
          clang_path,
          clang_major_version,
      )

      build_options.append(
          f"--action_env=CLANG_COMPILER_PATH={clang_path}")
      build_options.append(f"--repo_env=CC={clang_path}")
      build_options.append(f"--repo_env=BAZEL_COMPILER={clang_path}")

      if clang_major_version >= 16:
          # Enable clang settings that are needed for the build to work with newer
          # versions of Clang.
          build_options.append("--config=clang")
      if clang_major_version < 19:
          build_options.append(
              "--define=xnn_enable_avxvnniint8=false")

  else:
      logging.debug("Use Clang: False")

  if args.target_cpu_features == "release":
      if arch in ["x86_64", "AMD64"]:
          logging.debug(
              "Using release cpu features: --config=avx_%s",
              "windows" if os_name == "windows" else "posix",
          )
          build_options.append(
              "--config=avx_windows"
              if os_name == "windows"
              else "--config=avx_posix"
          )
  elif args.target_cpu_features == "native":
      if os_name == "windows":
          logger.warning(
              "--target_cpu_features=native is not supported on Windows;"
              " ignoring."
          )
      else:
          logging.debug("Using native cpu features: --config=native_arch_posix")
          build_options.append("--config=native_arch_posix")
  else:
      logging.debug("Using default cpu features")


  if args.enable_cuda:
      build_options.append("--config=cuda")
      build_options.append(
          f"--action_env=CLANG_CUDA_COMPILER_PATH={clang_path}"
      )

      if args.build_cuda_with_clang:
        logging.debug("Building CUDA with Clang")
        build_options.append("--config=build_cuda_with_clang")
      else:
        logging.debug("Building CUDA with NVCC")
        build_options.append("--config=build_cuda_with_nvcc")

      if args.cuda_version:
          logging.debug("Hermetic CUDA version: %s", args.cuda_version)
          build_options.append(
              f"--repo_env=HERMETIC_CUDA_VERSION={args.cuda_version}"
          )
      if args.cudnn_version:
          logging.debug("Hermetic cuDNN version: %s", args.cudnn_version)
          build_options.append(
              f"--repo_env=HERMETIC_CUDNN_VERSION={args.cudnn_version}"
          )
      if args.cuda_compute_capabilities:
          logging.debug(
              "Hermetic CUDA compute capabilities: %s",
              args.cuda_compute_capabilities,
          )
          build_options.append(
              f"--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES={args.cuda_compute_capabilities}"
          )


  with open("config.bazelrc", "w") as f:
      f.write(__BAZELRC)
      f.write("\n".join(build_options))

  if args.configure_only:
    return

  print("\nBuilding XLA and installing it in the jaxlib source tree...")

  command_base = (
    bazel_path,
    f"--bazelrc={config_path}",
    *args.bazel_startup_options,
    f"--output_base={output_path}",
    "build",
    "--verbose_failures=true",
    f"--compilation_mode=opt",
    f"--copt=-O3",
    *build_options,
  )

  if args.build_gpu_pjrt_plugin:
      assert args.enable_cuda, "Must enable cuda to build pjrt plugin."

      build_pjrt_plugin_command = [
          *command_base,
          "@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so",
          "--",
      ]

      print(" ".join(build_pjrt_plugin_command))
      shell(build_pjrt_plugin_command)
      out_dir = pathlib.Path("./lib")
      out_dir.mkdir(exist_ok=True, parents=True)
      (out_dir / f"pjrt_plugin.xla_cuda{args.cuda_version.split('.')[0]}.so").write_bytes(
          pathlib.Path("./bazel-bin/external/xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so").read_bytes()
      )
  elif args.build_cpu_pjrt_plugin:
      raise NotImplementedError(
            "Building CPU PJRT plugin is not supported yet."
      )

      build_pjrt_plugin_command = [
          *command_base,
          "@xla//xla/pjrt/c:pjrt_c_api_cpu_plugin.so",
          "--",
      ]

      print(" ".join(build_pjrt_plugin_command))
      shell(build_pjrt_plugin_command)
      out_dir = pathlib.Path("./lib")
      out_dir.mkdir(exist_ok=True, parents=True)
      (out_dir / f"pjrt_plugin.xla_cpu.so").write_bytes(
          pathlib.Path("./bazel-bin/external/xla/xla/pjrt/c/pjrt_c_api_cpu_plugin.so").read_bytes()
      )
  elif args.load_gpu_pjrt_plugin:
      # Loads a prebuilt pjrt plugin from the jaxlib wheels.
      out_dir = pathlib.Path("./lib")
      load_pjrt_plugin_libraries(out_dir)

  else:
      build_cpu_wheel_command = [
          *command_base,
          "//connector:libconnector.so", "--",
      ]
      print(" ".join(build_cpu_wheel_command))
      shell(build_cpu_wheel_command)
      out_dir = pathlib.Path("./lib")
      out_dir.mkdir(exist_ok=True, parents=True)
      (out_dir / "libconnector.so").write_bytes(
          pathlib.Path("./bazel-bin/connector/libconnector.so").read_bytes()
      )


if __name__ == "__main__":
  main()
