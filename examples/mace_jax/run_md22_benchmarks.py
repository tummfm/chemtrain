"""Run the single-GPU, JAX, and MPI MD22 benchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args():
    """Read benchmark-matrix options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs=2, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--discard-epochs", type=int, default=1)
    parser.add_argument("--model-version", default="medium-0b3")
    parser.add_argument(
        "--mpi-prefix", type=Path, default=Path("/opt/openmpi-4.1.8-cuda")
    )
    return parser.parse_args()


def main() -> None:
    """Run each configuration in a fresh process and compare throughput."""
    args = parse_args()
    if args.global_batch % len(args.gpus):
        raise ValueError("global_batch must be divisible by the GPU count.")

    output_directory = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).with_name("benchmark_md22.py")
    repository = Path(__file__).resolve().parents[2]
    workdir = output_directory / "work"
    environment = os.environ.copy()
    mpi_bin = args.mpi_prefix / "bin"
    mpi_lib = args.mpi_prefix / "lib"

    python_paths = [str(repository)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    environment["PATH"] = os.pathsep.join((str(mpi_bin), environment["PATH"]))

    library_paths = [str(mpi_lib)]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)

    results = []
    for asynchronous in (False, True):
        loading_name = "async" if asynchronous else "sync"
        loading_option = (
            "--async-dataloading" if asynchronous
            else "--no-async-dataloading"
        )
        common = [
            str(benchmark),
            "--global-batch", str(args.global_batch),
            "--epochs", str(args.epochs),
            "--discard-epochs", str(args.discard_epochs),
            "--workdir", str(workdir),
            "--model-version", args.model_version,
            loading_option,
        ]

        for backend, devices in (
            ("single", args.gpus[0]),
            ("jax", ",".join(args.gpus)),
        ):
            output = output_directory / f"{backend}-{loading_name}.json"
            command = [
                sys.executable, *common,
                "--parallelism", backend,
                "--output", str(output),
            ]
            run_environment = environment | {"CUDA_VISIBLE_DEVICES": devices}
            subprocess.run(command, check=True, env=run_environment)
            results.append(json.loads(output.read_text()))

        output = output_directory / f"mpi-{loading_name}.json"
        mpi_program = [
            sys.executable, *common,
            "--parallelism", "mpi",
            "--output", str(output),
        ]
        command = [
            str(mpi_bin / "mpirun"), "--bind-to", "none",
            "-np", "1", "env", f"CUDA_VISIBLE_DEVICES={args.gpus[0]}",
            *mpi_program,
            ":",
            "-np", "1", "env", f"CUDA_VISIBLE_DEVICES={args.gpus[1]}",
            *mpi_program,
        ]
        mpi_environment = environment | {"MPI4JAX_USE_CUDA_MPI": "1"}
        subprocess.run(command, check=True, env=mpi_environment)
        results.append(json.loads(output.read_text()))

    baselines = {
        result["async_dataloading"]: result["atoms_per_second"]
        for result in results if result["backend"] == "single"
    }
    for result in results:
        result["speedup_over_single"] = (
            result["atoms_per_second"]
            / baselines[result["async_dataloading"]]
        )

    report = output_directory / "comparison.json"
    report.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
