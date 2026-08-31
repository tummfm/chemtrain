"""Measure one MD22 MACE fine-tuning configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

import chemtrain
from chemtrain import util
from finetune_md22 import FineTuneConfig, main as fine_tune


def main() -> None:
    """Run one backend and write steady-state epoch throughput as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallelism", choices=("single", "mpi", "jax"), required=True
    )
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--discard-epochs", type=int, default=1)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", default="medium-0b3")
    parser.add_argument(
        "--async-dataloading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.discard_epochs >= args.epochs:
        raise ValueError("At least one measured epoch must remain.")

    trainer = fine_tune(FineTuneConfig(
        global_batch=args.global_batch,
        epochs=args.epochs,
        model_version=args.model_version,
        workdir=args.workdir,
        parallelism=args.parallelism,
        async_dataloading=args.async_dataloading,
        make_plot=False,
    ))
    measured_minutes = trainer.update_times[args.discard_epochs:]
    if trainer.parallel_context.mode == "mpi":
        comm = util.get_communicator()
        mpi = util.get_mpi()
        measured_minutes = [
            comm.allreduce(duration, op=mpi.MAX)
            for duration in measured_minutes
        ]

    epoch_seconds = [60.0 * value for value in measured_minutes]
    mean_seconds = sum(epoch_seconds) / len(epoch_seconds)
    with h5py.File(args.workdir / "md22-nanotube-train.h5", "r") as handle:
        sample_count, atom_count = handle["R"].shape[:2]
    used_samples = sample_count - sample_count % args.global_batch
    updates_per_epoch = used_samples // args.global_batch
    if updates_per_epoch == 0:
        raise ValueError(
            "global_batch must not exceed the number of training samples."
        )

    result = {
        "backend": args.parallelism,
        "devices": trainer.parallel_context.size,
        "global_batch": args.global_batch,
        "atoms_per_structure": atom_count,
        "measured_epochs": len(epoch_seconds),
        "epoch_seconds": epoch_seconds,
        "seconds_per_epoch": mean_seconds,
        "seconds_per_update": mean_seconds / updates_per_epoch,
        "samples_per_second": used_samples / mean_seconds,
        "atoms_per_second": used_samples * atom_count / mean_seconds,
        "model": f"MACE-MP {args.model_version}",
        "async_dataloading": args.async_dataloading,
        "validation_loss_history": [
            float(value) for value in trainer.val_losses
        ],
        "validation_target_loss_history": {
            key: [float(value) for value in values]
            for key, values in trainer.val_target_losses.items()
        },
        "chemtrain_source": str(Path(chemtrain.__file__).resolve()),
    }
    if trainer.parallel_context.is_root:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
