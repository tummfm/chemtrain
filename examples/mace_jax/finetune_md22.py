"""Fine-tune a MACE foundation model on the MD22 nanotube dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from urllib import request


# Configure JAX memory before importing JAX or a package that imports JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

from chemtrain import config as chemtrain_config
from chemtrain import util
from chemtrain.data import preprocessing


DATASET_URL = (
    "https://sgdml.org/secure_proxy.php?file="
    "repo/datasets/md22_double-walled_nanotube.npz"
)
ANGSTROM_TO_NM = 0.1
KCAL_TO_KJ = 4.184
EV_TO_KJ_PER_MOL = constants.electron_volt * constants.Avogadro / 1000.0


@dataclass
class FineTuneConfig:
    """Options for MD22 fine-tuning.

    ``global_batch`` is shared by all devices or MPI ranks. MD22 stores
    positions in Angstrom, energies in kcal/mol, and forces in
    kcal/(mol Angstrom); the prepared HDF5 files use chemtrain's nm and
    kJ/mol convention.
    """

    global_batch: int = 16
    cache_size: int = 4
    epochs: int = 3
    learning_rate: float = 1.0e-5
    model_version: str = "medium-0b3"
    workdir: Path = Path("mace-md22-run")
    parallelism: str = "auto"
    async_dataloading: bool = True
    seed: int = 0
    plot_samples: int = 64
    make_plot: bool = True


def parse_args() -> FineTuneConfig:
    """Read command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--model-version", default="medium-0b3")
    parser.add_argument("--workdir", type=Path, default=Path("mace-md22-run"))
    parser.add_argument(
        "--parallelism",
        choices=("auto", "single", "mpi", "jax"),
        default="auto",
    )
    parser.add_argument(
        "--async-dataloading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot-samples", type=int, default=64)
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, default=True
    )
    args = vars(parser.parse_args())
    args["make_plot"] = args.pop("plot")
    return FineTuneConfig(**args)


def prepare_dataset(workdir: Path, seed: int) -> dict[str, Path]:
    """Download MD22 once and create deterministic float32 HDF5 splits."""
    source = workdir / "md22_double-walled_nanotube.npz"
    paths = {
        "training": workdir / "md22-nanotube-train.h5",
        "validation": workdir / "md22-nanotube-validation.h5",
        "testing": workdir / "md22-nanotube-test.h5",
    }

    error = None
    if util.is_root():
        try:
            workdir.mkdir(parents=True, exist_ok=True)
            prepared = all(path.exists() for path in paths.values())
            if prepared:
                for path in paths.values():
                    with h5py.File(path, "r") as handle:
                        if handle.attrs.get("split_seed") != seed:
                            raise ValueError(
                                "Existing MD22 splits use a different seed. "
                                "Choose a new workdir or restore the original seed."
                            )
                        if handle.attrs.get("dataset") != "MD22 nanotube":
                            raise ValueError(
                                f"Unexpected prepared dataset in {path}."
                            )
            else:
                if not source.exists():
                    request.urlretrieve(DATASET_URL, source)

                with np.load(source) as raw:
                    if (
                        raw["r_unit"].item() != "Ang"
                        or raw["e_unit"].item() != "kcal/mol"
                    ):
                        raise ValueError("Unexpected units in the MD22 dataset.")

                    positions = (
                        np.asarray(raw["R"], dtype=np.float32) * ANGSTROM_TO_NM
                    )
                    energies = (
                        np.asarray(raw["E"], dtype=np.float32).reshape(-1)
                        * KCAL_TO_KJ
                    )
                    forces = np.asarray(raw["F"], dtype=np.float32)
                    forces *= KCAL_TO_KJ / ANGSTROM_TO_NM
                    species = np.asarray(raw["z"], dtype=np.int32)

                split_data = preprocessing.train_val_test_split(
                    {"R": positions, "U": energies, "F": forces},
                    train_ratio=0.70,
                    val_ratio=0.15,
                    shuffle=True,
                    shuffle_seed=seed,
                )

                for stage, data in zip(paths, split_data):
                    count = data["R"].shape[0]
                    chunk = min(16, count)
                    with h5py.File(paths[stage], "w") as handle:
                        handle.create_dataset(
                            "R",
                            data=data["R"],
                            chunks=(chunk, *positions.shape[1:]),
                        )
                        handle.create_dataset(
                            "U", data=data["U"], chunks=(chunk,)
                        )
                        handle.create_dataset(
                            "F",
                            data=data["F"],
                            chunks=(chunk, *forces.shape[1:]),
                        )
                        handle.create_dataset(
                            "species",
                            data=np.broadcast_to(species, (count, species.size)),
                            chunks=(chunk, species.size),
                        )
                        handle.attrs["dataset"] = "MD22 nanotube"
                        handle.attrs["split_seed"] = seed
                        handle.attrs["position_unit"] = "nm"
                        handle.attrs["energy_unit"] = "kJ/mol"
                        handle.attrs["force_unit"] = "kJ/(mol nm)"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    # All ranks check the root result before waiting for the prepared files.
    if util.use_mpi():
        communicator = util.get_communicator()
        assert communicator is not None
        error = communicator.bcast(error, root=0)
    elif jax.process_count() > 1:
        from jax.experimental import multihost_utils

        failed = multihost_utils.broadcast_one_to_all(
            np.asarray(error is not None), is_source=util.is_root()
        )
        if bool(failed):
            error = error or "MD22 preparation failed on the root process."
    if error is not None:
        raise RuntimeError(f"MD22 dataset preparation failed: {error}")

    if util.use_mpi():
        communicator.Barrier()
    elif jax.process_count() > 1:
        multihost_utils.sync_global_devices("md22-hdf5-ready")

    return paths


def plot_predictions(
    predictions: dict,
    reference: dict,
    output: Path,
    max_force_points: int = 100_000,
) -> None:
    """Plot energy and force predictions in eV/atom and eV/Angstrom."""
    atom_count = reference["R"].shape[1]
    predicted_energy = np.asarray(predictions["U"]).reshape(-1)
    reference_energy = np.asarray(reference["U"]).reshape(-1)
    predicted_energy /= atom_count * EV_TO_KJ_PER_MOL
    reference_energy /= atom_count * EV_TO_KJ_PER_MOL

    # kJ/(mol nm) multiplied by nm/Angstrom and divided by kJ/(mol eV)
    # gives eV/Angstrom without changing the force sign.
    force_scale = ANGSTROM_TO_NM / EV_TO_KJ_PER_MOL
    predicted_force = np.asarray(predictions["F"]).reshape(-1) * force_scale
    reference_force = np.asarray(reference["F"]).reshape(-1) * force_scale
    if predicted_force.size > max_force_points:
        selection = np.linspace(
            0, predicted_force.size - 1, max_force_points, dtype=np.int64
        )
        predicted_force = predicted_force[selection]
        reference_force = reference_force[selection]

    energy_mae = np.mean(np.abs(predicted_energy - reference_energy))
    force_mae = np.mean(np.abs(predicted_force - reference_force))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].scatter(reference_energy, predicted_energy, s=8, alpha=0.6)
    axes[0].set(
        xlabel="Reference energy [eV/atom]",
        ylabel="Predicted energy [eV/atom]",
        title=f"Energy MAE: {energy_mae * 1000:.1f} meV/atom",
    )
    axes[1].scatter(reference_force, predicted_force, s=2, alpha=0.2)
    axes[1].set(
        xlabel="Reference force [eV/Angstrom]",
        ylabel="Predicted force [eV/Angstrom]",
        title=f"Force MAE: {force_mae * 1000:.1f} meV/Angstrom",
    )
    figure.savefig(output)
    plt.close(figure)


def main(config: FineTuneConfig | None = None):
    """Run MD22 fine-tuning and return the chemtrain trainer."""
    from jax_md import partition, space
    import optax

    from jax_md_mod import custom_partition

    from chemtrain.compose import mace_jax as mace_compose
    from chemtrain.data.data_loaders import HDF5ParallelDataLoader
    from chemtrain.trainers import ForceMatching

    config = parse_args() if config is None else config
    if config.global_batch < 1:
        raise ValueError("global_batch must be positive.")
    if config.plot_samples < 1:
        raise ValueError("plot_samples must be positive.")

    chemtrain_config.update(async_dataloading=config.async_dataloading)
    paths = prepare_dataset(config.workdir, config.seed)

    # Convert the MACE-MP foundation model to chemtrain's energy interface.
    displacement_fn, _ = space.free()
    torch_model, model_config = mace_compose.load_foundational_model(
        family="mp", version=config.model_version
    )
    cutoff = float(model_config["r_max"]) * ANGSTROM_TO_NM
    variables, apply_fn = mace_compose.mace_jax_neighborlist_from_torch(
        model_config,
        torch_model,
        displacement_fn,
        max_edge_multiplier=1.25,
        per_particle=False,
        scale_pos=ANGSTROM_TO_NM,
        scale_pot=EV_TO_KJ_PER_MOL,
        species_mapping=mace_compose.AtomicNumberMapping(max_number=100),
        use_custom_batch_fn=True,
    )
    neighbor_fn = custom_partition.masked_neighbor_list(
        displacement_fn,
        cutoff,
        dr_threshold=None,
        capacity_multiplier=1.10,
        format=partition.Sparse,
    )

    def energy_fn_template(params):
        """Build a MACE energy function in kJ/mol for positions in nm."""
        model_variables = {**variables, "params": params}

        def energy_fn(position, neighbor, species, **kwargs):
            return apply_fn(
                model_variables,
                position,
                neighbor["default"],
                species=species,
                **kwargs,
            )

        return energy_fn

    with h5py.File(paths["training"], "r") as handle:
        sample_count = handle["R"].shape[0]
    updates = max(1, config.epochs * sample_count // config.global_batch)
    schedule = optax.cosine_decay_schedule(config.learning_rate, updates)
    trainer = ForceMatching(
        variables["params"],
        optax.adamw(schedule, weight_decay=1.0e-3),
        energy_fn_template,
        neighbor_fns=neighbor_fn,
        batch=config.global_batch,
        batch_cache=config.cache_size,
        gammas={"U": 1.0e-3, "F": 1.0e-2},
        parallelism=config.parallelism,
        checkpoint_path=config.workdir / "checkpoints",
        log_file=None,
    )
    trainer.set_loader(HDF5ParallelDataLoader(paths["training"]), "training")
    trainer.set_loader(
        HDF5ParallelDataLoader(paths["validation"]),
        "validation",
        include_all=True,
    )
    trainer.set_loader(
        HDF5ParallelDataLoader(paths["testing"]),
        "testing",
        include_all=True,
    )

    trainer.train(config.epochs, checkpoint_freq=1)
    validation_loss, validation_targets = trainer.evaluate("validation")
    if trainer.parallel_context.is_root:
        print("Validation loss:", validation_loss)
        print("Validation targets:", validation_targets)

    # Prediction plotting deliberately uses the public trainer interface.
    if config.make_plot:
        with h5py.File(paths["testing"], "r") as handle:
            count = min(config.plot_samples, handle["R"].shape[0])
            reference = {
                key: np.asarray(value[:count])
                for key, value in handle.items()
            }
        predictions = trainer.predict(
            reference, batch_size=min(config.global_batch, count)
        )
        if trainer.parallel_context.is_root:
            plot_predictions(
                predictions, reference, config.workdir / "predictions.pdf"
            )

    return trainer


if __name__ == "__main__":
    main()
