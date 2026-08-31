# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This file contains several Trainer classes as a quickstart for users."""
import pickle
import time

from jax import numpy as jnp, tree_util, jit

from chemtrain import (util)
from chemtrain.trainers import base as tt


class EnsembleOfModels(tt.ProbabilisticFMTrainerTemplate):
    """Train an ensemble of models by starting optimization from different
    initial parameter sets, for use in uncertainty quantification applications.

    Example:

        .. code-block:: python

           trainer_list = []
           for i in range(4):
               trainer_list.append(trainers.ForceMatching(...))
           trainer_ensemble = trainers.EnsembleOfModels(trainer_list)

           trainer_ensemble.train(*args, **kwargs)
           trained_params = trainer_ensemble.list_of_params

    """
    def __init__(self, trainers, ref_energy_fn_template=None):
        super().__init__(None, ref_energy_fn_template)
        self.trainers = trainers

    def train(self, *args, **kwargs):
        for i, trainer in enumerate(self.trainers):
            print(f"---------Starting trainer {i}-----------")
            trainer.train(*args, **kwargs)
        print("Finished training all models.")

    @property
    def params(self):
        return util.tree_stack(self.list_of_params)

    @params.setter
    def params(self, loaded_params):
        for i, params in enumerate(loaded_params):
            self.trainers[i].params = params

    @property
    def list_of_params(self):
        return [trainer.best_params for trainer in self.trainers]


class InterleaveTrainers(tt.TrainerInterface):
    """Interleaves updates to train models using multiple algorithms.

    This special trainer allows to train models simultaneously with different
    algorithms.

    Example:

        .. code-block::

            # First initialize the base-trainers, e.g.
            fm_trainer = trainers.ForceMatching(...)

            difftre_trainer = trainers.Difftre(...)
            difftre_trainer.add_statepoint(...)

            # Now combine the trainers. The trainers are executed in the
            # order in which they are added

            trainer = trainers.InterleaveTrainers('checkpoint_folder',
                                                  energy_fn_template,
                                                  full_checkpoint=False)

            # Force matching should run 10 epochs before difftre runs 2 epochs
            trainer.add_trainer(fm_trainer, num_updates=10, name='Force Matching')
            trainer.add_trainer(difftre_trainer, num_updates=2, name='DiffTRe')

            trainer.train(100, checkpoint_frequency=10)

    Args:
        sequential: Start the next trainer directly with the optimized
            parameters of the previous trainer. In the non-sequential case,
            the trainers start their epoch on the same parameter set and
            the final update is a weighted sum of both updates.
        checkpoint_base_path: Location to store checkpoints of the trainers.
        reference_energy_fn_template: Energy function template to optionally
            return an energy function with current parameters.
        full_checkpoint: Store the complete trainer or important properties
            only.

    """


    def __init__(self,
                 sequential = True,
                 checkpoint_base_path = "checkpoints",
                 reference_energy_fn_template=None,
                 full_checkpoint=False):
        super().__init__(checkpoint_base_path, reference_energy_fn_template,
                         full_checkpoint)
        self.sequential = sequential
        self._trainers = []
        self._epoch = 0

    def add_trainer(self, trainer, num_updates: int = 1, name: str = "trainer",
                    weight: float = 1.0, **trainer_kwargs):
        """Adds a trainer to the combined training.

        The trainers are executed in the order they are added to this instance.
        It is possible to specify how many epochs each trainer should train
        before the next trainer starts again.

        Args:
            trainer: Trainer to add to the chain.
            num_updates: Consecutive updates of the trainer in one epoch of the
                interleaved trainer.
            name: Display name of the trainer.
            weight: Weight for the interpolated update of the parameters.
            trainer_kwargs: Additional arguments for the training method
                of the trainer.

        """
        self._trainers.append(
            {"trainer": trainer, "num_updates": num_updates, "name": name,
             "kwargs": trainer_kwargs, "weight": weight}
        )

    @property
    def params(self):
        return self._trainers[-1]["trainer"].params

    @params.setter
    def params(self, params):
        for trainer in self._trainers:
            trainer["trainer"].params = params

    @property
    def _all_params(self):
        return [t["trainer"].params for t in self._trainers]

    @property
    def _all_weights(self):
        return [t["weight"] for t in self._trainers]

    def _init_interpolated_update(self):
        weights = jnp.asarray(self._all_weights)
        weights /= jnp.sum(weights)
        @jit
        def update(parameters):
            # Scale the parameters
            structure = tree_util.tree_structure(parameters[0])
            leaves = [tree_util.tree_leaves(t) for t in parameters]
            concat = [jnp.concatenate(l) for l in zip(*leaves)]
            summed = [jnp.sum(weights * l, axis=0) for l in concat]
            return tree_util.tree_unflatten(structure, summed)
        return update

    def train(self, epochs, checkpoint_frequency=None):
        """Train model with combined algorithms.

        Args:
            epochs: Number of epochs, where one epoch can contain multiple
                epochs for each added trainer.
            checkpoint_frequency: Save a checkpoint in the given frequency.

        """
        interpolated_update = self._init_interpolated_update()
        self._converged = False
        start_epoch = self._epoch
        end_epoch = start_epoch + epochs
        for e in range(start_epoch, end_epoch):
            start = time.time()
            for t, trainer in enumerate(self._trainers):
                print(f"---------Starting trainer {trainer['name']} for {trainer['num_updates']} updates -----------")
                trainer["trainer"].train(trainer["num_updates"], **trainer["kwargs"])

                next = (t + 1) % len(self._trainers)

                if self.sequential:
                    # Pass updated parameters to the next trainer
                    self._trainers[next]["trainer"].params = trainer["trainer"].params
            if not self.sequential:
                # Update the parameters of all trainers with a weighted sum of
                # the individual parameters
                self.params = interpolated_update(self.params)

            duration = (time.time() - start) / 60.
            self._epoch += 1
            print(f"Finished epoch {e} for all trainers in {duration : .2f} minutes.")
            self._dump_checkpoint_occasionally(frequency=checkpoint_frequency)

    def move_to_device(self):
        for trainer in self._trainers:
            trainer["trainer"].move_to_device()

    def save_trainer(self, save_path, format=".pkl"):
        data = {}
        for t, trainer in enumerate(self._trainers):
            number = str(t + 1).rjust(3, "0")
            key = "trainer_{0}_{1}".format(trainer["name"], number)
            data[key] = trainer["trainer"].save_trainer(None, format="none")

        if format == ".pkl":
            with open(save_path, "wb") as pickle_file:
                pickle.dump(data, pickle_file)
        elif format == "none":
            return data
