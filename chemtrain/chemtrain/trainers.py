"""This file contains several Trainer classes as a quickstart for users."""
import copy
import warnings

from jax import device_count, jit, value_and_grad, numpy as jnp
from jax_sgmc import data
import numpy as onp

from chemtrain import util, force_matching, traj_util, reweighting


class ForceMatching(util.MLETrainerTemplate):
    """Force-matching trainer.

    This implementation assumes a constant number of particles per box and
    constant box sizes for each snapshot.
    If this is not the case, please use the ForceMatchingPrecomputed trainer
    based on padded sparse neighborlists.
    Caution: Currently neighborlist overflow is not checked.
    Make sure to build nbrs_init large enough.
    """
    def __init__(self, init_params, energy_fn_template, nbrs_init,
                 optimizer, position_data, energy_data=None, force_data=None,
                 virial_data=None, box_tensor=None, gamma_f=1., gamma_p=1.e-6,
                 batch_per_device=1, batch_cache=10, train_ratio=0.875,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):

        # setup dataset
        self.n_devices, self.batches_per_epoch, self.get_train_batch, \
            self.get_val_batch, self.train_batch_state, self.val_batch_state, \
            include_virial, self.training_dict_keys = \
            self._process_dataset(position_data, train_ratio, energy_data,
                                  force_data, virial_data, box_tensor,
                                  batch_per_device, batch_cache)
        self.train_losses, self.val_losses = [], []
        self.best_params = None
        self.best_val_loss = 1.e16

        opt_state = optimizer.init(init_params)  # initialize optimizer state

        # replicate params and optimizer states for pmap
        init_params = util.tree_replicate(init_params, self.n_devices)
        opt_state = util.tree_replicate(opt_state, self.n_devices)

        init_state = util.TrainerState(params=init_params,
                                       opt_state=opt_state)

        checkpoint_path = 'output/force_matching/' + str(checkpoint_folder)
        super().__init__(optimizer, init_state, checkpoint_path,
                         checkpoint_format, energy_fn_template)

        self.grad_fns = force_matching.init_update_fns(
            energy_fn_template, nbrs_init, optimizer, gamma_f=gamma_f,
            gamma_p=gamma_p, box_tensor=box_tensor,
            include_virial=include_virial
        )

    def update_dataset(self, position_data, train_ratio=0.875, energy_data=None,
                       force_data=None, virial_data=None, box_tensor=None,
                       batch_per_device=1, batch_cache=10, **grad_fns_kwargs):
        """Allows changing dataset on the fly, which is particularly
        useful for active learning applications.
        """
        self.n_devices, self.batches_per_epoch, self.get_train_batch, \
            self.get_val_batch, self.train_batch_state, self.val_batch_state, \
            include_virial, training_dict_keys = \
            self._process_dataset(position_data,
                                  train_ratio,
                                  energy_data,
                                  force_data,
                                  virial_data,
                                  box_tensor,
                                  batch_per_device,
                                  batch_cache)

        if training_dict_keys != self.training_dict_keys:
            # the target quantities changes with respect to initialization
            # we need to re-initialize grad_fns
            self.grad_fns = force_matching.init_update_fns(
                include_virial=include_virial, **grad_fns_kwargs)

    @staticmethod
    def _process_dataset(position_data, train_ratio=0.875, energy_data=None,
                         force_data=None, virial_data=None, box_tensor=None,
                         batch_per_device=1, batch_cache=10):
        # Default train_ratio represents 70-10-20 split, when 20 % test
        # data was already deducted
        n_devices = device_count()
        batch_size = n_devices * batch_per_device
        train_set_size = position_data.shape[0]
        train_size = int(train_set_size * train_ratio)
        batches_per_epoch = train_size // batch_size

        # pylint: disable=unbalanced-tuple-unpacking
        r_train, r_val = onp.split(position_data, [train_size])
        train_dict = {'R': r_train}
        val_dict = {'R': r_val}
        if energy_data is not None:
            u_train, u_val = onp.split(energy_data, [train_size])
            train_dict['U'] = u_train
            val_dict['U'] = u_val
        if force_data is not None:
            f_train, f_val = onp.split(force_data, [train_size])
            train_dict['F'] = f_train
            val_dict['F'] = f_val
        if virial_data is not None:
            include_virial = True
            assert box_tensor is not None, ('If the virial is to be matched, '
                                            'box_tensor is a mandatory input.')
            p_train, p_val = onp.split(virial_data, [train_size])
            train_dict['p'] = p_train
            val_dict['p'] = p_val
        else:
            include_virial = False

        train_loader = data.NumpyDataLoader(**train_dict)
        val_loader = data.NumpyDataLoader(**val_dict)
        init_train_batch, get_train_batch = data.random_reference_data(
            train_loader, batch_cache, batch_size)
        init_val_batch, get_val_batch = data.random_reference_data(
            val_loader, batch_cache, batch_size)
        train_batch_state = init_train_batch()
        val_batch_state = init_val_batch()
        return n_devices, batches_per_epoch, get_train_batch, get_val_batch, \
            train_batch_state, val_batch_state, include_virial, \
            train_dict.keys()

    @property
    def params(self):
        single_params = util.tree_get_single(self.state.params)
        return single_params

    @params.setter
    def params(self, loaded_params):
        replicated_params = util.tree_replicate(loaded_params, self.n_devices)
        self.state = self.state.replace(params=replicated_params)

    def _get_batch(self):
        for _ in range(self.batches_per_epoch):
            self.train_batch_state, train_batch = \
                self.get_train_batch(self.train_batch_state)
            self.val_batch_state, val_batch = \
                self.get_val_batch(self.val_batch_state)
            train_batch = util.tree_split(train_batch, self.n_devices)
            val_batch = util.tree_split(val_batch, self.n_devices)
            yield train_batch, val_batch

    def _update(self, batch):
        """Function to iterate, optimizing parameters and saving
        training and validation loss values.
        """
        # both jitted functions stored together to delete them for checkpointing
        update_fn, batched_loss_fn = self.grad_fns

        train_batch, val_batch = batch
        params, opt_state, train_loss = update_fn(self.state.params,
                                                  self.state.opt_state,
                                                  train_batch)
        val_loss = batched_loss_fn(params, val_batch)

        self.state = self.state.replace(params=params, opt_state=opt_state)
        self.train_losses.append(train_loss[0])  # only from single device
        self.val_losses.append(val_loss[0])

    def _evaluate_convergence(self, duration, thresh):
        """Prints progress, saves best obtained params and signals converged if
        validation loss improvement over the last epoch is less than the thesh.
        """
        mean_train_loss = sum(self.train_losses[-self.batches_per_epoch:]
                              ) / self.batches_per_epoch
        mean_val_loss = sum(self.val_losses[-self.batches_per_epoch:]
                            ) / self.batches_per_epoch
        print(f'Epoch {self._epoch}: Average train loss: {mean_train_loss:.5f} '
              f'Average val loss: {mean_val_loss:.5f} '
              f'Elapsed time = {duration:.3f} min')

        improvement = self.best_val_loss - mean_val_loss
        if improvement > 0.:
            self.best_val_loss = mean_val_loss
            self.best_params = copy.copy(self.params)

        if thresh is not None:
            if improvement < thresh:
                self.converged = True


class Difftre(reweighting.PropagationBase):
    """Trainer class for parametrizing potentials via the DiffTRe method."""
    def __init__(self, init_params, optimizer, reweight_ratio=0.9,
                 sim_batch_size=1, energy_fn_template=None,
                 convergence_criterion='max_loss',
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):
        """Initializes a DiffTRe trainer instance.

        The implementation assumes a NVT ensemble in weight computation.
        The trainer initialization only sets the initial trainer state
        as well as checkpointing and save-functionality. For training,
        target state points with respective simulations need to be added
        via 'add_statepoint'.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            convergence_criterion: Either 'max_loss' or 'ave_loss'.
                                   If 'max_loss', stops if the maximum loss
                                   across all batches in the epoch is smaller
                                   than convergence_thresh. 'ave_loss' evaluates
                                   the average loss across the batch. For a
                                   single state point, both are equivalent.
                                   A criterion based on the rolling standatd
                                   deviation 'std' might be implemented in the
                                   future.
            checkpoint_folder: Name of folders to store ckeckpoints in.
            checkpoint_format: Checkpoint format, currently only .pkl supported.
        """

        self.batch_losses, self.epoch_losses = [], []
        self.predictions = {}
        # TODO doc: beware that for too short trajectory might have overfittet
        #  to single trajectory; if in doubt, set reweighting ratio = 1 towards
        #  end of optimization
        self.best_params = None
        self.best_loss = 1.e16
        self.loss_window = []  # for convergence criteria based on windows
        self.criterion = convergence_criterion
        checkpoint_path = 'output/difftre/' + str(checkpoint_folder)
        init_state = util.TrainerState(params=init_params,
                                       opt_state=optimizer.init(init_params))
        super().__init__(init_state, optimizer, checkpoint_path, reweight_ratio,
                         sim_batch_size, checkpoint_format, energy_fn_template)

    def add_statepoint(self, energy_fn_template, simulator_template,
                       neighbor_fn, timings, kbt, quantities,
                       reference_state, targets=None, ref_press=None,
                       loss_fn=None, initialize_traj=True):
        """
        Adds a state point to the pool of simulations with respective targets.

        Requires own energy_fn_template and simulator_template to allow
        maximum flexibility for state points: Allows different ensembles
        (NVT vs NpT), box sizes and target quantities per state point.
        The quantity dict defines the way target observations
        contribute to the loss function. Each target observable needs to be
        saved in the quantity dict via a unique key. Model predictions will
        be output under the same key. In case the default loss function should
        be employed, for each observable the 'target' dict containing
        a multiplier controlling the weight of the observable
        in the loss function under 'gamma' as well as the prediction target
        under 'target' needs to be provided.

        In many applications, the default loss function will be sufficient.
        If a target observable cannot be described directly as an average
        over instantaneous quantities (e.g. stiffness),
        a custom loss_fn needs to be defined. The signature of the loss_fn
        needs to be the following: It takes the trajectory of computed
        instantaneous quantities saved in a dict under its respective key of
        the quantities_dict. Additionally, it receives corresponding weights
        w_i to perform ensemble averages under the reweighting scheme. With
        these components, ensemble averages of more complex observables can
        be computed. The output of the function is (loss value, predicted
        ensemble averages). The latter is only necessary for post-processing
        the optimization process. See 'init_independent_mse_loss_fn' for
        an example implementation.

        Args:
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            quantities: Dict containing for each observables specified by the
                        key a corresponding function to compute it for each
                        snapshot using traj_util.quantity_traj.
            reference_state: Tuple of initial simulation state and neighbor list
            targets: Dict containing the same keys as quantities and containing
                     another dict providing 'gamma' and 'target' for each
                     observable. Targets are only necessary when using the
                     'independent_loss_fn'.
            loss_fn: Custom loss function taking the trajectory of quantities
                     and weights and returning the loss and predictions;
                     Default None initializes an independent MSE loss, which
                     computes reweighting averages from snapshot-based
                     observables.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        # init simulation, reweighting functions and initial trajectory
        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           ref_press,
                                                           initialize_traj)

        # build loss function for current state point
        if loss_fn is None:
            loss_fn = reweighting.independent_mse_loss_fn_init(targets)
        else:
            print('Using custom loss function. Ignoring "target" dict.')

        reweighting.checkpoint_quantities(quantities)

        def difftre_loss(params, traj_state):
            """Computes the loss using the DiffTRe formalism and
            additionally returns predictions of the current model.
            """
            weights, _ = weights_fn(params, traj_state)
            quantity_trajs = traj_util.quantity_traj(traj_state,
                                                     quantities,
                                                     params)
            loss, predictions = loss_fn(quantity_trajs, weights)
            return loss, predictions

        statepoint_grad_fn = jit(value_and_grad(difftre_loss, has_aux=True))

        def difftre_grad_and_propagation(params, traj_state):
            """The main DiffTRe function that recomputes trajectories
            when needed and computes gradients of the loss wrt. energy function
            parameters for a single state point.
            """
            traj_state = propagate(params, traj_state)
            outputs, grad = statepoint_grad_fn(params, traj_state)
            loss_val, predictions = outputs
            return traj_state, grad, loss_val, predictions

        self.grad_fns[key] = difftre_grad_and_propagation
        self.predictions[key] = {}  # init saving predictions for this point

        # Reset loss measures if new state point es added since loss values
        # are not necessarily comparable
        self.best_loss = 1.e16
        self.loss_window = []

    def _update(self, batch):
        """Computes gradient averaged over the sim_batch by propagating
        respective state points. Additionally saves predictions and loss
        for postprocessing."""
        # TODO parallelization? Maybe lift batch requirement and only
        #  sync sporadically?
        # https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices
        # https://github.com/mpi4jax/mpi4jax
        # TODO split gradient and loss computation from stepping optimizer for
        #  building hybrid trainers?

        # TODO is there good way to reuse this function in BaseClass?

        # Note: in principle, we could move all the use of instance attributes
        # into difftre_grad_and_propagation, which would increase re-usability
        # with relative_entropy. However, this would probably stop all
        # parallelization efforts
        grads, losses = [], []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]
            new_traj_state, curr_grad, loss_val, state_point_predictions = \
                grad_fn(self.params, self.trajectory_states[sim_key])

            self.trajectory_states[sim_key] = new_traj_state
            self.predictions[sim_key][self._epoch] = state_point_predictions
            grads.append(curr_grad)
            losses.append(loss_val)
            if jnp.isnan(loss_val):
                warnings.warn(f'Loss of state point {sim_key} in epoch '
                              f'{self._epoch} is NaN. This was likely caused by'
                              f' divergence of the optimization or a bad model '
                              f'setup causing a NaN trajectory.')
                self.converged = True  # ends training
                break

        self.batch_losses.append(sum(losses) / self.sim_batch_size)
        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)

    def _evaluate_convergence(self, duration, thresh):
        last_losses = jnp.array(self.batch_losses[-self.sim_batch_size:])
        epoch_loss = jnp.mean(last_losses)
        self.epoch_losses.append(epoch_loss)
        print(f'Epoch {self._epoch}: Epoch loss = {epoch_loss:.5f}, '
              f'Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        curr_epoch_loss = self.epoch_losses[-1]
        improvement = self.best_loss - curr_epoch_loss
        if improvement > 0.:
            self.best_loss = curr_epoch_loss
            self.best_params = copy.copy(self.params)

        if thresh is not None:
            if self.criterion == 'max_loss':
                if max(last_losses) < thresh: self.converged = True
            elif self.criterion == 'ave_loss':
                if epoch_loss < thresh: self.converged = True
            elif self.criterion == 'std':
                raise NotImplementedError('Currently, there is no criterion '
                                          'based on the std of the loss '
                                          'implemented.')
            else:
                raise ValueError(f'Convergence criterion {self.criterion} '
                                 f'unknown. Select "max_loss", "ave_loss" or '
                                 f'"std".')


class DifftreActive:
    """Active learning of state-transferable potentials from experimental data
     via DiffTRe.

     The input trainer can be pre-trained or freshly initialized. Pre-training
     usually comes with the advantage that the initial training from random
     parameters is usually the most unstable one. Hence, special care can be
     taken such as training on NVT initially to fix the pressure and swapping
     to NPT afterwards. This active learning trainer then takes care of learning
      statepoint transferability.
     """
    def __init__(self, trainer):
        self.trainer = trainer
        # other inits

    def add_statepoint(self, *args, **kwargs):
        """Add another statepoint to the target state points.

        Predominantly used to add statepoints with more / different targets
        not covered in  the on-the-fly tepoint addition, e.g. for an extensive
        initial statepoint. Please refer to :obj:'Difftre.add_statepoint
        <chemtrain.trainers.Difftre.add_statepoint>' for the full documentation.
        """
        self.trainer.add_statepoint(*args, **kwargs)

    def train(self, max_new_statepoints=100):
        for added_statepoints in range(max_new_statepoints):
            accuracy_met = False
            if accuracy_met:
                print('Visited state space covered with accuracy target met.')
                break

            # checkpoint: call checkpoint of trainer
        else:
            warnings.warn('Maximum number of added statepoints added without '
                          'reaching target accuracy over visited state space.')



    # TODO Ckeckpointing functions here not very useful here: Override those
    #  and use checkpointing of difftre trainer
    def load_checkpoint(self, file_path):
        self.trainer = util.MLETrainerTemplate.load_trainer(file_path)
        # TODO load rest?

    # Interface convenience functions from TrainerTemplate
    @property
    def energy_fn(self):
        return self.trainer.energy_fn

    def save_trainer(self, save_path):
        raise NotImplementedError

    @classmethod
    def load_trainer(cls, filepath):
        raise NotImplementedError

    def save_energy_params(self, *args, **kwargs):
        self.trainer.save_energy_params(*args, **kwargs)

    def load_energy_params(self, *args, **kwargs):
        self.trainer.load_energy_params(*args, **kwargs)

    @property
    def params(self):
        return self.trainer.params

    @params.setter
    def params(self, loaded_params):
        self.trainer.params = loaded_params


class RelativeEntropy(reweighting.PropagationBase):
    """Trainer for relative entropy minimization."""
    def __init__(self, init_params, optimizer,
                 reweight_ratio=0.9, sim_batch_size=1, energy_fn_template=None,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):
        """
        Initializes a relative entropy trainer instance.

        Uses first order method optimizer as Hessian is very expensive
        for neural networks. Both reweighting and the gradient formula
        currently assume a NVT ensemble.

        Args:
            init_params: Initial energy parameters
            optimizer: Optimizer from optax
            reweight_ratio: Ratio of reference samples required for n_eff to
                            surpass to allow re-use of previous reference
                            trajectory state. If trajectories should not be
                            re-used, a value > 1 can be specified.
            sim_batch_size: Number of state-points to be processed as a single
                            batch. Gradients will be averaged over the batch
                            before stepping the optimizer.
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function. Here, the
                                energy_fn_template is only a reference that
                                will be saved alongside the trainer. Each
                                state point requires its own due to the
                                dependence on the box size via the displacement
                                function, which can vary between state points.
            checkpoint_folder: Name of folders to store ckeckpoints in.
            checkpoint_format: Checkpoint format, currently only .pkl supported.
        """

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        init_trainer_state = util.TrainerState(
            params=init_params, opt_state=optimizer.init(init_params))
        super().__init__(init_trainer_state, optimizer, checkpoint_path,
                         reweight_ratio, sim_batch_size, checkpoint_format,
                         energy_fn_template)

        # in addition to the standard trajectory state, we also need to keep
        # track of dataloader states for reference snapshots
        self.data_states = {}

    def _set_dataset(self, key, reference_data, reference_batch_size,
                     batch_cache=1):
        """Set dataset and loader corresponding to current state point."""
        reference_loader = data.NumpyDataLoader(R=reference_data)
        cache_size = batch_cache * reference_batch_size
        init_reference_batch, get_reference_batch = data.random_reference_data(
            reference_loader, cache_size, reference_batch_size)
        init_reference_batch_state = init_reference_batch()
        self.data_states[key] = init_reference_batch_state
        return get_reference_batch

    def add_statepoint(self, reference_data, energy_fn_template,
                       simulator_template, neighbor_fn, timings, kbt,
                       reference_state, reference_batch_size=None,
                       batch_cache=1, initialize_traj=True):
        """
        Adds a state point to the pool of simulations.

        As each reference dataset / trajectory corresponds to a single
        state point, we initialize the dataloader together with the
        simulation.

        Args:
            reference_data: De-correlated reference trajectory
            energy_fn_template: Function that takes energy parameters and
                                initializes an new energy function.
            simulator_template: Function that takes an energy function and
                                returns a simulator function.
            neighbor_fn: Neighbor function
            timings: Instance of TimingClass containing information
                     about the trajectory length and which states to retain
            kbt: Temperature in kbT
            reference_state: Tuple of initial simulation state and neighbor list
            reference_batch_size: Batch size of dataloader for reference
                                  trajectory. If None, will use the same number
                                  of snapshots as generated via the optimizer.
            batch_cache: Number of reference batches to cache in order to
                         minimize host-device communication. Make sure the
                         cached data size does not exceed the full dataset size.
            initialize_traj: True, if an initial trajectory should be generated.
                             Should only be set to False if a checkpoint is
                             loaded before starting any training.
        """

        if reference_batch_size is None:
            # use same amount of snapshots as generated in trajectory by default
            reference_batch_size = jnp.size(timings.t_production_start)

        key, weights_fn, propagate = self._init_statepoint(reference_state,
                                                           energy_fn_template,
                                                           simulator_template,
                                                           neighbor_fn,
                                                           timings,
                                                           kbt,
                                                           initialize_traj)

        reference_dataloader = self._set_dataset(key,
                                                 reference_data,
                                                 reference_batch_size,
                                                 batch_cache)

        grad_fn = reweighting.init_rel_entropy_gradient(
            energy_fn_template, weights_fn, kbt)

        def propagation_and_grad(params, traj_state, batch_state):
            """Propagates the trajectory, if necessary, and computes the
            gradient via the relative entropy formalism.
            """
            traj_state = propagate(params, traj_state)
            new_batch_state, reference_batch = reference_dataloader(batch_state)
            reference_positions = reference_batch['R']
            grad = grad_fn(params, traj_state, reference_positions)
            return traj_state, grad, new_batch_state

        self.grad_fns[key] = propagation_and_grad

    def _update(self, batch):
        """Updates the potential using the gradient from relative entropy."""
        grads = []
        for sim_key in batch:
            grad_fn = self.grad_fns[sim_key]

            self.trajectory_states[sim_key], curr_grad, \
            self.data_states[sim_key] = grad_fn(self.params,
                                                self.trajectory_states[sim_key],
                                                self.data_states[sim_key])
            grads.append(curr_grad)

        batch_grad = util.tree_mean(grads)
        self._step_optimizer(batch_grad)

    def _evaluate_convergence(self, duration, thresh):
        print(f'Epoch {self._epoch}: Elapsed time = {duration:.3f} min')

        self._print_measured_statepoint()

        self.converged = False  # TODO implement convergence test
        if thresh is not None:
            raise NotImplementedError('Currently there is no convergence '
                                      'criterion implemented for relative '
                                      'entropy minimization. A possible '
                                      'implementation might be based on the '
                                      'variation of params or reweigting '
                                      'effective sample size.')
