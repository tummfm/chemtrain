from chex import dataclass
from functools import partial
from typing import Any

from jax import jit, tree_map, tree_multimap, grad, lax, numpy as jnp
from jax_sgmc.data import NumpyDataLoader, random_reference_data
import time
import warnings

from chemtrain.util import TrainerTemplate, TrainerState, step_optimizer
import chemtrain.difftre as difftre

# The relative entropy implementation builds on the functionalities
# implemented in DIffTRe. We therefore enforce a consistent interface
# with the DifftreState. Here, we additionally need a batch_state that
# keeps track of the reference data.


@partial(dataclass, frozen=True)
class EntropyState(TrainerState):
    """Extend trainer state with batch state."""
    batch_state: Any


def init_rel_entropy_gradient(energy_fn_template, neighbor_fn, init_state,
                              compute_weights, kbT):
    # TODO can be implemented on trainer level
    beta = 1 / kbT
    _, nbrs_init = init_state

    @jit
    def rel_entropy_gradient(params, traj_state, AA_traj):

        def energy(params, R):
            energy_fn = energy_fn_template(params)
            nbrs = neighbor_fn(R, nbrs_init)
            return energy_fn(R, neighbor=nbrs)

        def weighted_gradient(grad_carry, input):
            dUdtheta = grad(energy)  # gradient wrt. params
            R = input[0]
            weight = input[1]
            snapshot_grad = dUdtheta(params, R)
            # sum over weights represents average
            update_carry = lambda carry, new_grad: carry + weight * new_grad
            updated_carry = tree_multimap(update_carry,
                                          grad_carry,
                                          snapshot_grad)
            return updated_carry, 0

        CG_traj = traj_state.trajectory.position
        CG_weights, _ = compute_weights(params, traj_state)
        n_AA_snapshots = AA_traj.shape[0]
        AA_weights = jnp.ones(n_AA_snapshots) / n_AA_snapshots  # no reweighting

        # Note:
        # would be more efficient to partially batch here, however
        # sequential trajectory generation dominates and batching
        # here would therefore not result in a significant speed-up.
        # The average via scan avoids linear memory scaling with
        # the trajectory length, which would become prohibitive for
        # memory consuming models such as neural network potentials.
        initial_grad = tree_map(jnp.zeros_like, params)
        mean_AA_grad, _ = lax.scan(weighted_gradient,
                                   initial_grad,
                                   (AA_traj, AA_weights))
        mean_CG_grad, _ = lax.scan(weighted_gradient,
                                   initial_grad,
                                   (CG_traj, CG_weights))

        combine_grads = lambda x, y: beta * (x - y)
        dtheta = tree_multimap(combine_grads, mean_AA_grad, mean_CG_grad)
        return dtheta
    return rel_entropy_gradient


def init_update_fn(simulator_template, energy_fn_template, neighbor_fn,
                   reference_state, init_params, kbT, optimizer, timings_struct,
                   get_AA_batch, reweight_ratio=0.9):
    initial_traj_generator, compute_weights, propagate = \
        difftre.init_reweighting_propagation_fns(energy_fn_template,
                                                 simulator_template,
                                                 neighbor_fn,
                                                 timings_struct,
                                                 kbT,
                                                 reweight_ratio)

    t_start = time.time()
    init_trajectory = initial_traj_generator(init_params, reference_state)
    runtime = (time.time() - t_start) / 60.
    print('Time for a single trajectory generation:', runtime, 'mins')

    grad_fn = init_rel_entropy_gradient(energy_fn_template, neighbor_fn,
                                        reference_state, compute_weights, kbT)

    def propagation_and_grad(entropy_state):
        """Propagates the trajectory, if necessary, and computes the
        gradient via the relative entropy formalism.
        """
        traj_state = propagate(entropy_state.params, entropy_state.traj_state)
        new_batch_state, AA_batch = get_AA_batch(entropy_state.batch_state)
        AA_batch = AA_batch['R']
        grad = grad_fn(entropy_state.params, traj_state, AA_batch)
        return traj_state, new_batch_state, grad

    def update(entropy_state):
        """This function is not jitable as the update checks the
        neighborlist buffer and increases is when needed.
        """
        traj_state, new_batch_state, curr_grad = propagation_and_grad(entropy_state)
        params, opt_state = step_optimizer(entropy_state.params,
                                           entropy_state.opt_state,
                                           curr_grad,
                                           optimizer)

        new_state = entropy_state.replace(params=params,
                                          traj_state=traj_state,
                                          opt_state=opt_state,
                                          batch_state=new_batch_state)
        return new_state
    return update, init_trajectory


class Trainer(TrainerTemplate):
    """Uses first order method as Hessian is very expensive for neural
    networks. Both reweighting and the gradient formula assume a NVT
    ensemble.
    """
    # TODO can we inherit from difftre and override all unnecessray?
    # TODO is there a stopping criterion available?
    #  --> maybe based on exponential average of N_eff?
    def __init__(self, init_params, AA_traj, simulator_template,
                 energy_fn_template, neighbor_fn, reference_state, timings,
                 optimizer, kbT, reweight_ratio=0.9, n_AA=None, batch_cache=10,
                 checkpoint_folder='Checkpoints', checkpoint_format='pkl'):

        checkpoint_path = 'output/rel_entropy/' + str(checkpoint_folder)
        super().__init__(checkpoint_path, checkpoint_format, energy_fn_template)

        # use same amount of printouts as generated in trajectory by default
        if n_AA is None:
            n_AA = jnp.size(timings.t_production_start)
        AA_loader = NumpyDataLoader(n_AA, R=AA_traj)
        init_AA_batch, get_AA_batch = random_reference_data(
            AA_loader, batch_cache * n_AA)
        init_AA_batch_state = init_AA_batch()

        opt_state = optimizer.init(init_params)
        self.update, init_traj = init_update_fn(simulator_template,
                                                energy_fn_template,
                                                neighbor_fn,
                                                reference_state,
                                                init_params,
                                                kbT,
                                                optimizer,
                                                timings,
                                                get_AA_batch,
                                                reweight_ratio)

        self.__state = EntropyState(params=init_params,
                                    traj_state=init_traj,
                                    opt_state=opt_state,
                                    batch_state=init_AA_batch_state)

    @property
    def state(self):
        return self.__state

    @property
    def params(self):
        return self.__state.params

    @state.setter
    def state(self, loaded_state):
        self.__state = loaded_state

    def train(self, epochs, checkpoint_freq=None):
        start_epoch = self.epoch
        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):
            start_time = time.time()
            self.__state = self.update(self.__state)
            end_time = (time.time() - start_time) / 60.
            print('Time for update ' + str(epoch) + ':', str(end_time), 'min')

            if jnp.any(jnp.isnan(self.__state.traj_state.energies)):
                warnings.warn('Parameters are NaN. This was likely caused by '
                              'divergence of the optimization or a bad '
                              'model setup causing a NaN trajectory.')
                break

            self.epoch += 1
            self.dump_checkpoint_occasionally(frequency=checkpoint_freq)
        return
