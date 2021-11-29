
def _small_uncertainty(state, uncertainty_fn, uncertainty_threshold):
    """A function returning True if the uncertainty/error of the current
     state is below a threshold, otherwise false.
     """
    uncertainty_estimate = uncertainty_fn(state)
    small = uncertainty_estimate < uncertainty_threshold
    return small


def init_uq_md(get_energy_fn, simulator_template, neighbor_fn,
               timesteps_per_printout, uncertainty_fn, uncertainty_threshold,
               temperature_schedule=None):

    def uq_md(state):
        # TODO add time varying temperature schedule
        energy_fn = get_energy_fn(state.params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = run_to_next_printout_neighbors(apply_fn,
                                                         neighbor_fn,
                                                         timesteps_per_printout)
        # TODO change to lax.while_loop
        while uncertainty_small(state):
            new_sim_state = run_to_printout(state.sim_state)
            state.sim_state = new_sim_state  # TODO check this is compatible with jax/jit

        return state

    return uq_md

