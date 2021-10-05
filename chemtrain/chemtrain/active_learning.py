from difftre import run_to_next_printout_neighbors


def init_uq_md(energy_fn_template, simulator_template, neighbor_fn):
    def uq_md():
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_to_printout = run_to_next_printout_neighbors(apply_fn,
                                                         neighbor_fn,
                                                         timesteps_per_printout)

        final_state, nbrs = new_sim_state

        return TrajectoryState(new_sim_state, traj, energies)