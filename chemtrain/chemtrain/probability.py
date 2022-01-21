from jax import numpy as jnp
from chemtrain.force_matching import init_single_prediction


def build_likelihood(nbrs_init, energy_fn_template):
    """
    Builds the likelihood for stochastic gradient optimization. Force_fn used to calculate the 
    forces given a model is imported from force_matching.py
    """
    single_prediction = init_single_prediction(nbrs_init, energy_fn_template, virial_fn=None)

    from jax import tree_util
    import jax.scipy as jscp
    def likelihood(sample, observation):
        sample, observation = tree_util.tree_map(jnp.float32, (sample, observation))  # TODO: <-- poss. remove
        positions = {'R': observation['positions']}  # --> Key conflict between alias and single prediction
        prediction = single_prediction(params=sample['potential'], observation=positions)
        predicted_forces = prediction['F']

        # Simple likelihood calculation
        likelihoods = jscp.stats.norm.logpdf(predicted_forces,
                                            loc=observation['forces'],
                                            scale=sample['std'])
        return jnp.sum(likelihoods)
    return likelihood