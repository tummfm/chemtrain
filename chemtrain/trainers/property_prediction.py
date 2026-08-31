from chemtrain.learn import property_prediction
from chemtrain.trainers import base as tt


from jax import numpy as jnp, tree_util


import functools


class PropertyPrediction(tt.DataParallelTrainer):
    """Trainer for direct prediction of molecular properties."""
    def __init__(self, error_fn, prediction_model, init_params, optimizer,
                 graph_dataset, targets, batch=1, batch_per_device=None, batch_cache=10,
                 train_ratio=0.7, val_ratio=0.1, test_error_fn=None,
                 shuffle=False, convergence_criterion="window_median",
                 checkpoint_folder="Checkpoints"):
        # TODO documentation

        # TODO build graph on-the-fly as memory moving might be bottleneck here
        model = property_prediction.init_model(prediction_model)
        checkpoint_path = "output/property_prediction/" + str(checkpoint_folder)
        loss_fn = property_prediction.init_loss_fn(error_fn)

        super().__init__(
            loss_fn, model, init_params, optimizer, checkpoint_path,
            batch=batch, batch_cache=batch_cache, batch_per_device=batch_per_device,
            convergence_criterion=convergence_criterion
        )

        dataset_dict, _ = property_prediction.build_dataset(targets, graph_dataset)
        self.set_datasets(
            dataset_dict, train_ratio=train_ratio, val_ratio=val_ratio,
            shuffle=shuffle
        )

        self.test_error_fn = test_error_fn

    def predict(self, single_observation):
        """Prediction for a single input graph using the current param state."""
        batched_observation = tree_util.tree_map(
            functools.partial(jnp.expand_dims, axis=0), single_observation
        )
        batched_prediction = self.batched_model(
            self.best_inference_params, batched_observation)
        single_prediction = tree_util.tree_map(
            functools.partial(jnp.squeeze, axis=0), batched_prediction
        )
        return single_prediction

    def evaluate_testset_error(self, best_params=True):
        assert "testing" in self._batch_states.keys(), (
            "No test set available. Check train and val ratios."
        )
        assert self.test_error_fn is not None, (
            "`test_error_fn` is necessary during initialization."
        )

        params = (self.best_inference_params_replicated
                  if best_params else self.state.params)

        error = self.evaluate(
            "testing", self.test_error_fn, params=params
        )

        print(f"Error on test set: {error}")
        return error
