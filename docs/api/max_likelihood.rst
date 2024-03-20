Maximum Likelihood
===================

.. currentmodule:: chemtrain.max_likelihood

Loss Functions
---------------

.. autofunction:: mse_loss

.. autofunction:: mae_loss


Dataset Predictions
--------------------

.. autofunction:: pmap_update_fn

.. autofunction:: init_val_predictions

.. autofunction:: init_val_loss_fn


Trainer Base Classes
---------------------

.. autoclass:: MLETrainerTemplate
   :members:

.. autoclass:: EarlyStopping
  :members:

.. autoclass:: DataParallelTrainer
   :members:
