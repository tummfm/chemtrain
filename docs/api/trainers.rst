``chemtrain.trainers``
=======================

.. automodule::chemtrain.trainers

.. currentmodule:: chemtrain.trainers

Base Trainers
--------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   Difftre
   PropertyPrediction
   ForceMatching
   DifftreActive
   RelativeEntropy
   SGMCForceMatching



Combining Trainers
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   InterleaveTrainers


Train Multiple Models
----------------------

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   EnsembleOfModels

Trainer Templates
------------------

.. currentmodule:: chemtrain.trainers.base

.. autosummary::
   :toctree: _autosummary
   :template: class.rst

   MLETrainerTemplate
   DataParallelTrainer
   EarlyStopping
