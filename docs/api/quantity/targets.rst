chemtrain.quantity.targets
===========================

This module creates DiffTRe targets and builds the functions computing
predictions based on a simulated trajectory.


Intramolecular
--------------

Bonds
_____


.. autofunction:: chemtrain.quantity.structure.init_dihedral_distribution_target


Intermolecular
--------------

.. autofunction:: chemtrain.quantity.structure.init_radial_distribution_target

.. autofunction:: chemtrain.quantity.structure.init_angular_distribution_target


Thermodynamic
--------------

.. autofunction:: chemtrain.quantity.thermodynamics.init_relative_entropy_target


Build Targets and Compute Functions
------------------------------------

.. autoclass:: chemtrain.quantity.TargetBuilder


