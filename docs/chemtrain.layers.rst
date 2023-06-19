Layers
========================================

.. automodule:: chemtrain.layers


Initializers
________________

.. autoclass:: OrthogonalVarianceScalingInit
   :members:

   .. automethod:: __init__
.. autoclass:: RBFFrequencyInitializer

DimeNet++ Layers
__________________

Basis Layers
.....................

.. autoclass:: SmoothingEnvelope
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: RadialBesselLayer
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SphericalBesselLayer
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. _dimenet_building_blocks:

DimeNet++ Building Blocks
............................

.. autoclass:: ResidualLayer
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: EmbeddingBlock
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: OutputBlock
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: InteractionBlock
   :members:

   .. automethod:: __init__
   .. automethod:: __call__


Utility Functions
__________________

.. autosummary::
   :toctree: _autosummary

    high_precision_segment_sum


