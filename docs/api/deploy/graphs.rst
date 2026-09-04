:modulename:`deploy.graphs`
============================

.. automodule:: chemtrain.deploy.graphs

.. currentmodule:: chemtrain.deploy.graphs

Graphs and Neighbor Lists
-------------------------

.. autoclass:: NeighborList
   :members:

.. autoclass:: SimpleSparseNeighborList
   :members:

.. autoclass:: SimpleDenseNeighborList
   :members:

Neighbor-List Statistics
------------------------

Neighbor-list statistics describe capacities used during graph construction,
pruning, and executable compilation.

.. autoclass:: NeighborListStatistics
   :members:

Utility Functions
-----------------

.. autofunction:: prune_neighbor_list
.. autofunction:: prune_neighbor_list_dense
