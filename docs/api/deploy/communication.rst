:modulename:`deploy.comm`
==========================

.. automodule:: chemtrain.deploy.comm

.. currentmodule:: chemtrain.deploy.comm

The exporter creates the communication object and passes it to
:meth:`~chemtrain.deploy.exporter.Exporter.energy_fn`. Model implementations
use the supplied object at fixed locations in the traced computation. They do
not need to construct it themselves.

.. autoclass:: ExportCommunication
   :members: gather, reduce

Non-Communicating Fallback
--------------------------

The module-level function validates and converts a pytree without exchanging
data between ranks. It is useful when model setup or reference calculations
run outside an exporter-managed communication trace.

.. autofunction:: gather
