JCN Adapter API
===============

The JCN API is the versioned C boundary between ``libconnector.so`` and a
simulation-engine adapter. Applications using the chemtrain LAMMPS package do
not call this API directly.

An adapter creates one PJRT client, loads one or more model bundles, and creates
an executor for each model. It imports application-owned buffers through
DLPack, supplies neighbor-list buffers in a supported layout, and submits force
requests.
Opaque handles and the function table keep C++ implementation details out of
the public ABI.

ABI discovery
-------------

``JCN_API_VERSION`` identifies the ABI expected by the adapter.
``jcn_get_api`` returns the corresponding function table, or ``NULL`` when the
requested version is unavailable. The adapter should compare both the version
and ``struct_size`` before using the table.

.. doxygendefine:: JCN_API_VERSION

.. c:function:: const JCN_Api *jcn_get_api(uint32_t requested_version)

   Return the function table for ``requested_version``, or ``NULL`` when the
   connector does not support the requested API version.

.. doxygenstruct:: JCN_Api
   :members:

Runtime and model setup
-----------------------

The client selects a PJRT backend, device, and memory fraction. Model properties
describe the selected bundle before an adapter allocates atom, neighbor, output,
or communication storage. Set the engine ABI and, for communication-enabled
execution, the callback table before selecting the model variant. The client,
model, and callback context must outlive every executor that uses them.

Neighbor lists borrow their imported buffer handles. After outstanding calls
finish, destroy neighbor lists before their buffers. Then destroy the executor,
model, and client in that order by using the corresponding functions in
:c:struct:`JCN_Api`.

.. doxygenstruct:: jcn_client_options
   :members:

.. doxygenstruct:: jcn_runtime_info
   :members:

.. doxygenstruct:: jcn_model_options
   :members:

.. doxygenstruct:: jcn_engine_abi_options
   :members:

.. doxygenstruct:: jcn_named_tensor_dtype
   :members:

.. doxygenstruct:: jcn_model_properties
   :members:

Buffers and force requests
--------------------------

Buffers are imported from DLPack with an explicit role and copy policy.
Application storage remains application-owned. A successful import transfers
ownership of the ``DLManagedTensor`` wrapper to the returned JCN buffer.

Each force request supplies named particle and global inputs, a neighbor-list
handle, requested output buffers, and concrete rank-local capacities. A
``JCN_COMPUTE_NEEDS_CAPACITY_CHANGE`` result asks the adapter to coordinate a
capacity update across all participating ranks before retrying.

.. doxygenstruct:: jcn_buffer_import_options
   :members:

.. doxygenstruct:: jcn_atoms
   :members:

.. doxygenstruct:: jcn_named_input
   :members:

.. doxygenstruct:: jcn_named_output
   :members:

.. doxygenstruct:: jcn_requested_capacities
   :members:

.. doxygenstruct:: jcn_force_request
   :members:

.. doxygenstruct:: jcn_force_result
   :members:

Neighbor lists
--------------

Adapters convert their native neighbor representation to one of the buffer
layouts below and create an opaque ``jcn_neighbor_list``. Every index,
topology, and capacity buffer must match the shape and dtype declared by the
model and engine ABI.

.. doxygenstruct:: jcn_sparse_neighbors
   :members:

.. doxygenstruct:: jcn_dense_neighbors
   :members:

Communication callbacks
-----------------------

Communication-aware models call adapter-provided exchange and reduction
callbacks from PJRT execution. Host callbacks receive host buffers. Device
callbacks receive borrowed device scratch and a backend-native stream. A
callback must either finish its work before returning or order the work on the
supplied stream. The callback must not retain the scratch pointer.

.. doxygenstruct:: jcn_communication_callbacks
   :members:

Status and descriptors
----------------------

Status messages remain valid until the next JCN call on the same thread.
Descriptor names are case-sensitive and borrowed from the loaded model.

.. doxygenstruct:: jcn_status
   :members:

.. doxygenstruct:: jcn_particle_field_descriptor
   :members:

.. doxygenstruct:: jcn_global_field_descriptor
   :members:

.. doxygenstruct:: jcn_output_descriptor
   :members:

Enumerations
------------

.. doxygenenum:: jcn_status_code
.. doxygenenum:: jcn_compute_code
.. doxygenenum:: jcn_neighbor_format
.. doxygenenum:: jcn_tensor_dtype
.. doxygenenum:: jcn_communication_scalar_type
.. doxygenenum:: jcn_dlpack_copy_policy
.. doxygenenum:: jcn_buffer_role
.. doxygenenum:: jcn_species_encoding
.. doxygenenum:: jcn_dense_layout
.. doxygenenum:: jcn_output_scope
