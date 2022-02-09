"""Functions to extract the sparse (angular) graph representation employed in
DimeNet.
"""
from jax import numpy as jnp, vmap, lax


def angle(r_ij, r_kj):
    """Computes the angle (kj, ij) from vectors r_kj and r_ij,
    correctly selecting the quadrant.

    Based on tan(theta) = |(r_ji x r_kj)| / (r_ji . r_kj).
    Beware the non-differentability of arctan2(0,0).

    Args:
        r_ij: Vector pointing to position of particle i from particle j
        r_kj: Vector pointing to position of particle k from particle j

    Returns:
        Angle between vectors
    """
    cross = jnp.linalg.norm(jnp.cross(r_ij, r_kj))
    dot = jnp.dot(r_ij, r_kj)
    theta = jnp.arctan2(cross, dot)
    return theta


def safe_angle_mask(r_ji, r_kj, angle_mask):
    """Sets masked angles to pI/2 to ensure differentiablility.

    Args:
        r_ji: Vector pointing to position of particle i from particle j
        r_kj: Vector pointing to position of particle k from particle j
        angle_mask: Boolean mask for each triplet, which is False for triplets
                    that need to be masked.

    Returns:
        A tuple (r_ji_safe, r_kj_safe) of vectors r_ji and r_kj, where masked
        triplets are replaced such that the angle between them is pi/2.
    """
    safe_ji = jnp.array([1., 0., 0.], dtype=jnp.float32)
    safe_kj = jnp.array([0., 1., 0.], dtype=jnp.float32)
    r_ji_safe = jnp.where(angle_mask, r_ji, safe_ji)
    r_kj_safe = jnp.where(angle_mask, r_kj, safe_kj)
    return r_ji_safe, r_kj_safe


def angle_triplets(positions, displacement_fn, angle_idxs, angle_mask):
    """Computes the angle for all triplets between 0 and pi. Masked angles are
     set to pi/2.

    Args:
        positions: Array pf particle positions (N_particles x 3)
        displacement_fn: Jax_md displacement function
        angle_idxs: Array of particle indeces that form a triplet
                    (N_triples x 3)
        angle_mask: Boolean mask for each triplet, which is False for triplets
                    that need to be masked.

    Returns:
        A (N_triples,) array with the angle for each triplet.
    """
    r_i = positions[angle_idxs[:, 0]]
    r_j = positions[angle_idxs[:, 1]]
    r_k = positions[angle_idxs[:, 2]]

    # Note: The original DimeNet implementation uses R_ji, however r_ij is the
    #       correct vector to get the angle between both vectors. This is a
    #       known issue in DimeNet. We apply the correct angle definition.
    r_ij = vmap(displacement_fn)(r_i, r_j)  # r_i - r_j respecting periodic BCs
    r_kj = vmap(displacement_fn)(r_k, r_j)
    # we need to mask as the case where r_ij is co-linear with r_kj.
    # Otherwise, this generates NaNs on the backward pass
    r_ij_safe, r_kj_safe = safe_angle_mask(r_ij, r_kj, angle_mask)
    angles = vmap(angle)(r_ij_safe, r_kj_safe)
    return angles


def _flatten_sort_and_capp(matrix, sorting_args, cap_size):
    """Helper function that takes a 2D array, flattens it, sorts it using the
    args (usually provided via argsort) and capps the end of the resulting
    vector. Used to delete non-existing edges and returns the capped vector.
    """
    vect = jnp.ravel(matrix)
    sorted_vect = vect[sorting_args]
    capped_vect = sorted_vect[0:cap_size]
    return capped_vect


def sparse_graph(pair_distances, edge_idx_ji, max_edges=None, max_angles=None):
    """Constructs a sparse representation of graph edges and angles to save
     memory and computations over neighbor list.

    The speed-up over simply using the dense jax_md neighbor list is
    significant, particularly regarding triplets. To allow for a representation
    of constant size required by jit, we pad the resulting vectors.

    Args:
        pair_distances: Pairwise distance matrix of particles
                        (N_partixles x max_neighbors) array
        edge_idx_ji: Padded jax_md dense neighborlist connectivity:
                     (N_partixles x max_neighbors) array
        max_edges: Maximum number of edges storable in the graph. Can be used to
                   reduce the number of padded edges, but should be used
                   carefully, such that no existing edges are capped. Default
                   None uses the maximum possible number of edges as given by
                   the dense neighbor list.
        max_angles: Maximum number of triplets storable in the graph. Can be
                    used to reduce the number of padded triplets, but should be
                    used carefully, such that no existing triplets are capped.
                    Default None uses the maximum possible number of triplets as
                    given by the dense neighbor list.

    Returns:
        Tuple of arrays defining sparse graph connectivity:
        d_ij, pair_indicies, angle_idxs, angle_connectivity, (n_edges, n_angles)
    """

    # TODO might be worth updating this function to the new sparse-style
    #  neighborlist in jax_md
    # conservative estimates for initialization run
    # use guess from initialization for tighter bound to save memory and
    # computations during production runs
    n_particles, max_neighbors = edge_idx_ji.shape
    if max_edges is None:
        max_edges = n_particles * max_neighbors
    if max_angles is None:
        max_angles = max_edges * max_neighbors

    # sparse edge representation:
    # construct vectors from adjacency matrix and only keep existing edges
    # Target node (i) and source (j) of edges
    pair_mask = edge_idx_ji != n_particles  # non-existing neighbor encoded as N
    # due to undirectedness, each edge is included twice
    n_edges = jnp.count_nonzero(pair_mask)
    pair_mask_flat = jnp.ravel(pair_mask)
    # non-existing edges are sorted to the end for capping
    sorting_idxs = jnp.argsort(~pair_mask_flat)
    _, yy = jnp.meshgrid(jnp.arange(max_neighbors), jnp.arange(n_particles))  # pylint: disable=unbalanced-tuple-unpacking
    idx_i = _flatten_sort_and_capp(yy, sorting_idxs, max_edges)
    idx_j = _flatten_sort_and_capp(edge_idx_ji, sorting_idxs, max_edges)
    d_ij = _flatten_sort_and_capp(pair_distances, sorting_idxs, max_edges)
    sparse_pair_mask = _flatten_sort_and_capp(pair_mask_flat, sorting_idxs,
                                              max_edges)
    pair_indicies = (idx_i, idx_j, jnp.expand_dims(sparse_pair_mask, -1))

    # build sparse angle combinations from adjacency matrix:
    # angle defined for 3 particles with connections k->j and j->i
    # directional message passing accumulates all k->j to update each m_ji
    idx3_i = jnp.repeat(idx_i, max_neighbors)
    idx3_j = jnp.repeat(idx_j, max_neighbors)
    # retrieves for each j in idx_j its neighbors k: stored in 2nd axis
    idx3_k_mat = edge_idx_ji[idx_j]
    idx3_k = idx3_k_mat.ravel()
    angle_idxs = jnp.column_stack([idx3_i, idx3_j, idx3_k])

    # masking:
    # k and j are different particles, by edge_idx_ji construction.
    # The same applies to j - i, except for masked ones
    mask_i_eq_k = idx3_i != idx3_k
    # mask for ij known a priori
    mask_ij = jnp.repeat(sparse_pair_mask, max_neighbors)
    mask_k = idx3_k != n_particles
    angle_mask = mask_ij * mask_k * mask_i_eq_k  # union of masks
    angle_mask, sorting_idx3 = lax.top_k(angle_mask, max_angles)
    angle_idxs = angle_idxs[sorting_idx3]
    n_angles = jnp.count_nonzero(angle_mask)

    # retrieving edge_id m_ji from nodes i and j:
    # idx_i < N by construction, but idx_j can be N: will override
    # lookup[i, N-1], which is problematic if [i, N-1] is an existing edge
    edge_id_lookup = jnp.zeros([n_particles, n_particles + 1], dtype=jnp.int32)
    edge_id_lookup_direct = edge_id_lookup.at[(idx_i, idx_j)].set(
        jnp.arange(max_edges))

    # stores for each angle kji edge index j->i to aggregate messages via a
    # segment_sum: each m_ji is a distinct segment containing all incoming m_kj
    reduce_to_ji = edge_id_lookup_direct[(angle_idxs[:, 0], angle_idxs[:, 1])]
    # stores for each angle kji edge index k->j to gather all incoming edges
    # for message passing
    expand_to_kj = edge_id_lookup_direct[(angle_idxs[:, 1], angle_idxs[:, 2])]
    angle_connectivity = (jnp.expand_dims(angle_mask, -1), reduce_to_ji,
                          expand_to_kj)

    return d_ij, pair_indicies, angle_idxs, angle_connectivity, (n_edges,
                                                                 n_angles)
