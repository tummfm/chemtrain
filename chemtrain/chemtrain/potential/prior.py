import functools
import itertools
from io import StringIO

import numpy as np
from jax import tree_util, debug
from jax.typing import ArrayLike
import jax.numpy as jnp

from jax_md import energy

from chemtrain.jax_md_mod import custom_energy, io
from chemtrain import sparse_graph

import numpy as onp

import tomli

import importlib


@tree_util.register_pytree_node_class
class ForceField:

    def __init__(self, data=None, lookup=None, mapping=None):
        self._data = data
        self._mapping = mapping
        self._lookup = lookup

    @classmethod
    def load_ff(cls, fname):
        """Loads a force field from a toml file.

        Args:
            fname: Force field parameter file

        Returns:
            Returns a force field instance with parameters from the file.

        """
        data = {
            "nonbonded": None,
            "bonded": {}
        }

        lookup = {
            "nonbonded": None,
            "bonded": {}
        }

        with open(fname, "rb") as f:
            ff = tomli.load(f)

        # Parse the contents by numpy
        non_bonded = onp.genfromtxt(
            StringIO(ff["nonbonded"]["atomtypes"]),
            dtype=None, delimiter=","
        )
        mapping = {
            name.decode('UTF-8').strip(): index for name, index, *_ in
            non_bonded
        }

        num_species = max(mapping.values()) + 1

        # Create lookup matrices
        nonbonded_lookup = onp.full(num_species, -1, dtype=int)
        nonbonded_data = onp.zeros((num_species, 3))
        for idx, particle in enumerate(non_bonded):
            i, _, *params = particle
            s = mapping[i.decode('UTF-8').strip()]
            nonbonded_lookup[s] = idx
            nonbonded_data[idx, :] = onp.asarray(params)

        data["nonbonded"] = jnp.asarray(nonbonded_data)
        lookup["nonbonded"] = jnp.asarray(nonbonded_lookup)

        bonds = onp.genfromtxt(
            StringIO(ff["bonded"]["bondtypes"]),
            dtype=None, delimiter=","
        )

        # Fill up to indicate which bonds are not provided by the force field
        bond_lookup = onp.full(
            (num_species, num_species), -1, dtype=int)
        bond_data = onp.zeros((max([len(bonds), 1]), 2))
        for idx, bond in enumerate(bonds):
            i, j, *params = bond
            s1 = mapping[i.decode('UTF-8').strip()]
            s2 = mapping[j.decode('UTF-8').strip()]

            bond_lookup[s1, s2] = idx
            bond_lookup[s2, s1] = idx
            bond_data[idx, :] = onp.asarray(params)

        data["bonded"]["bonds"] = jnp.asarray(bond_data)
        lookup["bonded"]["bonds"] = jnp.asarray(bond_lookup)

        angles = onp.genfromtxt(
            StringIO(ff["bonded"]["angletypes"]),
            dtype=None, delimiter=","
        )

        angle_lookup = onp.full(
            (num_species, num_species, num_species), -1, dtype=int)
        angle_data = onp.zeros((max([len(angles), 1]), 2))
        for idx, angle in enumerate(angles):
            i, j, k, *params = angle
            s1 = mapping[i.decode('UTF-8').strip()]
            s2 = mapping[j.decode('UTF-8').strip()]
            s3 = mapping[k.decode('UTF-8').strip()]

            angle_lookup[s1, s2, s3] = idx
            angle_lookup[s3, s2, s1] = idx
            angle_data[idx, :] = onp.asarray(params)

        data["bonded"]["angles"] = jnp.asarray(angle_data)
        lookup["bonded"]["angles"] = jnp.asarray(angle_lookup)

        dihedrals = onp.genfromtxt(
            StringIO(ff["bonded"]["dihedraltypes"]),
            dtype=None, delimiter=","
        )

        dihedrals_lookup = onp.full(
            (num_species, num_species, num_species, num_species), -1, dtype=int)
        dihedral_data = onp.zeros((max([len(dihedrals), 1]), 3))
        for idx, dihedral in enumerate(dihedrals):
            i, j, k, l, *params = dihedral
            s1 = mapping[i.decode('UTF-8').strip()]
            s2 = mapping[j.decode('UTF-8').strip()]
            s3 = mapping[k.decode('UTF-8').strip()]
            s4 = mapping[l.decode('UTF-8').strip()]

            dihedrals_lookup[s1, s2, s3, s4] = idx
            dihedrals_lookup[s4, s3, s2, s1] = idx
            dihedral_data[idx, :] = onp.asarray(params)

        data["bonded"]["dihedrals"] = jnp.asarray(dihedral_data)
        lookup["bonded"]["dihedrals"] = jnp.asarray(dihedrals_lookup)

        return cls(data, lookup, mapping)

    def write_ff(self, fname):
        reverse_mapping = {
            value: key for key, value in self._mapping.items()
        }

        # TODO: Read out the data more efficiently
        bond_data = [
            (i, j, self._lookup["bonded"]["bonds"][i, j])
            for i, j in itertools.product(range(self.max_species), range(self.max_species))
            if self._lookup["bonded"]["bonds"][i, j] >= 0
            and i < j
        ]

        print(bond_data)

        bondtypes = "#    i,    j,    b0,    kb\n"
        bondtypes += "\n".join(
            ",".join([
                f"{reverse_mapping[i]}".rjust(5, " "),
                f"{reverse_mapping[j]}".rjust(5, " "),
                f"{self._data['bonded']['bonds'][idx][0] :.5f}".rjust(7, " "),
                f"{self._data['bonded']['bonds'][idx][0] :.1f}".rjust(7, " "),
            ])
            for i, j, idx in bond_data
        )

        return bondtypes

    @property
    def get_data(self):
        return self._data

    @property
    def max_species(self):
        return max(self._mapping.values()) + 1

    def mapping(self, by_name=False):
        def mapping_fn(symbol: int, is_water: bool, name: str = "", **kwargs):
            if by_name:
                return self._mapping[name]

            if is_water:
                return self._mapping[symbol + "W"]
            else:
                return self._mapping[symbol]
        return mapping_fn

    def get_nonbonded_params(self, s):
        idx = self._lookup["nonbonded"][s]
        data = self._data["nonbonded"][idx]
        valid = idx >= 0
        return data, valid

    def get_bond_params(self, s1, s2):
        idx = self._lookup["bonded"]["bonds"][s1, s2]
        data = self._data["bonded"]["bonds"][idx]
        valid = idx >= 0
        return data, valid

    def get_angle_params(self, s1, s2, s3):
        idx = self._lookup["bonded"]["angles"][s1, s2, s3]
        data = self._data["bonded"]["angles"][idx]
        valid = idx >= 0
        return data, valid

    def get_dihedral_params(self, s1, s2, s3, s4):
        idx = self._lookup["bonded"]["dihedrals"][s1, s2, s3, s4]
        print(f"Idx is {idx}")
        data = self._data["bonded"]["dihedrals"][idx]
        valid = idx >= 0
        return data, valid

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        structure, lookup, mapping = aux_data
        data = tree_util.tree_unflatten(structure, children)
        return cls(data, lookup, mapping)

    def tree_flatten(self):
        children, structure = tree_util.tree_flatten(self._data)
        aux_data = (structure, self._lookup, self._mapping)
        return children, aux_data

@tree_util.register_pytree_node_class
class Topology:
    """Class documenting the topology of a system."""

    def __init__(self,
                 num_particles: int,
                 species: ArrayLike = None,
                 bond_idx: ArrayLike = None,
                 angle_idx: ArrayLike = None,
                 dihedral_idx: ArrayLike = None,
                 improper_dihedral_idx: ArrayLike = None):
        if bond_idx is None:
            bond_idx = jnp.zeros((0, 2), dtype=jnp.int_)
        if angle_idx is None:
            angle_idx = jnp.zeros((0, 3), dtype=jnp.int_)
        if dihedral_idx is None:
            dihedral_idx = jnp.zeros((0, 4), dtype=jnp.int_)
        if improper_dihedral_idx is None:
            improper_dihedral_idx = jnp.zeros((0, 4), dtype=jnp.int_)

        self._num_particles = num_particles
        self._species = species
        self._bond_idx = bond_idx
        self._angle_idx = angle_idx
        self._dihedral_idx = dihedral_idx
        self._improper_dihedral_idx = improper_dihedral_idx


    def get_padded_topology(self,
                            num_particles: int,
                            max_species: int = 1,
                            max_bonds: int = 0,
                            max_angles: int = 0,
                            max_dihedrals: int = 0,
                            max_improper_dihedrals: int = 0):

        # Create a new data structure by padding the lists
        # TODO: Check that no files have less angles than given.
        #       Pad all the indices and replace the old invalid particle indices
        #       by the new num_atoms
        # TODO: Replace the invalid indices by the new num_particles
        raise NotImplementedError("Padding not yet implemented")


    def get_atom_species(self, idx=None):
        if self._species is None:
            return None

        if idx is None:
            idx = jnp.arange(self._num_particles)
        if self._species.ndim == 0:
            species = jnp.full_like(idx, self._species, dtype=jnp.int_)
        else:
            species = self._species[idx]
        species = jnp.where(
            idx < self._num_particles, species, -1
        )
        return species

    def get_bonds(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._bond_idx
        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def get_angles(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._angle_idx
        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1]),
                self.get_atom_species(idxs[:, 2])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def get_dihedrals(self):
        """Returns the indices, species and mask for all angles."""
        idxs = self._dihedral_idx
        species = jnp.stack(
            (
                self.get_atom_species(idxs[:, 0]),
                self.get_atom_species(idxs[:, 1]),
                self.get_atom_species(idxs[:, 2]),
                self.get_atom_species(idxs[:, 3])
            ), axis=-1
        )
        mask = idxs[:, 0] < self._num_particles
        return idxs, species, mask

    def prune_topology(self, force_field: ForceField):
        """Remove all bonds, angles, etc. not parametrized by the force field.

        Args:
            force_field: Force field to identify, which bonds, etc. are
                parametrized.

        Returns:
            Returns a topology with potentially reduced size.
        """

        bonds_idx, species, mask = self.get_bonds()
        mask = onp.logical_and(mask, force_field.get_bond_params(*species.T)[1])
        bonds_idx = bonds_idx[mask]

        angle_idx, species, mask = self.get_angles()
        mask = onp.logical_and(mask, force_field.get_angle_params(*species.T)[1])
        angle_idx = angle_idx[mask]

        dihedral_idx, species, mask = self.get_dihedrals()
        mask = onp.logical_and(mask, force_field.get_dihedral_params(*species.T)[1])
        dihedral_idx = dihedral_idx[mask]

        top = Topology(
            self._num_particles, self._species, bonds_idx, angle_idx,
            dihedral_idx, None
        )
        return top


    @classmethod
    def concatenate(cls, *topologies):
        """Concatenate the topologies. """
        pass

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)

    def tree_flatten(self):
        children = (
            self._species, self._bond_idx, self._angle_idx,
            self._dihedral_idx, self._improper_dihedral_idx,
        )
        aux_data = (
            self._num_particles,
        )
        return children, aux_data


    @classmethod
    def from_mdtraj(cls, topology, mapping=None):
        """Creates a topology instance from a mdtraj topology instance."""

        graph = topology.to_bondgraph()
        nx = importlib.import_module("networkx")

        if mapping is None:
            mapping = lambda **kwargs: kwargs["number"]

        species = [
            mapping(
                number=n.element.number, name=n.name,
                symbol=n.element.symbol, code=n.residue.code,
                is_water=n.residue.is_water, is_protein=n.residue.is_protein
            )
            for n in graph.nodes
        ]
        for node in graph.nodes:
            print(node)

        # Retrive topology from graph
        bonds = []
        angles = []
        dihedrals = []

        # Discover all molecules (connected graphs)
        for mol in nx.k_edge_subgraphs(graph, 1):
            # Find all bonds, angles and dihedrals between pairs of atoms
            for atm1, atm2 in itertools.combinations(mol, 2):
                if not nx.has_path(graph, atm1, atm2):
                    # Pairs not part of the same molecule
                    continue

                all_paths = nx.all_simple_paths(graph, atm1, atm2, cutoff=4)
                for path in all_paths:
                    # Sort the path and always start with the lowest numbered
                    # species
                    if species[path[0].index] > species[path[-1].index]:
                        path.reverse()

                    # Identify bonds, angles and dihedrals
                    if len(path) == 2:
                        bonds.append([a.index for a in path])
                    elif len(path) == 3:
                        angles.append([a.index for a in path])
                    elif len(path) == 4:
                        dihedrals.append([a.index for a in path])

        return cls(
            len(species), species=jnp.asarray(species),
            bond_idx=jnp.asarray(bonds), angle_idx=jnp.asarray(angles),
            dihedral_idx=jnp.asarray(dihedrals)
        )


def init_bond_potential(displacement_fn, topology, force_field):
    bond_idx, bond_species, bond_mask = topology.get_bonds()
    bond_params, mask = force_field.get_bond_params(
        bond_species[:, 0], bond_species[:, 1])
    # Mask out bonds by setting their energy to zero
    b0 = bond_params[:, 0]
    kb = jnp.where(
        jnp.logical_and(mask, bond_mask), bond_params[:, 1], 0.0
    )
    bond_potential = energy.simple_spring_bond(
        displacement_fn, bond_idx, length=b0, epsilon=kb)
    return bond_potential


def init_angle_potential(displacement_fn, topology, force_field):
    angle_idx, angle_species, angle_mask = topology.get_angles()
    angle_params, mask = force_field.get_angle_params(
        angle_species[:, 0], angle_species[:, 1], angle_species[:, 2])
    # Mask out bonds by setting their energy to zero
    th0 = angle_params[:, 0]
    kth = jnp.where(
        jnp.logical_and(mask, angle_mask), angle_params[:, 1], 0.0
    )
    angle_potential = custom_energy.harmonic_angle(
        displacement_fn, angle_idx, th0=th0, kth=kth)
    return angle_potential


def init_nonbonded_potentia(displacement_fn, topology, force_field: ForceField):
    species = topology.get_atom_species()
    nonbonded_params, mask = force_field.get_nonbonded_params(species)

    lennard_jones_fn = custom_energy.generic_repulsion_neighborlist(
        displacement_fn, initialize_neighbor_list=False, box_size=0.0,
        sigma=0.3156, epsilon=1.0,
        # sigma=nonbonded_params[:, 1], epsilon=nonbonded_params[:, 2]
        # sigma=(lambda s1, s2: 0.5 * (s1 + s2), nonbonded_params[:, 1]),
        # epsilon=(lambda e1, e2: jnp.sqrt(e1 * e2), nonbonded_params[:, 2])
    )
    return lennard_jones_fn


def init_prior_potential(displacement_fn, mask_bonded: bool = True):
    """Initializes the prior template.

    Args:
        displacement_fn:
        mask_bonded: Excluded non-bonded interactions between bonded atoms via
            masking them in the neighbor list. This might be ineffective if the
            number of bonds is very large. Otherwise, the non-bonded interaction
            is simply substracted from the ...

    """
    def prior_potential_template(topology: Topology, force_field: ForceField):

        # Bonded
        bond_potential = init_bond_potential(displacement_fn, topology, force_field)
        angle_potential = init_angle_potential(displacement_fn, topology, force_field)

        # Angles
        # angle_potential = ini

        # Dihedrals

        # Nonbonded
        nonbonded_potential_fn = init_nonbonded_potentia(displacement_fn, topology, force_field)

        def potential_fn(position, neighbor=None, **kwargs):
            pot = 0.0
            pot += bond_potential(position)
            pot += angle_potential(position)

            if mask_bonded:
                # Exclude pairs connected via bonds and angles from the
                # neighbor list to exclude them also from beeing considered
                # in the nonbonded potential computation
                neighbor = sparse_graph.subtract_topology_from_neighbor_list(
                    neighbor, topology)
                pot += nonbonded_potential_fn(position, neighbor)
            else:
                #
                pass

            return pot

        return potential_fn

    return prior_potential_template


if __name__ == "__main__":

    # TODO: Taks still open
    #       - Implement the dihedral potentials
    #       - Check pruning
    #       - Check bon removal
    #       - Write way to save trained parameters to ff
    #       - Correct nonbonded interactions in cg prior force field
    #       - Add better documentation and final example

    import jax
    from jax import tree_util
    from jax_md import space, partition


    import mdtraj
    top = mdtraj.load("../../examples/alanine_dipeptide/data/confs/heavy_2_7nm.gro")
    # top = mdtraj.load("../../examples/alanine_dipeptide/data/confs/atomistic.pdb")
    # print(top)
    names = [a.element.symbol for a in top.topology.atoms]

    # force_field = ForceField.load_ff("../../data/atomistic_ff.toml")
    force_field = ForceField.load_ff("../../examples/alanine_dipeptide/data/force_fields/heavy.toml")
    print(force_field._mapping)
    topology = Topology.from_mdtraj(top.topology, mapping=force_field.mapping(by_name=True))

    topology = topology.prune_topology(force_field)

    displacement = space.periodic_general(20.0, fractional_coordinates=False)[0]
    position = jnp.asarray(top[0].xyz[0, ...]) * 0.1 # Convert from A to nm

    print(position)
    print(position.shape)

    neighbor_list = partition.neighbor_list(displacement, 5.0, r_cutoff=0.5, disable_cell_list=True)
    nbrs = neighbor_list.allocate(position)
    #
    # print(jax.jit(sparse_graph.subtract_topology_from_neighbor_list)(nbrs, topology).idx)

    energy_fn_template = init_prior_potential(displacement)

    @jax.jit
    @jax.value_and_grad
    def trial_fn(position):
        energy_fn = energy_fn_template(topology, force_field)
        return energy_fn(position, neighbor=nbrs)

    lr = 0.1

    for idx in range(1000):
        value, grad = trial_fn(position)
        grad = jnp.clip(grad, -1e1, 1e1)
        lr *= 0.975
        position -= lr * grad
        print(f"Energy {value}")

    io.save_gro(position, 5.0, filename="../../examples/alanine_dipeptide/data/confs/atomistic.gro", group="ALA", species=names)

    print(topology.get_bonds()[1])
    print(topology.get_atom_species())

    @jax.jit
    @jax.value_and_grad
    def trial_fn(force_field):
        energy_fn = energy_fn_template(topology, force_field)
        return energy_fn(position, neighbor=nbrs)

    print(trial_fn(force_field)[1]._data)

    print(force_field.write_ff("test"))