import itertools
from io import StringIO

from jax import tree_util
from jax.typing import ArrayLike
import jax.numpy as jnp

import numpy as onp

import tomli

import importlib

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


    def get_atom_species(self, idx):
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
                number=n.element.number, name=n.element.name, code=n.residue.code,
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
        nonbonded_lookup = onp.zeros(num_species, dtype=int)
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
            (num_species, num_species), num_species, dtype=int)
        bond_data = onp.zeros((len(bonds), 2))
        for idx, bond in enumerate(bonds):
            i, j, *params = bond
            s1 = mapping[i.decode('UTF-8').strip()]
            s2 = mapping[j.decode('UTF-8').strip()]

            bond_lookup[s1, s2] = idx
            bond_lookup[s2, s1] = idx
            bond_data[idx, :] = onp.asarray(params)

        data["bonded"]["bonds"] = jnp.asarray(bond_data)
        lookup["bonded"]["bonds"] = jnp.asarray(bond_lookup)

        return cls(data, lookup, mapping)

    @property
    def max_species(self):
        return max(self._mapping.values()) + 1

    def get_bond_params(self, s1, s2):
        idx = self._lookup["bonded"]["bonds"][s1, s2]
        data = self._data["bonded"]["bonds"][idx]
        valid = idx < self.max_species
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


if __name__ == "__main__":

    import jax
    from jax import tree_util

    import mdtraj
    top = mdtraj.load("../../examples/alanine_dipeptide/data/confs/atomistic.pdb")
    print(top)
    print(top.topology.to_bondgraph().edges)


