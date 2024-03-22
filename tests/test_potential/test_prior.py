from pathlib import Path

import numpy as np
import numpy as onp
import pytest

import mdtraj

from jax import tree_util
import jax.numpy as jnp

from chemtrain.potential.prior import Topology, ForceField

class TestForceField:

    @pytest.mark.parametrize("sample_conf", [
            (
                "examples/alanine_dipeptide/data/confs/heavy_2_7nm.gro",
                "examples/alanine_dipeptide/data/force_fields/heavy_untrained.toml",
                True
            ),
        ])
    def test_io(self, tmp_path, sample_conf):
        # Test whether loading and saving of a force field works by comparing
        # the loaded values for multiple topologies

        top, ff_file, by_name = sample_conf
        top = mdtraj.load_topology(top)

        print(Path.cwd())
        assert (Path.cwd() / "chemtrain").is_dir()

        org_force_field = ForceField.load_ff(ff_file)
        topology = Topology.from_mdtraj(top, org_force_field.mapping(by_name=by_name))
        org_parameters = (
            org_force_field.get_bond_params(*topology.get_bonds()[1].T),
            org_force_field.get_angle_params(*topology.get_angles()[1].T),
            org_force_field.get_dihedral_params(*topology.get_dihedrals()[1].T)
        )

        # Test whether saving and loading changes the force field parameters
        new_ff_file = tmp_path / Path(ff_file).name
        org_force_field.write_ff(new_ff_file)
        new_force_field = ForceField.load_ff(new_ff_file)

        new_parameters = (
            new_force_field.get_bond_params(*topology.get_bonds()[1].T),
            new_force_field.get_angle_params(*topology.get_angles()[1].T),
            new_force_field.get_dihedral_params(*topology.get_dihedrals()[1].T)
        )

        assert tree_util.tree_structure(org_parameters) == tree_util.tree_structure(new_parameters)
        for old, new in zip(tree_util.tree_leaves(org_parameters), tree_util.tree_leaves(new_parameters)):
            assert onp.all(onp.isclose(old, new))


class TestTopology:

    @pytest.mark.skip
    def test_pruning(self):
        pass

        # Test whether pruning works

    @pytest.mark.skip
    def test_concatenation(self):
        pass

        # Test whether concatenating topologies work
