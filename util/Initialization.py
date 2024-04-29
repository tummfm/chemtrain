# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of functions to initialize Jax, M.D. simulations."""
from functools import partial

import chex
import haiku as hk
from jax import random, vmap, lax, numpy as jnp
from jax_md import util, simulate, partition, space, energy, minimize
import numpy as onp
from scipy import interpolate as sci_interpolate

from chemtrain import (util as chem_util)
from chemtrain.trajectory import traj_util
from chemtrain.potential import neural_networks, layers, dropout
from chemtrain.quantity import observables
from chemtrain.jax_md_mod import (custom_energy, custom_space, custom_quantity,
                                      custom_simulator, custom_interpolate)

from typing import Dict
from chemtrain.typing import TargetDict, ComputeFn

Array = util.Array


@chex.dataclass
class InitializationClass:
    """A dataclass containing initialization information.

    Notes:
      careful: dataclasses.astuple(InitializationClass) sometimes
      changes type from jnp.Array to onp.ndarray

    Attributes:
        r_init: Initial Positions
        box: Simulation box size
        kbT: Target thermostat temperature times Boltzmann constant
        mass: Particle masses
        dt: Time step size
        species: Species index for each particle
        ref_press: Target pressure for barostat
        temperature: Thermostat temperature; only used for computation of
                     thermal expansion coefficient and heat capacity
    """
    r_init: Array
    box: Array
    kbt: float
    masses: Array
    dt: float
    species: Array = None
    ref_press: float = 1.
    temperature: float = None


def select_target_rdf(target_rdf, rdf_start=0., nbins=300):
    if target_rdf == 'LJ':
        reference_rdf = util.f32(onp.loadtxt('data/LJ_reference_RDF.csv'))
        rdf_cut = 1.5
        raise NotImplementedError
    elif target_rdf == 'SPC':
        reference_rdf = onp.loadtxt('data/water_models/SPC_955_RDF.csv')
        rdf_cut = 1.0
    elif target_rdf == 'SPC_FW':
        reference_rdf = onp.loadtxt('data/water_models/SPC_FW_RDF.csv')
        rdf_cut = 1.0
    elif target_rdf == 'TIP4P/2005':
        reference_rdf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_300_COM_RDF.csv')
        rdf_cut = 0.85
    elif target_rdf == 'Water_Ox':
        reference_rdf = onp.loadtxt('data/experimental/O_O_RDF.csv')
        rdf_cut = 1.0
    else:
        raise ValueError(f'The reference rdf {target_rdf} is not implemented.')

    rdf_bin_centers, rdf_bin_boundaries, sigma_rdf = \
        custom_quantity.rdf_discretization(rdf_cut, nbins, rdf_start)
    rdf_spline = sci_interpolate.interp1d(reference_rdf[:, 0],
                                          reference_rdf[:, 1], kind='cubic')
    reference_rdf = util.f32(rdf_spline(rdf_bin_centers))
    rdf_struct = custom_quantity.RDFParams(reference_rdf, rdf_bin_centers,
                                           rdf_bin_boundaries, sigma_rdf)
    return rdf_struct


def select_target_adf(target_adf, r_outer, r_inner=0., nbins_theta=150):
    if target_adf == 'Water_Ox':
        reference_adf = onp.loadtxt('data/experimental/O_O_O_ADF.csv')
    elif target_adf == 'TIP4P/2005':
        reference_adf = onp.loadtxt('data/water_models/'
                                    'TIP4P-2005_150_COM_ADF.csv')
    else:
        raise ValueError(f'The reference adf {target_adf} is not implemented.')

    adf_bin_centers, sigma_adf = custom_quantity.adf_discretization(nbins_theta)

    adf_spline = sci_interpolate.interp1d(reference_adf[:, 0],
                                          reference_adf[:, 1], kind='cubic')
    reference_adf = util.f32(adf_spline(adf_bin_centers))

    adf_struct = custom_quantity.ADFParams(reference_adf, adf_bin_centers,
                                           sigma_adf, r_outer, r_inner)
    return adf_struct


def select_target_tcf(target_tcf, tcf_cut, tcf_start=0.2, nbins=30):
    if target_tcf == 'TIP4P/2005':
        if tcf_cut == 0.5:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut05.npy')
            dx_bin = 0.3 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.5 - dx_bin/2., 50)
        elif tcf_cut == 0.6:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut06.npy')
            dx_bin = 0.4 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.6 - dx_bin/2., 50)
        elif tcf_cut == 0.8:
            reference_tcf = onp.load('data/water_models/TIP4P-2005_1k_50b_TCF'
                                     '_cut08.npy')
            dx_bin = 0.6 / nbins
            bins_centers = onp.linspace(0.2 + dx_bin/2., 0.8 - dx_bin/2., 50)
        else:
            raise ValueError(f'The cutoff {tcf_cut} is not implemented.')
    else:
        raise ValueError(f'The reference tcf {target_tcf} is not implemented.')

    (sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
     tcf_z_bin_centers) = custom_quantity.tcf_discretization(tcf_cut, nbins,
                                                             tcf_start)

    equilateral = onp.diagonal(onp.diagonal(reference_tcf))
    tcf_spline = sci_interpolate.interp1d(bins_centers, equilateral,
                                          kind='cubic')
    reference_tcf = util.f32(tcf_spline(tcf_x_binx_centers[0, :, 0]))
    tcf_struct = custom_quantity.TCFParams(
        reference_tcf, sigma_tcf, volume, tcf_x_binx_centers, tcf_y_bin_centers,
        tcf_z_bin_centers
    )
    return tcf_struct


def select_target_dihedral_dist(target, nbins=150):
    if target == 'alanine-phi':
        angles, dist = onp.loadtxt('data/distributions/heavy_phi.csv', unpack=True)
        angles *= jnp.pi / 180.
        bonds = onp.array([
            [1, 3, 4, 6],
        ])
    elif target == 'alanine-psi':
        angles, dist = onp.loadtxt('data/distributions/heavy_psi.csv', unpack=True)
        angles *= jnp.pi / 180.
        bonds = onp.array([
            [3, 4, 6, 8],
        ])
    else:
        raise ValueError(f"Target {target} is unknown.")

    bin_centers, sigma = custom_quantity.dihedral_discretization(nbins)
    bin_boundaries = 0.5 * (bin_centers[0:-1] + bin_centers[1:])
    bin_boundaries = jnp.concatenate(
        (jnp.asarray([-jnp.pi]), bin_boundaries, jnp.asarray([jnp.pi])))

    dihedral_spline = sci_interpolate.interp1d(angles, dist, kind='cubic')
    reference_dist = util.f32(dihedral_spline(bin_centers))

    dihedral_struct = custom_quantity.BondDihedralParams(
        reference=reference_dist, sigma=sigma, bonds=bonds,
        bin_centers=bin_centers, bin_boundaries=bin_boundaries)

    return dihedral_struct


def prior_potential(prior_fns, pos, neighbor, **dynamic_kwargs):
    """Evaluates the prior potential for a given snapshot."""
    sum_priors = 0.
    if prior_fns is not None:
        for key in prior_fns:
            sum_priors += prior_fns[key](pos, neighbor=neighbor,
                                         **dynamic_kwargs)
    return sum_priors


def select_priors(displacement, prior_constants, prior_idxs, kbt=None):
    """Build prior potential from combination of classical potentials."""
    prior_fns = {}
    if 'bond' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for bond prior.'
        bond_mean, bond_variance = prior_constants['bond']
        bonds = prior_idxs['bond']
        prior_fns['bond'] = energy.simple_spring_bond(
            displacement, bonds, length=bond_mean, epsilon=kbt / bond_variance)

    if 'angle' in prior_constants:
        assert kbt is not None, 'Need to provide kbT for angle prior.'
        angle_mean, angle_variance = prior_constants['angle']
        angles = prior_idxs['angle']
        prior_fns['angle'] = custom_energy.harmonic_angle(
            displacement, angles, angle_mean, angle_variance, kbt)

    if 'LJ' in prior_constants:
        lj_sigma, lj_epsilon = prior_constants['LJ']
        lj_idxs = prior_idxs['LJ']
        prior_fns['LJ'] = custom_energy.lennard_jones_nonbond(
            displacement, lj_idxs, lj_sigma, lj_epsilon)

    if 'repulsive' in prior_constants:
        re_sigma, re_epsilon, re_cut = prior_constants['repulsive']
        prior_fns['repulsive'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=12,
            initialize_neighbor_list=False, r_onset=0.9 * re_cut,
            r_cutoff=re_cut)

    if 'repulsive' in prior_constants:
        re_sigma, re_epsilon, re_cut = prior_constants['repulsive']
        prior_fns['repulsive'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=12,
            initialize_neighbor_list=False, r_onset=0.9 * re_cut,
            r_cutoff=re_cut)

    if 'soft_sphere_repulsion' in prior_constants:
        re_sigma, re_epsilon = prior_constants['soft_sphere_repulsion']
        prior_fns['soft_sphere_repulsion'] = custom_energy.generic_repulsion_neighborlist(
            displacement, sigma=re_sigma, epsilon=re_epsilon, exp=12,
            initialize_neighbor_list=False, r_onset=0.0,
            r_cutoff=re_sigma)

    if 'soft_sphere' in prior_constants:
      sph_sigma, sph_epsilon = prior_constants['soft_sphere']
      prior_fns['soft_sphere'] = custom_energy.soft_sphere_neighbor_list(
        displacement, sigma=sph_sigma, epsilon=sph_epsilon, alpha=12,
        initialize_neighbor_list=False
      )

    if 'dihedral' in prior_constants:
        dih_phase, dih_constant, dih_n = prior_constants['dihedral']
        dihdral_idxs = prior_idxs['dihedral']
        prior_fns['dihedral'] = custom_energy.periodic_dihedral(
            displacement, dihdral_idxs, dih_phase, dih_constant, dih_n)

    if 'repulsive_nonbonded' in prior_constants:
        # only repulsive part of LJ via idxs instead of nbrs list
        ren_sigma, ren_epsilon = prior_constants['repulsive_nonbonded']
        ren_idxs = prior_idxs['repulsive_nonbonded']
        prior_fns['repulsive_1_4'] = custom_energy.generic_repulsion_nonbond(
            displacement, ren_idxs, sigma=ren_sigma, epsilon=ren_epsilon, exp=6)

    return prior_fns


def select_protein(protein, prior_list):
    idxs = {}
    constants = {}
    if protein == 'heavy_alanine_dipeptide':
        print('Distinguishing different C_Hx atoms')
        species = jnp.array([6, 1, 8, 7, 2, 6, 1, 8, 7, 6])
        if 'bond' in prior_list:
            bond_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq_bond'
                                 '_length.npy')
            bond_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                     '_bond_variance.npy')
            bond_idxs = onp.array([[0, 1],
                                   [1, 2],
                                   [1, 3],
                                   [4, 6],
                                   [6, 7],
                                   [4, 5],
                                   [3, 4],
                                   [6, 8],
                                   [8, 9]])
            idxs['bond'] = bond_idxs
            constants['bond'] = (bond_mean, bond_variance)

        if 'angle' in prior_list:
            angle_mean = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                  '_angle.npy')
            angle_variance = onp.load('data/prior/Alanine_dipeptide_heavy_eq'
                                      '_angle_variance.npy')
            angle_idxs = onp.array([[0, 1, 2],
                                    [0, 1, 3],
                                    [2, 1, 3],
                                    [1, 3, 4],
                                    [3, 4, 5],
                                    [3, 4, 6],
                                    [5, 4, 6],
                                    [4, 6, 7],
                                    [4, 6, 8],
                                    [7, 6, 8],
                                    [6, 8, 9]])
            idxs['angle'] = angle_idxs
            constants['angle'] = (angle_mean, angle_variance)

        if 'LJ' in prior_list:
            lj_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            lj_epsilon = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                  'epsilon.npy')
            lj_idxs = onp.array([[0, 5],
                                 [0, 6],
                                 [0, 7],
                                 [0, 8],
                                 [0, 9],
                                 [1, 7],
                                 [1, 8],
                                 [1, 9],
                                 [2, 5],
                                 [2, 6],
                                 [2, 7],
                                 [2, 8],
                                 [2, 9],
                                 [3, 9],
                                 [5, 9]])
            idxs['LJ'] = lj_idxs
            constants['LJ'] = (lj_sigma, lj_epsilon)

        if 'dihedral' in prior_list:
            dihedral_phase = onp.load('data/prior/Alanine_dipeptide_heavy_'
                                      'dihedral_phase.npy')
            dihedral_constant = onp.load('data/prior/Alanine_dipeptide_heavy'
                                         '_dihedral_constant.npy')
            dihedral_n = onp.load('data/prior/Alanine_dipeptide_heavy_dihedral'
                                  '_multiplicity.npy')

            dihedral_idxs = onp.array([[1, 3, 4, 6],
                                       [3, 4, 6, 8],
                                       [0, 1, 3, 4],
                                       [2, 1, 3, 4],
                                       [1, 3, 4, 5],
                                       [5, 4, 6, 8],
                                       [4, 6, 8, 9],
                                       [7, 6, 8, 9]])
            idxs['dihedral'] = dihedral_idxs
            constants['dihedral'] = (dihedral_phase, dihedral_constant,
                                     dihedral_n)

        if 'repulsive_nonbonded' in prior_list:
            # repulsive part of the LJ
            if 'LJ' in prior_list:
                raise ValueError('Not sensible to have LJ and repulsive part of'
                                 ' LJ together. Choose one.')
            ren_sigma = onp.load('data/prior/Alanine_dipeptide_heavy_sigma.npy')
            ren_epsilon = onp.load('data/prior/Alanine_dipeptide'
                                   '_heavy_epsilon.npy')
            ren_idxs = onp.array([[0, 5],
                                  [0, 6],
                                  [0, 7],
                                  [0, 8],
                                  [0, 9],
                                  [1, 7],
                                  [1, 8],
                                  [1, 9],
                                  [2, 5],
                                  [2, 6],
                                  [2, 7],
                                  [2, 8],
                                  [2, 9],
                                  [3, 9],
                                  [5, 9]])
            idxs['repulsive_nonbonded'] = ren_idxs
            constants['repulsive_nonbonded'] = (ren_sigma, ren_epsilon)
    else:
        raise ValueError(f'The protein {protein} is not implemented.')
    return species, idxs, constants


def build_quantity_dict(pos_init, box_tensor, displacement, energy_fn_template,
                        nbrs, target_dict, init_class, expansion_formula=False
                        ) -> [Dict[str, ComputeFn], TargetDict]:
    targets = {}
    compute_fns = {}
    kj_mol_nm3_to_bar = 16.6054


    if 'kappa' in target_dict or 'alpha' in target_dict or 'cp' in target_dict:
        compute_fns['volume'] = custom_quantity.volume_npt
        if 'alpha' in target_dict or 'cp' in target_dict:
            compute_fns['energy'] = custom_quantity.energy_wrapper(
                energy_fn_template)

    if expansion_formula:
      init_traj_fn = observables.init_linear_traj_mean_fn
    else:
      init_traj_fn = observables.init_traj_mean_fn

    if 'rdf' in target_dict:
        rdf_struct = target_dict['rdf']
        rdf_fn = custom_quantity.init_rdf(displacement, rdf_struct, box_tensor)
        rdf_dict = {'target': rdf_struct.reference, 'gamma': 1.,
                    'traj_fn': init_traj_fn('rdf')}
        targets['rdf'] = rdf_dict
        compute_fns['rdf'] = rdf_fn

    if 'adf' in target_dict:
        adf_struct = target_dict['adf']
        adf_fn = custom_quantity.init_adf_nbrs(
            displacement, adf_struct, smoothing_dr=0.01, r_init=pos_init,
            nbrs_init=nbrs)
        adf_target_dict = {'target': adf_struct.reference, 'gamma': 1.,
                           'traj_fn': init_traj_fn('adf')}
        targets['adf'] = adf_target_dict
        compute_fns['adf'] = adf_fn

    if 'angle' in target_dict:
        angle_struct = target_dict['angle']
        angle_fn = custom_quantity.init_bond_angle_distribution(
            displacement, angle_struct, reference_box=box_tensor
        )
        angle_target_dict = {'target': angle_struct.reference, 'gamma': 1.,
                             'traj_fn': init_traj_fn('angle')}
        targets['angle'] = angle_target_dict
        compute_fns['angle'] = angle_fn

    if 'tcf' in target_dict:
        tcf_struct = target_dict['tcf']
        tcf_fn = custom_quantity.init_tcf_nbrs(displacement, tcf_struct,
                                               box_tensor, nbrs_init=nbrs,
                                               batch_size=1000)
        tcf_target_dict = {'target': tcf_struct.reference, 'gamma': 1.,
                           'traj_fn': init_traj_fn('tcf')}
        targets['tcf'] = tcf_target_dict
        compute_fns['tcf'] = tcf_fn

    if 'pressure' in target_dict:
        pressure_fn = custom_quantity.init_pressure(energy_fn_template,
                                                    box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure'], 'gamma': 1.e-7,
            'traj_fn': init_traj_fn('pressure')}
        targets['pressure'] = pressure_target_dict
        compute_fns['pressure'] = pressure_fn

    if 'pressure_tensor' in target_dict:
        pressure_fn = custom_quantity.init_virial_stress_tensor(
            energy_fn_template, box_tensor)
        pressure_target_dict = {
            'target': target_dict['pressure_tensor'], 'gamma': 1.e-7,
            'traj_fn': init_traj_fn('pressure_tensor')}
        targets['pressure_tensor'] = pressure_target_dict
        compute_fns['pressure_tensor'] = pressure_fn

    if 'density' in target_dict:
        density_dict = {
            'target': target_dict['density'], 'gamma': 1.e-3,  # 1.e-5
            'traj_fn': init_traj_fn('density')
        }
        targets['density'] = density_dict
        compute_fns['density'] = custom_quantity.density

    if 'volume' in target_dict:
        volume_dict = {
            'target': target_dict['volume'], 'gamma': 1e-5,
            'traj_fn': init_traj_fn('volume')
        }
        targets['volume'] = volume_dict
        compute_fns['volume'] = custom_quantity.volume_npt

    if 'kappa' in target_dict:
        def compress_traj_fn(quantity_trajs):
            volume_traj = quantity_trajs['volume']
            kappa = observables.isothermal_compressibility_npt(volume_traj,
                                                                 init_class.kbt)
            return kappa

        comp_dict = {
            'target': target_dict['kappa'],
            'gamma': 1. / (5.e-5 * kj_mol_nm3_to_bar),
            'traj_fn': compress_traj_fn
        }
        targets['kappa'] = comp_dict

    if 'alpha' in target_dict:
        def thermo_expansion_traj_fn(quantity_trajs):
            alpha = observables.thermal_expansion_coefficient_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbt, init_class.ref_press)
            return alpha

        alpha_dict = {
            'target': target_dict['alpha'], 'gamma': 1.e4,
            'traj_fn': thermo_expansion_traj_fn
        }
        targets['alpha'] = alpha_dict

    if 'cp' in target_dict:
        n_particles, dim = pos_init.shape
        # assuming no reduction, e.g. due to rigid bonds
        n_dof = dim * n_particles

        def cp_traj_fn(quantity_trajs):
            cp = observables.specific_heat_capacity_npt(
                quantity_trajs['volume'], quantity_trajs['energy'],
                init_class.temperature, init_class.kbt, init_class.ref_press,
                n_dof)
            return cp

        cp_dict = {
            'target': target_dict['cp'], 'gamma': 10.,
            'traj_fn': cp_traj_fn
        }
        targets['cp'] = cp_dict

    return compute_fns, targets


def default_x_vals(r_cut, delta_cut):
    return jnp.linspace(0.05, r_cut + delta_cut, 100, dtype=jnp.float32)


def select_model(model, init_pos, displacement, box, model_init_key, kbt=None,
                 species=None, x_vals=None, fractional=True,
                 kbt_dependent=False, prior_constants=None, prior_idxs=None,
                 dropout_init_seed=None, constraint_dict=None, **energy_kwargs):
    if model == 'LJ':
        r_cut = 0.9
        init_params = jnp.array([0.2, 1.2], dtype=jnp.float32)  # initial guess
        lj_neighbor_energy = partial(
            custom_energy.customn_lennard_jones_neighbor_list, displacement,
            box, r_onset=0.8, r_cutoff=r_cut, dr_threshold=0.2,
            capacity_multiplier=1.25, fractional=fractional)
        neighbor_fn, _ = lj_neighbor_energy(sigma=init_params[0],
                                            epsilon=init_params[1])
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        def energy_fn_template(energy_params):
            # we only need to re-create energy_fn, neighbor function is re-used
            lj_energy = lj_neighbor_energy(
                sigma=energy_params[0], epsilon=energy_params[1],
                initialize_neighbor_list=False)
            return lj_energy
        
    elif model == "ME":
        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        # TODO: Create a MD model, i.e. that is proportional to the observable of the ensemble average to be constrained
        if constraint_dict is None:
            constraint_dict = {}
            
        compute_fns = {}
        init_params = {}

        # TODO: How to set this depending on the constraints?
        r_cut = 0.5
        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)
        
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=1.5)

        
        for key, settings in constraint_dict.items():
            if key == "rdf":
                # Initialize the rdf compute function and set the structure of the lagrange multipliers
                _rdf_fn = custom_quantity.init_rdf(displacement, settings, box)
                def rdf_fn(pos, *args, **kwargs):
                    return _rdf_fn(pos)

                compute_fns["rdf"] = {
                    'compute': rdf_fn, 'evals': settings.rdf_bin_centers,
                    'socket': jnp.linspace(0, 1, 100)
                }
                init_params["rdf"] = jnp.zeros(100)

            if key == "adf":
                adf_fn = custom_quantity.init_adf_nbrs(
                    displacement, settings, smoothing_dr=0.05, r_init=init_pos, nbrs_init=nbrs_init)

                compute_fns['adf'] = adf_fn
                init_params['adf'] = jnp.zeros(30)
                compute_fns["adf"] = {
                    'compute': adf_fn, 'evals': settings.adf_bin_centers,
                    'socket': jnp.linspace(0, jnp.pi, 30)
                }


        from collections import namedtuple
        pseudostate = namedtuple("pseudostate", ("position",))
        def energy_fn_template(params):
            multipliers = {
                key: custom_interpolate.InterpolatedUnivariateSpline(vals['socket'], params[key])(vals["evals"])
                for key, vals in compute_fns.items()
            }

            def energy_fn(pos, neighbor, **dynamic_kwargs):
                # Prior terms
                potential_energy = prior_potential(prior_fns, pos, neighbor, **dynamic_kwargs)
                # Biasing terms: Propotional to the instantaneous observables, which ensemble average should be matched
                for key in params.keys():
                    obs = compute_fns[key]['compute'](pseudostate(pos), neighbor, **dynamic_kwargs)
                    potential_energy += jnp.sum(multipliers[key] * obs)
                return potential_energy
            return energy_fn

    elif model == 'Tabulated':
        # TODO: change initial guess to generic LJ or random initialization
        # TODO adjust to new prior interface
        r_cut = 0.9
        delta_cut = 0.1
        if x_vals is None:
            x_vals = default_x_vals(r_cut, delta_cut)

        # load PMF initial guess
        # pmf_init = False  # for IBI
        pmf_init = False
        if pmf_init:
            # table_loc = 'data/tabulated_potentials/CG_potential_SPC_955.csv'
            table_loc = 'data/tabulated_potentials/IBI_initial_guess.csv'
            tabulated_array = onp.loadtxt(table_loc)
            # compute tabulated values at spline support points
            u_init_int = sci_interpolate.interp1d(tabulated_array[:, 0],
                                        tabulated_array[:, 1], kind='cubic')
            init_params = jnp.array(u_init_int(x_vals), dtype=jnp.float32)
        else:
            # random initialisation + prior
            init_params = 0.1 * random.normal(model_init_key, x_vals.shape)
            init_params = jnp.array(init_params, dtype=jnp.float32)
            prior_fn = custom_energy.generic_repulsion_neighborlist(
                displacement, sigma=0.3165, epsilon=1., exp=12,
                initialize_neighbor_list=False, r_onset=0.9 * r_cut,
                r_cutoff=r_cut)

        tabulated_energy = partial(
            custom_energy.tabulated_neighbor_list, displacement, x_vals,
            box_size=box, r_onset=(r_cut - 0.2), r_cutoff=r_cut,
            dr_threshold=0.05, capacity_multiplier=1.25
        )
        neighbor_fn, _ = tabulated_energy(init_params)

        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if pmf_init:
            def energy_fn_template(energy_params):
                tab_energy = tabulated_energy(energy_params,
                                              initialize_neighbor_list=False)
                return tab_energy
        else:  # with prior
            def energy_fn_template(energy_params):
                tab_energy = tabulated_energy(energy_params,
                                              initialize_neighbor_list=False)

                def energy_fn(pos, neighbor, **dynamic_kwargs):
                    return (tab_energy(pos, neighbor, **dynamic_kwargs)
                            + prior_fn(pos, neighbor=neighbor, **dynamic_kwargs)
                            )
                return energy_fn

    elif model == 'PairNN':
        # TODO adjust to new prior interface
        r_cut = 3.  # 3 sigma in LJ units
        hidden_layers = [64, 64]  # with 32 higher best force error

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.5,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional)
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)
        prior_fn = custom_energy.generic_repulsion_neighborlist(
            displacement,
            sigma=0.7,
            epsilon=1.,
            exp=12,
            initialize_neighbor_list=False,
            r_onset=0.9 * r_cut,
            r_cutoff=r_cut
        )

        init_fn, pair_nn_energy = neural_networks.pair_interaction_nn(
            displacement, r_cut, hidden_layers)
        if isinstance(model_init_key, list):
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init)
                           for key in model_init_key]
        else:
            init_params = init_fn(model_init_key, init_pos, neighbor=nbrs_init)

        def energy_fn_template(energy_params):
            pair_nn_energy_fix = partial(pair_nn_energy, energy_params,
                                         species=species)

            def energy_fn(pos, neighbor, **dynamic_kwargs):
                return (pair_nn_energy_fix(pos, neighbor, **dynamic_kwargs) +
                        prior_fn(pos, neighbor=neighbor, **dynamic_kwargs))
            return energy_fn

    elif model == 'CGDimeNet':
        r_cut = 0.5
        n_species = 10

        mlp_init = {
            'b_init': hk.initializers.Constant(0.),
            'w_init': layers.OrthogonalVarianceScalingInit(scale=1.)
        }

        neighbor_fn = partition.neighbor_list(displacement, box, r_cut,
                                              dr_threshold=0.05,
                                              capacity_multiplier=1.5,
                                              fractional_coordinates=fractional,
                                              disable_cell_list=True)

        # create neighborlist for init of GNN
        nbrs_init = neighbor_fn.allocate(init_pos, extra_capacity=0)

        if prior_constants is not None:
            prior_fns = select_priors(displacement, prior_constants, prior_idxs,
                                      kbt)
            print('Using the following priors:')
            [print(key) for key in prior_fns]
        else:
            print('Using no priors')
            prior_fns = None

        dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}

        init_fn, gnn_energy_fn = neural_networks.dimenetpp_neighborlist(
            displacement, r_cut, n_species, init_pos, nbrs_init,
            kbt_dependent=kbt_dependent, embed_size=32, init_kwargs=mlp_init,
            dropout_mode=dropout_mode
        )

        # needs to know positions to know shape for network init
        if isinstance(model_init_key, list):
            # ensemble of neural networks not needed together with dropout
            init_params = [init_fn(key, init_pos, neighbor=nbrs_init,
                                   species=species, **energy_kwargs)
                           for key in model_init_key]
        else:
            if dropout_init_seed is None:
                init_params = init_fn(model_init_key, init_pos,
                                      neighbor=nbrs_init,
                                      species=species, **energy_kwargs)
            else:
                dropout_init_key = random.PRNGKey(dropout_init_seed)
                init_params = init_fn(model_init_key, init_pos,
                                      neighbor=nbrs_init, species=species,
                                      dropout_key=dropout_init_key,
                                      **energy_kwargs)
                init_params = dropout.build_dropout_params(init_params,
                                                           dropout_init_key)

        # this pattern allows changing the energy parameters on-the-fly
        def energy_fn_template(energy_params):
            def energy_fn(pos, neighbor, **dynamic_kwargs):
                gnn_energy = gnn_energy_fn(energy_params, pos, neighbor,
                                           species=species, **dynamic_kwargs)

                prior_dynamic_kwargs = {
                    key: value for key, value in dynamic_kwargs.items()
                    if key != "dropout_key"
                }
                prior_energy = prior_potential(prior_fns, pos, neighbor,
                                               **prior_dynamic_kwargs)
                return gnn_energy + prior_energy
            return energy_fn

    else:
        raise ValueError('The model' + model + 'is not implemented.')

    return energy_fn_template, neighbor_fn, init_params, nbrs_init


def initialize_simulation(init_class, model, target_dict=None, x_vals=None,
                          model_init_key=random.PRNGKey(0),
                          simulation_init_key=random.PRNGKey(1),
                          fractional=True, integrator='Nose_Hoover',
                          wrapped=True, kbt_dependent=False,
                          prior_constants=None, prior_idxs=None,
                          dropout_init_seed=None,
                          expansion_formula=False, committee=None,
                          n_replicas=None, constraint_dict=None):
    box_tensor, scale_fn = custom_space.init_fractional_coordinates(
        init_class.box)
    r_inits = init_class.r_init

    if fractional:
        r_inits = scale_fn(r_inits)

    if n_replicas is not None:
        r_inits = jnp.tile(r_inits, (n_replicas, 1, 1))

    multi_trajectory = r_inits.ndim > 2
    init_pos = r_inits[0] if multi_trajectory else r_inits

    displacement, shift = space.periodic_general(
        box_tensor, fractional_coordinates=fractional, wrapped=wrapped)

    energy_kwargs = {}
    if kbt_dependent:
        # to allow init of kbt_embedding
        energy_kwargs['kT'] = init_class.kbt

    _energy_fn_template, neighbor_fn, _init_params, nbrs = select_model(
        model, init_pos, displacement, init_class.box, model_init_key, init_class.kbt,
        init_class.species, x_vals, fractional, kbt_dependent,
        prior_idxs=prior_idxs, prior_constants=prior_constants,
        dropout_init_seed=dropout_init_seed, constraint_dict=constraint_dict, **energy_kwargs
    )

    if committee is not None:
      energy_fn_template = traj_util.committee_energy_fn(_energy_fn_template, **committee)
      init_params = [_init_params]
    else:
      energy_fn_template = _energy_fn_template
      init_params = _init_params

    energy_fn_init = energy_fn_template(init_params)


    if target_dict is None:
        target_dict = {}
    compute_fns, targets = build_quantity_dict(
        init_pos, box_tensor, displacement, energy_fn_template, nbrs,
        target_dict, init_class, expansion_formula=expansion_formula)

    print(f"Selected integrator: {integrator}")

    # setup simulator
    if integrator == 'Fire':
      simulator_template = partial(minimize.fire_descent, shift_fn=shift)
    elif integrator == 'GDE':
      simulator_template = partial(minimize.gradient_descent, shift_fn=shift,
                                   step_size=5e-6)
    elif integrator == 'Nose_Hoover':
        simulator_template = partial(simulate.nvt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     chain_length=3, chain_steps=1,
                                     tau=50. * init_class.dt)
    elif integrator == 'Langevin':
        simulator_template = partial(custom_simulator.nvt_langevin, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     gamma=100.)
    elif integrator == 'NPT':
        barostat_kwargs = {
          'chain_steps': 1,
          'tau': 2000. * init_class.dt,
        }
        thermostat_kwargs = {
          'chain_steps': 1,
          'tau': 100. * init_class.dt,
        }

        print(f"Use barostat settings {barostat_kwargs} and thermostat settings {thermostat_kwargs}")

        simulator_template = partial(simulate.npt_nose_hoover, shift_fn=shift,
                                     dt=init_class.dt, kT=init_class.kbt,
                                     pressure=init_class.ref_press,
                                     barostat_kwargs=barostat_kwargs,
                                     thermostat_kwargs=thermostat_kwargs)
    elif integrator == 'NVE':
        simulator_template = partial(simulate.nve, shift_fn=shift,
                                     dt=init_class.dt)
    elif integrator == 'None':
        simulation_funs = (None, energy_fn_template, neighbor_fn)
        return None, init_params, simulation_funs, compute_fns, targets
    else:
        raise NotImplementedError('Integrator string not recognized!')

    init, _ = simulator_template(energy_fn_init)
    # init = jit(init)  # avoid throwing initialization NaN for debugging NaNs

    if integrator == 'NVE':
        init = partial(init, kT=init_class.kbt)
    # box only used in NPT: needs to be box tensor as 1D box leads to error as
    # box is erroneously mapped over N dimensions (usually only for eps, sigma)

    def init_sim_state(inputs):
        rng_key, pos = inputs
        nbrs_update = nbrs.update(pos)
        if integrator == 'Fire' or integrator == 'GDE':
          state = init(pos, mass=init_class.masses, neighbor=nbrs_update,
                       box=box_tensor, **energy_kwargs)
        else:
          state = init(rng_key, pos, mass=init_class.masses, neighbor=nbrs_update,
                       box=box_tensor, **energy_kwargs)
        return state, nbrs_update  # store together

    if multi_trajectory:
        # batch_size = 20  # initialize 10 simulations in parallel; avoid OOM
        n_inits = r_inits.shape[0]
        batch_size = n_inits
        init_keys = random.split(simulation_init_key, n_inits)
        batched_init_keys = chem_util.tree_vmap_split(init_keys, batch_size)
        batched_r_inits = chem_util.tree_vmap_split(r_inits, batch_size)
        bachted_sim_states = lax.map(vmap(init_sim_state),
                                     (batched_init_keys, batched_r_inits))
        sim_state = chem_util.tree_combine(bachted_sim_states)
    else:
        sim_state = init_sim_state((simulation_init_key, init_pos))

    if target_dict is None:
        target_dict = {}
    compute_fns, targets = build_quantity_dict(
        init_pos, box_tensor, displacement, energy_fn_template, nbrs,
        target_dict, init_class, expansion_formula=expansion_formula)

    simulation_funs = (simulator_template, energy_fn_template, neighbor_fn)
    return sim_state, init_params, simulation_funs, compute_fns, targets
