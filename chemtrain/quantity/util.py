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

"""Create quantitiy targets for training. """
import functools

from jax_md.partition import NeighborList

from typing import TypedDict, Callable, Dict, Any, Tuple, List
from chemtrain.typing import ArrayLike, ComputeFn, EnergyFnTemplate, TargetDict

InitReturn = Tuple[Dict[str, TargetDict], Dict[str, ComputeFn]]
ComputeFnInit = Callable[..., ComputeFn]
ComputeFnInitDict = Dict[str, Callable[..., ComputeFnInit]]
ComputeFnDict = Dict[str, ComputeFn]
TargetInit = Callable[..., Tuple[TargetDict, ComputeFnDict]]


class InitArguments(TypedDict, total=False):
    """Non-exhaustive dictionary of arguments to initialize the compute functions. """
    displacement_fn: Callable
    energy_fn_template: EnergyFnTemplate
    nbrs_init: NeighborList
    reference_box: ArrayLike
    ref_box_tensor: ArrayLike
    r_init: ArrayLike


class TargetBuilder:

    def __init__(self):
        self._targets = {}

    def __setitem__(self, key, value: TargetInit):
        assert key not in self._targets, (
            f"Duplicate target {key}."
        )
        self._targets[key] = value

    def build(self, system: InitArguments):
        target_dicts = {}
        compute_fns = {}
        for key, init_fn in self._targets.items():
            target_dicts, compute_fns = init_fn(
                key, target_dicts, compute_fns, system)
        return target_dicts, compute_fns


# def set_init_kwargs(init_fn: ComputeFnInit,
#                     required: List[str],
#                     optional: List[str],
#                     **set_kwargs: Any):
#     """Helper function initializing the compute functions.
#
#     Known arguments are set as keyword arguments while arguments that are
#     required but now yet available are marked:
#
#     .. code-block ::
#
#         # The compute function requires the indicies of the bonds given by the
#         # bond dihedral params.
#         dihedral_params = custom_quantity.BondDihedralParams(bond_idxs, ...)
#
#         compute_fn_init = set_init_kwargs(
#             custom_quantity.init_bond_dihedral_distribution,
#             'displacement_fn', bond_dihedral_params=dihedral_struct
#         )
#
#         # The final compute functions can be initialized after the simulation
#         # has been setup.
#
#         compute_fn_templates = {
#             'dihedral_dist' compute_fn_init,
#             ...
#         }
#
#         init_args = {
#             'displacement_fn': ...
#         }
#
#         compute_fns = initialize_compute_fns(
#             compute_fn_templates, init_args)
#
#     Args:
#         init_fn: Function initializing the compute function.
#         required: List of keyword arguments required for the initialization
#             that are not yet available.
#         set_kwargs: Keyword arguments that are known and can be set directly.
#
#     Returns:
#         Returns a partially initialized compute function.
#
#     """
#     @functools.wraps(init_fn)
#     def partial_init_fn(**kwargs):
#         filtered_kwargs = {
#             key: kwargs[key] for key in required
#         }
#         # Collect all optional args
#         filtered_optional_kwargs = {
#             key: value for key in optional
#             if (value := kwargs.get(key)) is not None
#         }
#         filtered_kwargs.update(filtered_optional_kwargs)
#         return init_fn(**filtered_kwargs, **set_kwargs)
#     return partial_init_fn

def target_quantity(required: list = None, optional: list = None):
    if required is None:
        required = []
    if optional is None:
        optional = []

    def quantity_init_wrapper(init_fn):
        @functools.wraps(init_fn)
        def quantity_init(key: str,
                          targets_dict: TargetDict,
                          compute_fns: ComputeFnInitDict,
                          init_args: InitArguments
                          ) -> Tuple[TargetDict,  ComputeFnDict]:
            # Assert that all required arguments are provided and collect all
            # optional arguments
            init_kwargs = {
                init_key: init_args[init_key] for init_key in required
            }
            optional_kwargs = {
                init_key: value for init_key in optional
                if (value := init_args.get(init_key)) is not None
            }
            init_kwargs.update(optional_kwargs)

            # Initialize the compute function and target function with the
            # provided arguments
            target_dict, compute_fn = init_fn(
                key, compute_fns, init_kwargs)

            # Add the initialized compute function and target dict
            if target_dict is not None:
                assert key not in target_dict.keys(), (
                    f"Duplicate target {key} provided.")
                targets_dict.update({key: target_dict})
            if compute_fn is not None:
                assert key not in compute_fns.keys(), (
                    f"Duplicate quantitiy {key} initialized.")
                compute_fns.update({key: compute_fn})

            return targets_dict, compute_fns
        return quantity_init

    return quantity_init_wrapper
