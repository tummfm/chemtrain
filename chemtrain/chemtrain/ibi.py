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

"""Implementation of Iterative Boltzmann Inversion (IBI) update step and
estimate of potential of mean force (PMF).
"""
from jax import numpy as jnp
import numpy as onp


def initial_guess(rdf_target, kbt):
    """Compute IBI initial guess via potential of mean force (PMF) from
    target RDF g(r).

    U(r) = -kbT * log(g(r))
    """
    u_vals_initial = onp.asarray(-kbt) * onp.log(rdf_target)  # PMF

    # to avoid inf where rdf is 0:
    # keep force = potential derivative constant from last non-inf point onwards
    # we assume that infs are all grouped together at the left side of the PMF
    u_vals_initial[onp.isinf(u_vals_initial)] = onp.nan
    i_max = onp.nanargmax(u_vals_initial)  # last non-inf value
    max_difference = u_vals_initial[i_max] - u_vals_initial[i_max + 1]
    correction_array = onp.flip(onp.arange(i_max) + 1.)
    correction_array = correction_array * max_difference + u_vals_initial[i_max]
    u_vals_initial[:i_max] = correction_array
    return u_vals_initial


def update_potential(cur_rdf, target_rdf, u_vals, kbt):
    """IBI update step.

    U_new = U_old + kbT * log(g(r) / g_target(r))
    Zeros are handles by casting them to a small number, avioding inf/nan.
    """
    epsilon = 1.e-7
    target_safe = jnp.where(target_rdf < epsilon, epsilon, target_rdf)
    cur_safe = jnp.where(cur_rdf < epsilon, epsilon, cur_rdf)
    delta_u = kbt * jnp.log(cur_safe / target_safe)
    return u_vals + delta_u
