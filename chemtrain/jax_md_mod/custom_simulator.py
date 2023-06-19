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

from typing import Callable, TypeVar, Union, Tuple

from jax import lax, ops, random, tree_util, numpy as jnp
from jax_md import quantity, util, space, dataclasses
from jax_md.simulate import NVTNoseHooverState, NVEState, \
    SUZUKI_YOSHIDA_WEIGHTS, velocity_verlet

from Legacy_files import adjoint_ode as ode

static_cast = util.static_cast


# Types
Array = util.Array
f32 = util.f32
f64 = util.f64

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]

Schedule = Union[Callable[..., float], float]


def nvt_nose_hoover_gradient_stop(energy_or_force: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    kT: float,
                    chain_length: int=5,
                    chain_steps: int=2,
                    sy_steps: int=3,
                    tau: float=None,
                    stop_ratio: float=0.01) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [1, 2, 3]. We employ a similar
  notation to [2] and the interested reader might want to look at that paper as
  a reference.

  As described in [3], the NHC evolves on a faster timescale than the rest of
  the simulation. Therefore, it often desirable to integrate the chain over
  several substeps for each step of MD. To do this we follow the Suzuki-Yoshida
  scheme. Specifically, we subdivide our chain simulation into $n_c$ substeps.
  These substeps are further subdivided into $n_sy$ steps. Each $n_sy$ step has
  length $\delta_i = \Delta t w_i / n_c$ where $w_i$ are constants such that
  $\sum_i w_i = 1$. See the table of Suzuki_Yoshida weights above for specific
  values.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, $n_c$, of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either 1, 3, 5, or 7.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of dt. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  [1] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
      "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
      The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  [2] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
      "Explicit reversible integrators for extended systems dynamics."
      Molecular Physics 87. (1998) 1117-1157.
  [3] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
      Andrea L. Jochim, and Glenn J. Martyna.
      "A Liouville-operator derived measure-preserving integrator for molecular
      dynamics simulations in the isothermal-isobaric ensemble."
      Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt = f32(dt)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)
  dt_2 = dt / f32(2.0)

  kT = f32(kT)

  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    mass = quantity.canonicalize_mass(mass)
    V = jnp.sqrt(_kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - jnp.mean(V, axis=0, keepdims=True)
    KE = quantity.kinetic_energy(V, mass)

    # Nose-Hoover parameters.
    xi = jnp.zeros(chain_length, R.dtype)
    v_xi = jnp.zeros(chain_length, R.dtype)

    # TODO(schsam): Really, it seems like Q should be set by the goal
    # temperature rather than the initial temperature.
    DOF = f32(R.shape[0] * R.shape[1])
    Q = _kT * tau ** f32(2) * jnp.ones(chain_length, dtype=R.dtype)
    Q = ops.index_update(Q, 0, Q[0] * DOF)

    F = force_fn(R, **kwargs)

    return NVTNoseHooverState(R, V, F, mass, KE, xi, v_xi, Q)  # pytype: disable=wrong-arg-count

  def substep_chain_fn(delta, KE, V, xi, v_xi, Q, DOF, T):
    """Applies a single update to the chain parameters and rescales velocity."""
    delta_2 = delta   / f32(2.0)
    delta_4 = delta_2 / f32(2.0)
    delta_8 = delta_4 / f32(2.0)

    M = chain_length - 1

    G = (Q[M - 1] * v_xi[M - 1] ** f32(2) - T) / Q[M]
    v_xi = ops.index_add(v_xi, M, delta_4 * G)

    def backward_loop_fn(v_xi_new, m):
      G = (Q[m - 1] * v_xi[m - 1] ** 2 - T) / Q[m]
      scale = jnp.exp(-delta_8 * v_xi_new)
      v_xi_new = scale * (scale * v_xi[m] + delta_4 * G)
      return v_xi_new, v_xi_new
    idx = jnp.arange(M - 1, 0, -1)
    _, v_xi_update = lax.scan(backward_loop_fn, v_xi[M], idx, unroll=2)
    v_xi = ops.index_update(v_xi, idx, v_xi_update)

    G = (f32(2.0) * KE - DOF * T) / Q[0]
    scale = jnp.exp(-delta_8 * v_xi[1])
    v_xi = ops.index_update(v_xi, 0, scale * (scale * v_xi[0] + delta_4 * G))

    scale = jnp.exp(-delta_2 * v_xi[0])
    KE = KE * scale ** f32(2)
    V = V * scale

    xi = xi + delta_2 * v_xi

    G = (f32(2) * KE - DOF * T) / Q[0]
    def forward_loop_fn(G, m):
      scale = jnp.exp(-delta_8 * v_xi[m + 1])
      v_xi_update = scale * (scale * v_xi[m] + delta_4 * G)
      G = (Q[m] * v_xi_update ** f32(2) - T) / Q[m + 1]
      return G, v_xi_update
    idx = jnp.arange(M)
    G, v_xi_update = lax.scan(forward_loop_fn, G, idx, unroll=2)
    v_xi = ops.index_update(v_xi, idx, v_xi_update)
    v_xi = ops.index_add(v_xi, M, delta_4 * G)

    return KE, V, xi, v_xi, Q, DOF, T

  def half_step_chain_fn(*chain_state):
    if chain_steps == 1 and sy_steps == 1:
      return substep_chain_fn(dt, *chain_state)

    delta = dt / chain_steps
    ws = jnp.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps], dtype=chain_state[1].dtype)
    return lax.scan(lambda chain_state, i:
                    (substep_chain_fn(delta * ws[i % sy_steps], *chain_state),
                     0),
                    chain_state,
                    jnp.arange(chain_steps * sy_steps))[0]

  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    R, V, F, mass, KE, xi, v_xi, Q = dataclasses.astuple(state)

    DOF = R.size

    Q = _kT * tau ** f32(2) * jnp.ones(chain_length, dtype=R.dtype)
    Q = ops.index_update(Q, 0, Q[0] * DOF)

    KE, V, xi, v_xi, *_ = half_step_chain_fn(KE, V, xi, v_xi, Q, DOF, _kT)

    R = shift_fn(R, V * dt + F * dt ** 2 / (2 * mass), **kwargs)

    F_new = force_fn(R, **kwargs)

    V = V + dt_2 * (F_new + F) / mass

    V = V - jnp.mean(V, axis=0, keepdims=True)
    KE = quantity.kinetic_energy(V, mass)

    KE, V, xi, v_xi, *_ = half_step_chain_fn(KE, V, xi, v_xi, Q, DOF, _kT)

    return NVTNoseHooverState(R, V, F_new, mass, KE, xi, v_xi, Q)

  def gradient_stop(state, **kwargs):
    R, V, F, mass, KE, xi, v_xi, Q = dataclasses.astuple(state)
    non_stop = 1. - stop_ratio

    R_new = non_stop * R + stop_ratio * lax.stop_gradient(R)
    V_new = non_stop * V + stop_ratio * lax.stop_gradient(V)
    F_new = non_stop * F + stop_ratio * lax.stop_gradient(F)
    KE_new = non_stop * KE + stop_ratio * lax.stop_gradient(KE)
    xi_new = non_stop * xi + stop_ratio * lax.stop_gradient(xi)
    v_xi_new = non_stop * v_xi + stop_ratio * lax.stop_gradient(v_xi)
    Q_new = non_stop * Q + stop_ratio * lax.stop_gradient(Q)  # is constant for T=const --> in this case gradient stop not necessary, but just changes nothing
    return NVTNoseHooverState(R_new, V_new, F_new, mass, KE_new, xi_new, v_xi_new, Q_new)
  return init_fn, apply_fn, gradient_stop


def gradient_stop(state, stop_ratio):
    non_stop = 1. - stop_ratio
    stop_component = lambda x: non_stop * x + stop_ratio * lax.stop_gradient(x)
    stopped_state = tree_util.tree_map(stop_component, state)
    return stopped_state


def nve_gradstop(energy_or_force_fn: Callable[..., Array],
        shift_fn: ShiftFn,
        dt: float,
        stopratio: float) -> Simulator:
  """Simulates a system in the NVE ensemble.

  Samples from the microcanonical ensemble in which the number of particles
  (N), the system volume (V), and the energy (E) are held constant. We use a
  standard velocity verlet integration scheme.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
    shift_fn: A function that displaces positions, R, by an amount dR. Both R
      and dR should be ndarrays of shape [n, spatial_dimension].
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    quant: Either a quantity.Energy or a quantity.Force specifying whether
      energy_or_force is an energy or force respectively.
  Returns:
    See above.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
    mass = quantity.canonicalize_mass(mass)
    V = jnp.sqrt(kT / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - jnp.mean(V, axis=0, keepdims=True)
    return NVEState(R, V, force_fn(R, **kwargs), mass)  # pytype: disable=wrong-arg-count

  def step_fn(state, **kwargs):
    # TODO reduce copy-paste code by implementing gradient-stop as a
    #  wrapper around step_fn
    stepped = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)
    stopped_state = gradient_stop(stepped, stopratio)
    return stopped_state

  return init_fn, step_fn


@dataclasses.dataclass
class NVEIntegrationState:
  """A struct containing the state of an NVE simulation for integration.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    position: An ndarray of shape [n, spatial_dimension] storing the position
      of particles.
    velocity: An ndarray of shape [n, spatial_dimension] storing the velocity
      of particles.
  """
  position: Array
  velocity: Array

@dataclasses.dataclass
class NVEDynamicsState:
  """A struct containing the state of an NVE simulation for integration.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    Velocity: An ndarray of shape [n, spatial_dimension] storing the velocity particles (position prime).
    Acceleration: An ndarray of shape [n, spatial_dimension] storing the acceleration of particles (velocity prime).
  """
  velocity: Array
  acceleration: Array

# pylint: disable=invalid-name
def nve_adjoint(energy_or_force: Callable[..., Array],
        stop_ratio: float=0.02, wrap_output: Callable[..., Array]=None) -> Simulator:
  """Simulates a system in the NVE ensemble. Gradients are memory-efficiently computed via the adjoint method.

  Samples from the microcanonical ensemble in which the number of particles (N),
  the system volume (V), and the energy (E) are held constant.
  The integration uses the adaptive RK scheme as implemented in jax.experimental.ode.
  Currently only works for un-wrapped configurations: Allows unconstrained black-box time integration

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      [n, spatial_dimension].
      wrap_output: If re-mapping function given, maps unwrapped output back to original box
  Returns:
    See above.
  """


  def init_fun(key: Array,
               R: Array,
               T_initial: float=f32(1.0),
               mass=f32(1.0),
               **kwargs) -> (NVEIntegrationState, Callable):

    mass = quantity.canonicalize_mass(mass)
    V = jnp.sqrt(T_initial / mass) * random.normal(key, R.shape, dtype=R.dtype)
    V = V - jnp.mean(V, axis=0, keepdims=True)  # subtract mean velocity to avoid overall drift

    def nve_dynamics(state, t: float, args, **kwargs):
      """Computes derivative of state variables for use in time integration scheme.
          Needs to depend on params wrt. which derivative should be computed via adjoint
           --> need to build energy function here
           TODO: This does not seem ideal! Is there a better way to prebuild function outside but keep dependence on *args?"""

      pre_built_energy_or_force = energy_or_force(args)
      force = quantity.canonicalize_force(pre_built_energy_or_force)

      R, V = dataclasses.astuple(state)
      A = force(R, t=t, **kwargs) / mass
      return NVEDynamicsState(V, A)

    return NVEIntegrationState(R, V), nve_dynamics  # pytype: disable=wrong-arg-count


  def integrate(state: NVEIntegrationState, dynamics, ts: Array, *args, rtol=1.e-4):
    trajectory = ode.odeint(dynamics, state, ts, *args, rtol=rtol)  # rtol significantly impacts runtime (and integration quality)

    # extract last state for easy continuation of integration
    last_state = NVEIntegrationState(trajectory.position[-1], trajectory.velocity[-1])

    if not wrap_output is None:
      # call function on trajectory.position
      pass  # TODO wrap snapshots back in original simulation box

    return trajectory, last_state

  return init_fun, integrate
