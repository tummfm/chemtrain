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

import sys

from jax import Array
import jax.numpy as jnp
import jax_md.partition

from . import (
  custom_energy,
  custom_interpolate,
  custom_quantity,
  custom_simulator,
  custom_space,
  io
)

def uncache(exclude):
  """Remove package modules from cache except excluded ones.
  On next import they will be reloaded.

  Args:
      exclude (iter<str>): Sequence of module paths.
  """
  pkgs = []
  for mod in exclude:
    pkg = mod.split('.', 1)[0]
    pkgs.append(pkg)

  to_uncache = []
  for mod in sys.modules:
    if mod in exclude:
      continue

    if mod in pkgs:
      to_uncache.append(mod)
      continue

    for pkg in pkgs:
      if mod.startswith(pkg + '.'):
        to_uncache.append(mod)
        break

  for mod in to_uncache:
    del sys.modules[mod]


def is_box_valid(box: Array) -> bool:
  print(f"Used patched box check")
  if jnp.isscalar(box) or box.ndim == 0 or box.ndim == 1:
    return True
  if box.ndim == 2:
    return jnp.bool_(jnp.all(jnp.triu(box) == box))
  return False

jax_md.partition.is_box_valid = is_box_valid
uncache('jax_md.partition')
