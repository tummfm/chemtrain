import sys

from jax import Array
import jax.numpy as jnp
import jax_md.partition

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
