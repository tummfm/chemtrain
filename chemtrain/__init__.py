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

import warnings

import jax


def _warn_if_default_matmul_precision_unset() -> None:
    """Warn when JAX matmul precision falls back to the backend default."""
    if jax.config.jax_default_matmul_precision is not None:
        return
    warnings.warn(
        "JAX default matmul precision is not set. For float32 model training, "
        "evaluation, and deployment export, consider setting "
        "JAX_DEFAULT_MATMUL_PRECISION=highest or "
        "jax.config.update('jax_default_matmul_precision', 'highest') for more "
        "reproducible matmul/contraction behavior.",
        RuntimeWarning,
        stacklevel=2,
    )


_warn_if_default_matmul_precision_unset()

# Applies patches
import jax_md_mod

# Minimal global config
from . import config  # noqa: F401
