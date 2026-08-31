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

import os
from pathlib import Path

import jax
import pytest


def pytest_collection_modifyitems(items):
    """Skip marked tests when the requested JAX devices are unavailable."""
    available_devices = jax.device_count()
    for item in items:
        marker = item.get_closest_marker("jax_multidevice")
        if marker is None:
            continue
        required_devices = marker.kwargs.get("devices", 2)
        if available_devices < required_devices:
            item.add_marker(pytest.mark.skip(
                reason=(
                    f"Test requires {required_devices} JAX devices; "
                    f"only {available_devices} available."
                )
            ))


@pytest.fixture(scope="module")
def datafiles(request):
    """Returns the corresponding datafiles folder."""
    # Path where pytest is executed can be overridden
    base_path = os.environ.get("PYTEST_PATH", ".")
    base_path = Path(base_path).absolute() / "tests"

    rel_path = Path(request.fspath).relative_to(base_path)
    data_path = base_path / "data" / rel_path.parent / rel_path.stem

    return data_path
