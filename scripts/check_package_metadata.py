# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Checks publishable dependency metadata in a built chemtrain wheel."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import zipfile

from packaging.requirements import Requirement


JAX_SPECIFIER = "!=0.10.2,<0.12,>=0.5.0"
JAX_CUDA13_SPECIFIER = "!=0.10.2,<0.12,>=0.7.0"


def read_metadata(wheel: Path):
    """Returns parsed core metadata from one wheel archive."""
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"Expected one METADATA file in {wheel}, found {metadata_files}"
            )
        return Parser().parsestr(
            archive.read(metadata_files[0]).decode("utf-8")
        )


def main() -> None:
    """Validates base and CUDA-extra JAX requirements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    metadata = read_metadata(parser.parse_args().wheel)
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]

    expected = {
        ("jax", (), "", JAX_SPECIFIER),
        ("jax", ("cuda12",), 'extra == "cuda12"', JAX_SPECIFIER),
        (
            "jax",
            ("cuda13",),
            'extra == "cuda13"',
            JAX_CUDA13_SPECIFIER,
        ),
    }
    actual = {
        (
            requirement.name,
            tuple(sorted(requirement.extras)),
            str(requirement.marker or ""),
            str(requirement.specifier),
        )
        for requirement in requirements
        if requirement.name == "jax"
    }
    if actual != expected:
        raise RuntimeError(
            f"Wheel metadata has unexpected JAX requirements. "
            f"Expected {sorted(expected)}, found {sorted(actual)}"
        )

    jax_md_requirements = [
        requirement
        for requirement in requirements
        if requirement.name == "jax-md"
    ]
    if len(jax_md_requirements) != 1 or str(
        jax_md_requirements[0].specifier
    ) != ">=0.2.29":
        raise RuntimeError(
            "Wheel metadata must contain exactly one jax-md>=0.2.29 "
            f"requirement, found {jax_md_requirements}"
        )

    direct_references = [
        str(requirement)
        for requirement in requirements
        if requirement.url is not None
    ]
    if direct_references:
        raise RuntimeError(
            "PyPI distributions must not contain direct dependency "
            f"references: {direct_references}"
        )


if __name__ == "__main__":
    main()
