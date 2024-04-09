import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def datafiles(request):
    """Returns the corresponding datafiles folder."""
    # Path where pytest is executed can be overridden
    base_path = os.environ.get("PYTEST_PATH", ".")
    base_path = Path(base_path).absolute() / "tests"

    rel_path = Path(request.fspath).relative_to(base_path)
    data_path = base_path / "data" / rel_path.parent / rel_path.stem

    return data_path
