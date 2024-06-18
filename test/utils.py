import os

import pytest


@pytest.fixture
def change_dir(tmp_path):
    # Remember the current working directory
    old_dir = os.getcwd()
    # Change to the temporary directory
    os.chdir(tmp_path)
    yield
    # Change back to the original directory
    os.chdir(old_dir)
