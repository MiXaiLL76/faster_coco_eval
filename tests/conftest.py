"""Shared pytest fixtures and fixture paths for deterministic test execution."""

from pathlib import Path

import numpy as np
import pytest

TESTS_DIR = Path(__file__).parent


@pytest.fixture
def datadir() -> Path:
    """Return the repository-controlled directory containing test fixtures."""
    return TESTS_DIR / "dataset"


@pytest.fixture(autouse=True)
def seed_numpy_per_test():
    """Reset NumPy's legacy global generator before each test."""
    np.random.seed(0)
