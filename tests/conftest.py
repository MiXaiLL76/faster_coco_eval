"""Shared pytest fixtures for deterministic test execution."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def seed_numpy_per_test():
    """Reset NumPy's legacy global generator before each test."""
    np.random.seed(0)
