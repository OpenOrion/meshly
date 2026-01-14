"""Pytest configuration and shared fixtures for meshly tests."""

import numpy as np
import pytest


@pytest.fixture
def cube_vertices():
    """Vertices for a unit cube mesh."""
    return np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)


@pytest.fixture
def cube_indices():
    """Triangle indices for a cube mesh."""
    return np.array([
        0, 1, 2, 2, 3, 0,  # front
        1, 5, 6, 6, 2, 1,  # right
        5, 4, 7, 7, 6, 5,  # back
        4, 0, 3, 3, 7, 4,  # left
        3, 2, 6, 6, 7, 3,  # top
        4, 5, 1, 1, 0, 4   # bottom
    ], dtype=np.uint32)


@pytest.fixture
def triangle_vertices():
    """Vertices for a simple triangle."""
    return np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)


@pytest.fixture
def triangle_indices():
    """Indices for a simple triangle."""
    return np.array([0, 1, 2], dtype=np.uint32)


@pytest.fixture
def simple_mesh_vertices():
    """Vertices for simple mesh tests."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0]
    ], dtype=np.float32)
