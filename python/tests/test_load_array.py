"""Tests for Packable.load_array functionality."""

import pytest
import numpy as np
from io import BytesIO
from typing import Optional

from meshly import Mesh
from meshly.packable import Packable


class NormalsMesh(Mesh):
    """Custom mesh with normals for testing."""
    normals: Optional[np.ndarray] = None


class TestLoadArray:
    """Test cases for load_array method."""

    def test_load_single_array(self):
        """Test loading a single array from a zip file."""
        mesh = NormalsMesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        )

        buf = BytesIO()
        mesh.save_to_zip(buf)
        buf.seek(0)

        normals = NormalsMesh.load_array(buf, 'normals')

        assert isinstance(normals, np.ndarray)
        assert normals.shape == (3, 3)
        np.testing.assert_array_equal(normals, mesh.normals)

    def test_load_nested_array(self):
        """Test loading nested dictionary arrays using dotted notation."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32),
            markers={'inlet': [[0, 1]], 'outlet': [[2, 3]]}
        )

        buf = BytesIO()
        mesh.save_to_zip(buf)
        buf.seek(0)

        # Load nested marker array
        inlet = Mesh.load_array(buf, 'markers.inlet')

        assert isinstance(inlet, np.ndarray)
        np.testing.assert_array_equal(inlet, np.array([0, 1]))

        buf.seek(0)

        # Load marker sizes
        inlet_sizes = Mesh.load_array(buf, 'marker_sizes.inlet')
        np.testing.assert_array_equal(inlet_sizes, np.array([2]))

    def test_load_array_not_found(self):
        """Test that loading a non-existent array raises KeyError."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        buf = BytesIO()
        mesh.save_to_zip(buf)
        buf.seek(0)

        with pytest.raises(KeyError, match='nonexistent'):
            Mesh.load_array(buf, 'nonexistent')

    def test_load_builtin_array(self):
        """Test loading built-in arrays like index_sizes."""
        mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
        )

        buf = BytesIO()
        mesh.save_to_zip(buf)
        buf.seek(0)

        # Load index_sizes (auto-generated)
        index_sizes = Mesh.load_array(buf, 'index_sizes')

        assert isinstance(index_sizes, np.ndarray)
        np.testing.assert_array_equal(index_sizes, np.array([3]))

    def test_load_array_preserves_dtype(self):
        """Test that load_array preserves the original array dtype."""
        mesh = NormalsMesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint32),
            normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        )

        buf = BytesIO()
        mesh.save_to_zip(buf)
        buf.seek(0)

        normals = NormalsMesh.load_array(buf, 'normals')
        assert normals.dtype == np.float32

        buf.seek(0)
        index_sizes = Mesh.load_array(buf, 'index_sizes')
        assert index_sizes.dtype == np.uint32
