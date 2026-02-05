"""
Tests for JAX array support in meshly.
"""

import pytest
import numpy as np
from io import BytesIO
from meshly import Mesh, Array
from meshly.array import HAS_JAX


class TestJAXSupport:
    """Test JAX array support functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.indices = np.array([0, 1, 2], dtype=np.uint32)

    def test_array_type_definition(self):
        """Test that Array type is properly defined."""
        assert Array is not None

        # Test that numpy arrays are compatible with Array type
        np_array = np.array([1, 2, 3])
        assert isinstance(np_array, np.ndarray)

    def test_has_jax_flag(self):
        """Test that HAS_JAX flag is properly set."""
        assert isinstance(HAS_JAX, bool)

    def test_numpy_functionality_preserved(self):
        """Test that existing numpy functionality still works."""
        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Verify arrays are numpy arrays
        assert isinstance(mesh.vertices, np.ndarray)
        assert isinstance(mesh.indices, np.ndarray)

        # Test basic properties
        assert mesh.vertex_count == 3
        assert mesh.index_count == 3

        # Test encoding/decoding via zip round-trip
        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded = Mesh.load_from_zip(buffer, array_type="numpy")

        assert isinstance(decoded.vertices, np.ndarray)
        assert isinstance(decoded.indices, np.ndarray)
        np.testing.assert_array_equal(decoded.vertices, self.vertices)
        np.testing.assert_array_equal(decoded.indices, self.indices)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_functionality(self):
        """Test JAX functionality when available."""
        import jax.numpy as jnp

        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Test encoding/decoding with JAX via zip round-trip
        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_jax = Mesh.load_from_zip(buffer, array_type="jax")

        # Verify vertices are JAX arrays
        assert hasattr(decoded_jax.vertices, 'device'), "Vertices should be JAX arrays"
        # Indices stay as numpy for meshoptimizer compatibility
        assert isinstance(decoded_jax.indices, np.ndarray), "Indices are numpy for meshoptimizer"

        # Verify data is preserved
        np.testing.assert_array_equal(np.array(decoded_jax.vertices), self.vertices)
        np.testing.assert_array_equal(np.array(decoded_jax.indices), self.indices)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_jax_input_arrays(self):
        """Test using JAX arrays as input."""
        import jax.numpy as jnp

        # Create JAX arrays
        jax_vertices = jnp.array(self.vertices)
        jax_indices = jnp.array(self.indices)

        # Create mesh with JAX arrays
        mesh = Mesh(vertices=jax_vertices, indices=jax_indices)

        # Vertices should remain JAX, indices converted to numpy for meshoptimizer
        assert hasattr(mesh.vertices, 'device'), "Vertices should remain JAX arrays"
        # Converted for meshoptimizer compatibility
        assert isinstance(mesh.indices, np.ndarray)

    @pytest.mark.skipif(HAS_JAX, reason="JAX is available, cannot test unavailable scenario")
    def test_jax_unavailable_error(self):
        """Test error handling when JAX is requested but unavailable."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)

        with pytest.raises(AssertionError, match="JAX is not available"):
            Mesh.load_from_zip(buffer, array_type="jax")

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_mesh_copy_with_jax_arrays(self):
        """Test mesh copying with JAX arrays."""
        import jax.numpy as jnp

        jax_vertices = jnp.array(self.vertices)
        mesh = Mesh(vertices=jax_vertices, indices=self.indices)

        copied_mesh = mesh.model_copy(deep=True)

        assert hasattr(copied_mesh.vertices, 'device'), "Copied vertices should be JAX arrays"
        np.testing.assert_array_equal(np.array(copied_mesh.vertices), self.vertices)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_additional_arrays_jax_support(self):
        """Test that additional arrays also support JAX conversion."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Optional

        class CustomMesh(Mesh):
            normals: Optional[Array] = Field(None, description="Normal vectors")

        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = CustomMesh(vertices=self.vertices, indices=self.indices, normals=normals)

        buffer = BytesIO()
        mesh.save_to_zip(buffer)
        buffer.seek(0)
        decoded_jax = CustomMesh.load_from_zip(buffer, array_type="jax")

        assert hasattr(decoded_jax.vertices, 'device'), "Vertices should be JAX arrays"
        assert hasattr(decoded_jax.normals, 'device'), "Normals should be JAX arrays"

        np.testing.assert_array_equal(np.array(decoded_jax.normals), normals)
