"""
Tests for array type conversion functions (convert_to).
"""

import pytest
import numpy as np
from io import BytesIO
from meshly import Mesh, Array
from meshly.array import HAS_JAX


class TestConversion:
    """Test array type conversion functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.indices = np.array([0, 1, 2], dtype=np.uint32)

    def test_convert_to_numpy(self):
        """Test converting mesh to NumPy arrays."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        numpy_mesh = mesh.convert_to("numpy")

        assert isinstance(numpy_mesh.vertices, np.ndarray)
        assert isinstance(numpy_mesh.indices, np.ndarray)
        np.testing.assert_array_equal(numpy_mesh.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh.indices, self.indices)
        assert numpy_mesh is not mesh

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_convert_to_jax(self):
        """Test converting mesh to JAX arrays."""
        import jax.numpy as jnp

        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        jax_mesh = mesh.convert_to("jax")

        assert hasattr(jax_mesh.vertices, 'device'), "Vertices should be JAX arrays"
        assert hasattr(jax_mesh.indices, 'device'), "Indices should be JAX arrays"
        np.testing.assert_array_equal(np.array(jax_mesh.vertices), self.vertices)
        np.testing.assert_array_equal(np.array(jax_mesh.indices), self.indices)
        assert jax_mesh is not mesh
        assert isinstance(mesh.vertices, np.ndarray)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_bidirectional_conversion(self):
        """Test converting between NumPy and JAX arrays."""
        import jax.numpy as jnp

        numpy_mesh = Mesh(vertices=self.vertices, indices=self.indices)
        jax_mesh = numpy_mesh.convert_to("jax")
        assert hasattr(jax_mesh.vertices, 'device')

        numpy_mesh2 = jax_mesh.convert_to("numpy")
        assert isinstance(numpy_mesh2.vertices, np.ndarray)
        assert isinstance(numpy_mesh2.indices, np.ndarray)

        np.testing.assert_array_equal(numpy_mesh2.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh2.indices, self.indices)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_convert_to_jax_with_custom_fields(self):
        """Test converting custom mesh class to JAX."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Optional

        class CustomMesh(Mesh):
            normals: Optional[Array] = Field(None, description="Normal vectors")

        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = CustomMesh(vertices=self.vertices, indices=self.indices, normals=normals)

        jax_mesh = mesh.convert_to("jax")

        assert hasattr(jax_mesh.vertices, 'device')
        assert hasattr(jax_mesh.normals, 'device')
        np.testing.assert_array_equal(np.array(jax_mesh.normals), normals)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_convert_to_numpy_with_nested_arrays(self):
        """Test converting mesh with nested dictionary arrays to NumPy."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Dict, Any, Optional

        class CustomMesh(Mesh):
            materials: Optional[Dict[str, Any]] = Field(None, description="Material properties")

        materials = {
            'diffuse': jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            'properties': {
                'roughness': jnp.array([0.5], dtype=jnp.float32),
            }
        }

        jax_mesh = CustomMesh(
            vertices=jnp.array(self.vertices),
            indices=jnp.array(self.indices),
            materials=materials
        )

        numpy_mesh = jax_mesh.convert_to("numpy")

        assert isinstance(numpy_mesh.materials['diffuse'], np.ndarray)
        assert isinstance(numpy_mesh.materials['properties']['roughness'], np.ndarray)

    @pytest.mark.skipif(HAS_JAX, reason="JAX is available, cannot test unavailable scenario")
    def test_convert_to_jax_without_jax_raises_error(self):
        """Test that convert_to("jax") raises error when JAX is unavailable."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        with pytest.raises(AssertionError, match="JAX is not available"):
            mesh.convert_to("jax")
