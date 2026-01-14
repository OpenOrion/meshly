"""
Tests for array type conversion functions (convert_to).
"""

import unittest
import numpy as np
from meshly import Mesh, Array
from meshly.array import HAS_JAX


class TestConversion(unittest.TestCase):
    """Test array type conversion functionality."""

    def setUp(self):
        """Set up test data."""
        self.vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.indices = np.array([0, 1, 2], dtype=np.uint32)

    def test_convert_to_numpy(self):
        """Test converting mesh to NumPy arrays."""
        # Start with NumPy mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Convert to NumPy (should be no-op but create new instance)
        numpy_mesh = mesh.convert_to("numpy")

        self.assertIsInstance(numpy_mesh.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh.indices, np.ndarray)
        np.testing.assert_array_equal(numpy_mesh.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh.indices, self.indices)

        # Verify it's a different instance
        self.assertIsNot(numpy_mesh, mesh)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_to_jax(self):
        """Test converting mesh to JAX arrays."""
        import jax.numpy as jnp

        # Start with NumPy mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Convert to JAX
        jax_mesh = mesh.convert_to("jax")

        self.assertTrue(hasattr(jax_mesh.vertices, 'device'),
                        "Vertices should be JAX arrays")
        self.assertTrue(hasattr(jax_mesh.indices, 'device'),
                        "Indices should be JAX arrays")
        np.testing.assert_array_equal(
            np.array(jax_mesh.vertices), self.vertices)
        np.testing.assert_array_equal(np.array(jax_mesh.indices), self.indices)

        # Verify it's a different instance
        self.assertIsNot(jax_mesh, mesh)

        # Original mesh should still have NumPy arrays
        self.assertIsInstance(mesh.vertices, np.ndarray)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_bidirectional_conversion(self):
        """Test converting between NumPy and JAX arrays."""
        import jax.numpy as jnp

        # Start with NumPy mesh
        numpy_mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Convert to JAX
        jax_mesh = numpy_mesh.convert_to("jax")
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))

        # Convert back to NumPy
        numpy_mesh2 = jax_mesh.convert_to("numpy")
        self.assertIsInstance(numpy_mesh2.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh2.indices, np.ndarray)

        # Data should be preserved
        np.testing.assert_array_equal(numpy_mesh2.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh2.indices, self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_to_jax_with_custom_fields(self):
        """Test converting custom mesh class to JAX."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Optional

        class CustomMesh(Mesh):
            normals: Optional[Array] = Field(
                None, description="Normal vectors")

        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = CustomMesh(vertices=self.vertices,
                          indices=self.indices, normals=normals)

        # Convert to JAX
        jax_mesh = mesh.convert_to("jax")

        # Verify all arrays are converted
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))
        self.assertTrue(hasattr(jax_mesh.normals, 'device'))

        # Verify data is preserved
        np.testing.assert_array_equal(np.array(jax_mesh.normals), normals)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_to_numpy_with_nested_arrays(self):
        """Test converting mesh with nested dictionary arrays to NumPy."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Dict, Any, Optional

        class CustomMesh(Mesh):
            materials: Optional[Dict[str, Any]] = Field(
                None, description="Material properties")

        # Create with JAX arrays in nested structure
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

        # Convert to NumPy
        numpy_mesh = jax_mesh.convert_to("numpy")

        # Verify nested arrays are converted
        self.assertIsInstance(numpy_mesh.materials['diffuse'], np.ndarray)
        self.assertIsInstance(
            numpy_mesh.materials['properties']['roughness'], np.ndarray)

    def test_convert_to_jax_without_jax_raises_error(self):
        """Test that convert_to("jax") raises error when JAX is unavailable."""
        if HAS_JAX:
            self.skipTest("JAX is available, cannot test unavailable scenario")

        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        with self.assertRaises(AssertionError) as context:
            mesh.convert_to("jax")

        self.assertIn("JAX is not available", str(context.exception))


if __name__ == '__main__':
    unittest.main()
