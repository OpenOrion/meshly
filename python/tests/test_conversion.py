"""
Tests for array type conversion functions (to_numpy, to_jax).
"""

import unittest
import numpy as np
from meshly import Mesh, MeshUtils, Array, HAS_JAX


class TestConversion(unittest.TestCase):
    """Test array type conversion functionality."""

    def setUp(self):
        """Set up test data."""
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.indices = np.array([0, 1, 2], dtype=np.uint32)

    def test_to_numpy_conversion(self):
        """Test converting mesh to NumPy arrays."""
        # Start with NumPy mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Convert to NumPy (should be no-op but create new instance)
        numpy_mesh = MeshUtils.to_numpy(mesh)
        
        self.assertIsInstance(numpy_mesh.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh.indices, np.ndarray)
        np.testing.assert_array_equal(numpy_mesh.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh.indices, self.indices)
        
        # Verify it's a different instance
        self.assertIsNot(numpy_mesh, mesh)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_to_jax_conversion(self):
        """Test converting mesh to JAX arrays."""
        import jax.numpy as jnp
        
        # Start with NumPy mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Convert to JAX
        jax_mesh = MeshUtils.to_jax(mesh)
        
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'), "Vertices should be JAX arrays")
        self.assertTrue(hasattr(jax_mesh.indices, 'device'), "Indices should be JAX arrays")
        np.testing.assert_array_equal(np.array(jax_mesh.vertices), self.vertices)
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
        jax_mesh = MeshUtils.to_jax(numpy_mesh)
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))
        
        # Convert back to NumPy
        numpy_mesh2 = MeshUtils.to_numpy(jax_mesh)
        self.assertIsInstance(numpy_mesh2.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh2.indices, np.ndarray)
        
        # Data should be preserved
        np.testing.assert_array_equal(numpy_mesh2.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh2.indices, self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_to_jax_with_custom_fields(self):
        """Test converting custom mesh class to JAX."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Optional
        
        class CustomMesh(Mesh):
            normals: Optional[Array] = Field(None, description="Normal vectors")
        
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = CustomMesh(vertices=self.vertices, indices=self.indices, normals=normals)
        
        # Convert to JAX
        jax_mesh = MeshUtils.to_jax(mesh)
        
        # Verify all arrays are converted
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))
        self.assertTrue(hasattr(jax_mesh.normals, 'device'))
        
        # Verify data is preserved
        np.testing.assert_array_equal(np.array(jax_mesh.normals), normals)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_to_numpy_with_nested_arrays(self):
        """Test converting mesh with nested dictionary arrays to NumPy."""
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Dict, Any, Optional
        
        class CustomMesh(Mesh):
            materials: Optional[Dict[str, Any]] = Field(None, description="Material properties")
        
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
        numpy_mesh = MeshUtils.to_numpy(jax_mesh)
        
        # Verify nested arrays are converted
        self.assertIsInstance(numpy_mesh.materials['diffuse'], np.ndarray)
        self.assertIsInstance(numpy_mesh.materials['properties']['roughness'], np.ndarray)

    def test_to_jax_without_jax_raises_error(self):
        """Test that to_jax raises error when JAX is unavailable."""
        if HAS_JAX:
            self.skipTest("JAX is available, cannot test unavailable scenario")
        
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        with self.assertRaises(ValueError) as context:
            MeshUtils.to_jax(mesh)
        
        self.assertIn("JAX is not available", str(context.exception))


if __name__ == '__main__':
    unittest.main()
