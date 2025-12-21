"""
Tests for JAX array support in meshly.
"""

import unittest
import numpy as np
from meshly import Mesh, MeshUtils, Array, HAS_JAX


class TestJAXSupport(unittest.TestCase):
    """Test JAX array support functionality."""

    def setUp(self):
        """Set up test data."""
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.indices = np.array([0, 1, 2], dtype=np.uint32)

    def test_array_type_definition(self):
        """Test that Array type is properly defined."""
        self.assertIsNotNone(Array)
        
        # Test that numpy arrays are compatible with Array type
        np_array = np.array([1, 2, 3])
        self.assertIsInstance(np_array, np.ndarray)

    def test_has_jax_flag(self):
        """Test that HAS_JAX flag is properly set."""
        self.assertIsInstance(HAS_JAX, bool)

    def test_numpy_functionality_preserved(self):
        """Test that existing numpy functionality still works."""
        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Verify arrays are numpy arrays
        self.assertIsInstance(mesh.vertices, np.ndarray)
        self.assertIsInstance(mesh.indices, np.ndarray)
        
        # Test basic properties
        self.assertEqual(mesh.vertex_count, 3)
        self.assertEqual(mesh.index_count, 3)
        
        # Test encoding/decoding without JAX
        encoded = MeshUtils.encode(mesh)
        decoded = MeshUtils.decode(Mesh, encoded, use_jax=False)
        
        self.assertIsInstance(decoded.vertices, np.ndarray)
        self.assertIsInstance(decoded.indices, np.ndarray)
        np.testing.assert_array_equal(decoded.vertices, self.vertices)
        np.testing.assert_array_equal(decoded.indices, self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_jax_functionality(self):
        """Test JAX functionality when available."""
        import jax.numpy as jnp
        
        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Test encoding/decoding with JAX
        encoded = MeshUtils.encode(mesh)
        decoded_jax = MeshUtils.decode(Mesh, encoded, use_jax=True)
        
        # Verify arrays are JAX arrays
        self.assertTrue(hasattr(decoded_jax.vertices, 'device'), "Vertices should be JAX arrays")
        self.assertTrue(hasattr(decoded_jax.indices, 'device'), "Indices should be JAX arrays")
        
        # Verify data is preserved
        np.testing.assert_array_equal(np.array(decoded_jax.vertices), self.vertices)
        np.testing.assert_array_equal(np.array(decoded_jax.indices), self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_jax_input_arrays(self):
        """Test using JAX arrays as input."""
        import jax.numpy as jnp
        
        # Create JAX arrays
        jax_vertices = jnp.array(self.vertices)
        jax_indices = jnp.array(self.indices)
        
        # Create mesh with JAX arrays
        mesh = Mesh(vertices=jax_vertices, indices=jax_indices)
        
        # Vertices should remain JAX, indices converted to numpy for meshoptimizer
        self.assertTrue(hasattr(mesh.vertices, 'device'), "Vertices should remain JAX arrays")
        self.assertIsInstance(mesh.indices, np.ndarray)  # Converted for meshoptimizer compatibility

    def test_jax_unavailable_error(self):
        """Test error handling when JAX is requested but unavailable."""
        if HAS_JAX:
            self.skipTest("JAX is available, cannot test unavailable scenario")
            
        # Create mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        encoded = MeshUtils.encode(mesh)
        
        # Should raise error when JAX is requested but not available
        with self.assertRaises(ValueError) as context:
            MeshUtils.decode(Mesh, encoded, use_jax=True)
        
        self.assertIn("JAX is not available", str(context.exception))

    def test_mesh_copy_with_jax_arrays(self):
        """Test mesh copying with JAX arrays."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
            
        import jax.numpy as jnp
        
        # Create mesh with JAX vertices
        jax_vertices = jnp.array(self.vertices)
        mesh = Mesh(vertices=jax_vertices, indices=self.indices)
        
        # Test copying
        copied_mesh = mesh.copy()
        
        # Verify copy preserves array types and data
        self.assertTrue(hasattr(copied_mesh.vertices, 'device'), "Copied vertices should be JAX arrays")
        np.testing.assert_array_equal(np.array(copied_mesh.vertices), self.vertices)

    def test_additional_arrays_jax_support(self):
        """Test that additional arrays also support JAX conversion."""
        if not HAS_JAX:
            self.skipTest("JAX not available")
            
        import jax.numpy as jnp
        from pydantic import Field
        from typing import Optional
        
        # Create a custom mesh class with additional arrays
        class CustomMesh(Mesh):
            normals: Optional[Array] = Field(None, description="Normal vectors")
        
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        mesh = CustomMesh(vertices=self.vertices, indices=self.indices, normals=normals)
        
        # Test encoding/decoding with JAX
        encoded = MeshUtils.encode(mesh)
        decoded_jax = MeshUtils.decode(CustomMesh, encoded, use_jax=True)
        
        # Verify all arrays are JAX arrays
        self.assertTrue(hasattr(decoded_jax.vertices, 'device'), "Vertices should be JAX arrays")
        self.assertTrue(hasattr(decoded_jax.normals, 'device'), "Normals should be JAX arrays")
        
        # Verify data is preserved
        np.testing.assert_array_equal(np.array(decoded_jax.normals), normals)


if __name__ == '__main__':
    unittest.main()