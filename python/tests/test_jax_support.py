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
        
        # Create a custom mesh class with additional arrays
        class CustomMesh(Mesh):
            def __init__(self, **kwargs):
                # Extract additional fields
                self.normals = kwargs.pop('normals', None)
                super().__init__(**kwargs)
        
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

    def test_convert_arrays_to_numpy(self):
        """Test converting mesh arrays to numpy."""
        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Convert to numpy (should be no-op)
        numpy_mesh = MeshUtils.convert_arrays(mesh, "numpy")
        
        self.assertIsInstance(numpy_mesh.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh.indices, np.ndarray)
        np.testing.assert_array_equal(numpy_mesh.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh.indices, self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_arrays_to_jax(self):
        """Test converting mesh arrays to JAX."""
        import jax.numpy as jnp
        
        # Create mesh with numpy arrays
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Convert to JAX
        jax_mesh = MeshUtils.convert_arrays(mesh, "jax")
        
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'), "Vertices should be JAX arrays")
        self.assertTrue(hasattr(jax_mesh.indices, 'device'), "Indices should be JAX arrays")
        np.testing.assert_array_equal(np.array(jax_mesh.vertices), self.vertices)
        np.testing.assert_array_equal(np.array(jax_mesh.indices), self.indices)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_arrays_bidirectional(self):
        """Test converting between numpy and JAX arrays."""
        import jax.numpy as jnp
        
        # Start with numpy mesh
        numpy_mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Convert to JAX
        jax_mesh = MeshUtils.convert_arrays(numpy_mesh, "jax")
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))
        
        # Convert back to numpy
        numpy_mesh2 = MeshUtils.convert_arrays(jax_mesh, "numpy")
        self.assertIsInstance(numpy_mesh2.vertices, np.ndarray)
        self.assertIsInstance(numpy_mesh2.indices, np.ndarray)
        
        # Data should be preserved
        np.testing.assert_array_equal(numpy_mesh2.vertices, self.vertices)
        np.testing.assert_array_equal(numpy_mesh2.indices, self.indices)

    def test_convert_arrays_invalid_type(self):
        """Test error handling for invalid target type."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        with self.assertRaises(ValueError) as context:
            MeshUtils.convert_arrays(mesh, "invalid")  # type: ignore
        
        self.assertIn("Invalid target_type", str(context.exception))

    def test_convert_arrays_jax_unavailable(self):
        """Test error handling when JAX conversion is requested but JAX is unavailable."""
        if HAS_JAX:
            self.skipTest("JAX is available, cannot test unavailable scenario")
        
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        with self.assertRaises(ValueError) as context:
            MeshUtils.convert_arrays(mesh, "jax")
        
        self.assertIn("JAX is not available", str(context.exception))

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_arrays_with_additional_fields(self):
        """Test array conversion with custom mesh class that has additional array fields."""
        import jax.numpy as jnp
        
        # Create a custom mesh class with additional arrays
        class CustomMesh(Mesh):
            def __init__(self, **kwargs):
                self.normals = kwargs.pop('normals', None)
                self.texture_coords = kwargs.pop('texture_coords', None)
                super().__init__(**kwargs)
        
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        texture_coords = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
        
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=normals,
            texture_coords=texture_coords
        )
        
        # Convert to JAX
        jax_mesh = MeshUtils.convert_arrays(mesh, "jax")
        
        # Verify all arrays are converted
        self.assertTrue(hasattr(jax_mesh.vertices, 'device'))
        self.assertTrue(hasattr(jax_mesh.normals, 'device'))
        self.assertTrue(hasattr(jax_mesh.texture_coords, 'device'))
        
        # Verify data is preserved
        np.testing.assert_array_equal(np.array(jax_mesh.normals), normals)
        np.testing.assert_array_equal(np.array(jax_mesh.texture_coords), texture_coords)

    @unittest.skipUnless(HAS_JAX, "JAX not available")
    def test_convert_arrays_with_nested_dict_arrays(self):
        """Test array conversion with nested dictionary fields containing arrays."""
        import jax.numpy as jnp
        
        # Create a custom mesh class with nested dictionary arrays
        class CustomMesh(Mesh):
            def __init__(self, **kwargs):
                self.materials = kwargs.pop('materials', None)
                super().__init__(**kwargs)
        
        materials = {
            'diffuse': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'properties': {
                'roughness': np.array([0.5], dtype=np.float32),
                'metallic': np.array([0.0], dtype=np.float32)
            }
        }
        
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            materials=materials
        )
        
        # Convert to JAX
        jax_mesh = MeshUtils.convert_arrays(mesh, "jax")
        
        # Verify nested arrays are converted
        self.assertTrue(hasattr(jax_mesh.materials['diffuse'], 'device'))
        self.assertTrue(hasattr(jax_mesh.materials['properties']['roughness'], 'device'))
        self.assertTrue(hasattr(jax_mesh.materials['properties']['metallic'], 'device'))
        
        # Verify data is preserved
        np.testing.assert_array_equal(np.array(jax_mesh.materials['diffuse']), materials['diffuse'])
        np.testing.assert_array_equal(np.array(jax_mesh.materials['properties']['roughness']), materials['properties']['roughness'])


if __name__ == '__main__':
    unittest.main()