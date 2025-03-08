"""
Tests for the io module.
"""
import os
import tempfile
import unittest
import numpy as np
from pymeshoptimizer import (
    Mesh,
    EncodedMesh,
    encode_mesh,
    encode_array,
    decode_array,
)
from pymeshoptimizer.io import (
    save_encoded_array_to_file,
    load_encoded_array_from_file,
    save_array_to_file,
    load_array_from_file,
    save_encoded_array_to_zip,
    load_encoded_array_from_zip,
    save_array_to_zip,
    load_array_from_zip,
    save_arrays_to_zip,
    load_arrays_from_zip,
    save_encoded_mesh_to_zip,
    load_encoded_mesh_from_zip,
    save_mesh_to_zip,
    load_mesh_from_zip,
    save_combined_data_to_zip,
    get_combined_data_as_bytes,
    load_combined_data_from_zip,
)

class TestIO(unittest.TestCase):
    """Test cases for the io module."""
    
    def setUp(self):
        """Set up test data."""
        # Create test arrays
        self.array_1d = np.linspace(0, 10, 100, dtype=np.float32)
        self.array_2d = np.random.random((50, 3)).astype(np.float32)
        self.array_3d = np.random.random((10, 10, 10)).astype(np.float32)
        self.array_int = np.random.randint(0, 100, (20, 20), dtype=np.int32)
        
        # Create test mesh (a cube)
        self.vertices = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,  # front
            1, 5, 6, 6, 2, 1,  # right
            5, 4, 7, 7, 6, 5,  # back
            4, 0, 3, 3, 7, 4,  # left
            3, 2, 6, 6, 7, 3,  # top
            4, 5, 1, 1, 0, 4   # bottom
        ], dtype=np.uint32)
        
        self.mesh = Mesh(self.vertices, self.indices)
        self.encoded_mesh = encode_mesh(self.vertices, self.indices)
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        # Remove temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
    
    def get_triangles_set(self, vertices, indices):
        """
        Get a set of triangles from vertices and indices.
        Each triangle is represented as a frozenset of tuples of vertex coordinates.
        This makes the comparison invariant to vertex order within triangles.
        """
        triangles = set()
        for i in range(0, len(indices), 3):
            # Get the three vertices of the triangle
            v1 = tuple(vertices[indices[i]])
            v2 = tuple(vertices[indices[i+1]])
            v3 = tuple(vertices[indices[i+2]])
            # Create a frozenset of the vertices (order-invariant)
            triangle = frozenset([v1, v2, v3])
            triangles.add(triangle)
        return triangles
    
    def test_save_load_file(self):
        """Test saving and loading an array to/from a file."""
        file_path = os.path.join(self.temp_dir, "test_array.bin")
        
        # Create a larger array for this test to ensure compression is effective
        large_array = np.random.random((1000, 3)).astype(np.float32)
        
        # Save the array to a file
        save_array_to_file(large_array, file_path)
        
        # Load the array from the file
        loaded = load_array_from_file(file_path)
        
        # Check that the loaded array matches the original
        np.testing.assert_allclose(loaded, large_array, rtol=1e-5)
        
        # Check that the file is smaller than the original array
        self.assertLess(os.path.getsize(file_path), large_array.nbytes)
        
        # Print compression ratio
        print(f"File compression ratio: {os.path.getsize(file_path) / large_array.nbytes:.2f}")
    
    def test_save_load_encoded_file(self):
        """Test saving and loading an encoded array to/from a file."""
        file_path = os.path.join(self.temp_dir, "test_encoded_array.bin")
        
        # Encode the array
        encoded = encode_array(self.array_2d)
        
        # Save the encoded array to a file
        save_encoded_array_to_file(encoded, file_path)
        
        # Load the encoded array from the file
        loaded_encoded = load_encoded_array_from_file(file_path)
        
        # Check original and loaded encoded arrays have same attributes
        self.assertEqual(loaded_encoded.shape, encoded.shape)
        self.assertEqual(loaded_encoded.dtype, encoded.dtype)
        self.assertEqual(loaded_encoded.itemsize, encoded.itemsize)
        
        # Check that decoded data matches original
        decoded = decode_array(loaded_encoded)
        np.testing.assert_allclose(decoded, self.array_2d, rtol=1e-5)
    
    def test_save_load_array_zip(self):
        """Test saving and loading an array to/from a zip file."""
        zip_path = os.path.join(self.temp_dir, "test_array.zip")
        
        # Create a larger array for this test to ensure compression is effective
        large_array = np.random.random((1000, 3)).astype(np.float32)
        
        # Save the array to a zip file
        save_array_to_zip(large_array, zip_path)
        
        # Load the array from the zip file
        loaded = load_array_from_zip(zip_path)
        
        # Check that the loaded array matches the original
        np.testing.assert_allclose(loaded, large_array, rtol=1e-5)
        
        # Check that the zip file is smaller than the original array
        self.assertLess(os.path.getsize(zip_path), large_array.nbytes)
        
        # Print compression ratio
        print(f"Zip compression ratio: {os.path.getsize(zip_path) / large_array.nbytes:.2f}")
    
    def test_save_load_encoded_array_zip(self):
        """Test saving and loading an encoded array to/from a zip file."""
        zip_path = os.path.join(self.temp_dir, "test_encoded_array.zip")
        
        # Encode the array
        encoded = encode_array(self.array_2d)
        
        # Save the encoded array to a zip file
        save_encoded_array_to_zip(encoded, zip_path)
        
        # Load the encoded array from the zip file
        loaded_encoded = load_encoded_array_from_zip(zip_path)
        
        # Check original and loaded encoded arrays have same attributes
        self.assertEqual(loaded_encoded.shape, encoded.shape)
        self.assertEqual(loaded_encoded.dtype, encoded.dtype)
        self.assertEqual(loaded_encoded.itemsize, encoded.itemsize)
        
        # Check that decoded data matches original
        decoded = decode_array(loaded_encoded)
        np.testing.assert_allclose(decoded, self.array_2d, rtol=1e-5)
    
    def test_save_load_multiple_arrays(self):
        """Test saving and loading multiple arrays to/from a zip file."""
        zip_path = os.path.join(self.temp_dir, "test_multiple_arrays.zip")
        
        # Create a dictionary of arrays
        arrays = {
            "array_1d": self.array_1d,
            "array_2d": self.array_2d,
            "array_3d": self.array_3d,
            "array_int": self.array_int
        }
        
        # Save all arrays to a single zip file
        save_arrays_to_zip(arrays, zip_path)
        
        # Load the arrays from the zip file
        loaded_arrays = load_arrays_from_zip(zip_path)
        
        # Check that all arrays were loaded
        self.assertEqual(set(loaded_arrays.keys()), set(arrays.keys()))
        
        # Check that each loaded array matches the original
        for name, array in arrays.items():
            np.testing.assert_allclose(loaded_arrays[name], array, rtol=1e-5)
        
        # Check that the zip file is smaller than the combined size of all arrays
        total_size = sum(array.nbytes for array in arrays.values())
        self.assertLess(os.path.getsize(zip_path), total_size)
        
        # Print compression ratio
        print(f"Multiple arrays compression ratio: {os.path.getsize(zip_path) / total_size:.2f}")
    
    def test_save_load_encoded_mesh(self):
        """Test saving and loading an encoded mesh to/from a zip file."""
        # Create a temporary file
        zip_path = os.path.join(self.temp_dir, "test_encoded_mesh.zip")
        
        # Save the encoded mesh to the zip file
        save_encoded_mesh_to_zip(self.encoded_mesh, zip_path)
        
        # Load the encoded mesh from the zip file
        loaded_encoded_mesh = load_encoded_mesh_from_zip(zip_path)
        
        # Check that the loaded encoded mesh has the correct attributes
        self.assertEqual(loaded_encoded_mesh.vertex_count, self.encoded_mesh.vertex_count)
        self.assertEqual(loaded_encoded_mesh.vertex_size, self.encoded_mesh.vertex_size)
        self.assertEqual(loaded_encoded_mesh.index_count, self.encoded_mesh.index_count)
        self.assertEqual(loaded_encoded_mesh.index_size, self.encoded_mesh.index_size)
        self.assertEqual(loaded_encoded_mesh.vertices, self.encoded_mesh.vertices)
        self.assertEqual(loaded_encoded_mesh.indices, self.encoded_mesh.indices)
        
        # Decode the loaded encoded mesh
        from pymeshoptimizer import decode_mesh
        decoded_vertices, decoded_indices = decode_mesh(loaded_encoded_mesh)
        
        # Check that the decoded vertices match the original
        np.testing.assert_array_almost_equal(self.vertices, decoded_vertices)
        
        # Check that the triangles match
        original_triangles = self.get_triangles_set(self.vertices, self.indices)
        decoded_triangles = self.get_triangles_set(decoded_vertices, decoded_indices)
        self.assertEqual(original_triangles, decoded_triangles)
    
    def test_save_load_mesh(self):
        """Test saving and loading a mesh to/from a zip file."""
        # Create a temporary file
        zip_path = os.path.join(self.temp_dir, "test_mesh.zip")
        
        # Save the mesh to the zip file
        save_mesh_to_zip(self.mesh, zip_path)
        
        # Load the mesh from the zip file
        loaded_mesh = load_mesh_from_zip(Mesh, zip_path)
        
        # Check that the loaded mesh has the correct attributes
        self.assertEqual(loaded_mesh.vertex_count, self.mesh.vertex_count)
        self.assertEqual(loaded_mesh.index_count, self.mesh.index_count)
        
        # Check that the vertices match
        np.testing.assert_array_almost_equal(loaded_mesh.vertices, self.mesh.vertices)
        
        # Check that the triangles match
        original_triangles = self.get_triangles_set(self.mesh.vertices, self.mesh.indices)
        loaded_triangles = self.get_triangles_set(loaded_mesh.vertices, loaded_mesh.indices)
        self.assertEqual(original_triangles, loaded_triangles)
    
    def test_custom_filenames(self):
        """Test saving and loading with custom filenames."""
        # Create a temporary file
        zip_path = os.path.join(self.temp_dir, "test_custom_filenames.zip")
        
        # Custom filenames
        vertices_filename = "custom_vertices.bin"
        indices_filename = "custom_indices.bin"
        metadata_filename = "custom_metadata.json"
        
        # Save the encoded mesh to the zip file with custom filenames
        save_encoded_mesh_to_zip(
            self.encoded_mesh, 
            zip_path, 
            vertices_filename, 
            indices_filename, 
            metadata_filename
        )
        
        # Load the encoded mesh from the zip file with custom filenames
        loaded_encoded_mesh = load_encoded_mesh_from_zip(
            zip_path, 
            vertices_filename, 
            indices_filename, 
            metadata_filename
        )
        
        # Check that the loaded encoded mesh has the correct attributes
        self.assertEqual(loaded_encoded_mesh.vertex_count, self.encoded_mesh.vertex_count)
        self.assertEqual(loaded_encoded_mesh.vertex_size, self.encoded_mesh.vertex_size)
        self.assertEqual(loaded_encoded_mesh.index_count, self.encoded_mesh.index_count)
        self.assertEqual(loaded_encoded_mesh.index_size, self.encoded_mesh.index_size)
        self.assertEqual(loaded_encoded_mesh.vertices, self.encoded_mesh.vertices)
        self.assertEqual(loaded_encoded_mesh.indices, self.encoded_mesh.indices)
    
    def test_save_load_combined_data(self):
        """Test saving and loading combined data (mesh and arrays) to/from a zip file."""
        # Create a temporary file
        zip_path = os.path.join(self.temp_dir, "test_combined_data.zip")
        
        # Create encoded arrays
        encoded_array1 = encode_array(self.array_1d)
        encoded_array2 = encode_array(self.array_2d)
        encoded_arrays = {
            "array1": encoded_array1,
            "array2": encoded_array2
        }
        
        # Create metadata
        metadata = {
            "name": "Test Mesh",
            "version": "1.0",
            "description": "A test mesh with arrays"
        }
        
        # Save the combined data to the zip file
        save_combined_data_to_zip(
            encoded_mesh=self.encoded_mesh,
            encoded_arrays=encoded_arrays,
            metadata=metadata,
            zip_path=zip_path
        )
        
        # Load the combined data from the zip file
        loaded_mesh, loaded_arrays, loaded_metadata = load_combined_data_from_zip(zip_path)
        
        # Check that the loaded mesh has the correct attributes
        self.assertEqual(loaded_mesh.vertex_count, self.encoded_mesh.vertex_count)
        self.assertEqual(loaded_mesh.vertex_size, self.encoded_mesh.vertex_size)
        self.assertEqual(loaded_mesh.index_count, self.encoded_mesh.index_count)
        self.assertEqual(loaded_mesh.index_size, self.encoded_mesh.index_size)
        self.assertEqual(loaded_mesh.vertices, self.encoded_mesh.vertices)
        self.assertEqual(loaded_mesh.indices, self.encoded_mesh.indices)
        
        # Check that the loaded arrays have the correct attributes
        self.assertEqual(set(loaded_arrays.keys()), set(encoded_arrays.keys()))
        for name, encoded_array in encoded_arrays.items():
            loaded_array = loaded_arrays[name]
            self.assertEqual(loaded_array.shape, encoded_array.shape)
            self.assertEqual(loaded_array.dtype, encoded_array.dtype)
            self.assertEqual(loaded_array.itemsize, encoded_array.itemsize)
            self.assertEqual(loaded_array.data, encoded_array.data)
        
        # Check that the loaded metadata matches the original
        self.assertEqual(loaded_metadata, metadata)
    
    def test_get_combined_data_as_bytes(self):
        """Test getting combined data as bytes and loading it back."""
        # Create encoded arrays
        encoded_array1 = encode_array(self.array_1d)
        encoded_array2 = encode_array(self.array_2d)
        encoded_arrays = {
            "array1": encoded_array1,
            "array2": encoded_array2
        }
        
        # Create metadata
        metadata = {
            "name": "Test Mesh",
            "version": "1.0",
            "description": "A test mesh with arrays"
        }
        
        # Get the combined data as bytes
        zip_bytes = get_combined_data_as_bytes(
            encoded_mesh=self.encoded_mesh,
            encoded_arrays=encoded_arrays,
            metadata=metadata
        )
        
        # Verify that we got bytes
        self.assertIsInstance(zip_bytes, bytes)
        self.assertTrue(len(zip_bytes) > 0)
        
        # Load the combined data from the bytes
        loaded_mesh, loaded_arrays, loaded_metadata = load_combined_data_from_zip(zip_bytes)
        
        # Check that the loaded mesh has the correct attributes
        self.assertEqual(loaded_mesh.vertex_count, self.encoded_mesh.vertex_count)
        self.assertEqual(loaded_mesh.vertex_size, self.encoded_mesh.vertex_size)
        self.assertEqual(loaded_mesh.index_count, self.encoded_mesh.index_count)
        self.assertEqual(loaded_mesh.index_size, self.encoded_mesh.index_size)
        self.assertEqual(loaded_mesh.vertices, self.encoded_mesh.vertices)
        self.assertEqual(loaded_mesh.indices, self.encoded_mesh.indices)
        
        # Check that the loaded arrays have the correct attributes
        self.assertEqual(set(loaded_arrays.keys()), set(encoded_arrays.keys()))
        for name, encoded_array in encoded_arrays.items():
            loaded_array = loaded_arrays[name]
            self.assertEqual(loaded_array.shape, encoded_array.shape)
            self.assertEqual(loaded_array.dtype, encoded_array.dtype)
            self.assertEqual(loaded_array.itemsize, encoded_array.itemsize)
            self.assertEqual(loaded_array.data, encoded_array.data)
        
        # Check that the loaded metadata matches the original
        self.assertEqual(loaded_metadata, metadata)
        
        # Compare with file-based approach
        zip_path = os.path.join(self.temp_dir, "test_combined_data_comparison.zip")
        save_combined_data_to_zip(
            encoded_mesh=self.encoded_mesh,
            encoded_arrays=encoded_arrays,
            metadata=metadata,
            zip_path=zip_path
        )
        
        # Load from file for comparison
        file_loaded_mesh, file_loaded_arrays, file_loaded_metadata = load_combined_data_from_zip(zip_path)
        
        # Check that both approaches yield the same results
        self.assertEqual(loaded_mesh.vertices, file_loaded_mesh.vertices)
        self.assertEqual(loaded_mesh.indices, file_loaded_mesh.indices)
        for name in encoded_arrays.keys():
            self.assertEqual(loaded_arrays[name].data, file_loaded_arrays[name].data)
        self.assertEqual(loaded_metadata, file_loaded_metadata)

if __name__ == '__main__':
    unittest.main()