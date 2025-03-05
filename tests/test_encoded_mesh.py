"""
Tests for the EncodedMesh functionality.

This file contains tests to verify that the EncodedMesh class and related
functions work correctly.
"""
import numpy as np
import unittest
from pymeshoptimizer import Mesh, EncodedMesh, encode_mesh, decode_mesh

class TestEncodedMesh(unittest.TestCase):
    """Test EncodedMesh functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple mesh (a cube)
        self.vertices = np.array([
            # positions          
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
    
    def test_encode_mesh_function(self):
        """Test that the encode_mesh function works correctly."""
        # Encode the mesh using the encode_mesh function
        encoded_mesh = encode_mesh(
            self.vertices,
            self.indices
        )
        
        # Check that the encoded_mesh has the correct attributes
        self.assertEqual(encoded_mesh.vertex_count, len(self.vertices))
        self.assertEqual(encoded_mesh.vertex_size, self.vertices.itemsize * self.vertices.shape[1])
        self.assertEqual(encoded_mesh.index_count, len(self.indices))
        self.assertEqual(encoded_mesh.index_size, 4)  # uint32 = 4 bytes
        self.assertIsNotNone(encoded_mesh.vertices)
        self.assertIsNotNone(encoded_mesh.indices)
    
    def test_decode_mesh_function(self):
        """Test that the decode_mesh function works correctly."""
        # Encode the mesh
        encoded_mesh = encode_mesh(
            self.vertices,
            self.indices
        )
        
        # Decode the mesh using the decode_mesh function
        decoded_vertices, decoded_indices = decode_mesh(encoded_mesh)
        
        # Check that the decoded vertices match the original
        np.testing.assert_array_almost_equal(self.vertices, decoded_vertices)
        
        # The indices might not match exactly due to how the encoding/decoding works,
        # but the geometry should be preserved. Let's check that by comparing
        # the triangles.
        original_triangles = self.get_triangles_set(self.vertices, self.indices)
        decoded_triangles = self.get_triangles_set(decoded_vertices, decoded_indices)
        
        # Check that the triangles match
        self.assertEqual(original_triangles, decoded_triangles)
    
    def test_mesh_encode_decode(self):
        """Test that the Mesh.encode and Mesh.decode methods work."""
        # Encode the mesh using the Mesh.encode method
        encoded_mesh = self.mesh.encode()
        
        # Check that the encoded_mesh is an instance of EncodedMesh
        self.assertIsInstance(encoded_mesh, EncodedMesh)
        
        # Decode the mesh using the Mesh.decode method
        decoded_mesh = Mesh.decode(encoded_mesh)
        
        # Check that the decoded vertices match the original
        np.testing.assert_array_almost_equal(self.mesh.vertices, decoded_mesh.vertices)
        
        # Check that the triangles match
        original_triangles = self.get_triangles_set(self.mesh.vertices, self.mesh.indices)
        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)
        
        self.assertEqual(original_triangles, decoded_triangles)

if __name__ == '__main__':
    unittest.main()