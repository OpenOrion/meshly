"""
Tests for the EncodedMesh functionality.

This file contains tests to verify that the EncodedMesh class and related
functions work correctly.
"""
import os
import tempfile
import numpy as np
import unittest
from typing import Optional, List
from pydantic import Field

from pymeshoptimizer import Mesh, EncodedMesh

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
        
        self.mesh = Mesh(vertices=self.vertices, indices=self.indices)
    
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
    
    def test_mesh_encode_decode(self):
        """Test that the Mesh.encode and Mesh.decode methods work."""
        # Encode the mesh using the Mesh.encode method
        encoded_data = self.mesh.encode()
        
        # Check that the encoded_data is a dictionary with 'mesh' and 'arrays' keys
        self.assertIsInstance(encoded_data, dict)
        self.assertIn('mesh', encoded_data)
        self.assertIn('arrays', encoded_data)
        
        # Check that the encoded_mesh is an instance of EncodedMesh
        encoded_mesh = encoded_data['mesh']
        self.assertIsInstance(encoded_mesh, EncodedMesh)
        
        # Decode the mesh using the Mesh.decode method
        decoded_mesh = Mesh.decode(encoded_mesh)
        
        # Check that the decoded vertices match the original
        np.testing.assert_array_almost_equal(self.mesh.vertices, decoded_mesh.vertices)
        
        # Check that the triangles match
        original_triangles = self.get_triangles_set(self.mesh.vertices, self.mesh.indices)
        decoded_triangles = self.get_triangles_set(decoded_mesh.vertices, decoded_mesh.indices)
        
        self.assertEqual(original_triangles, decoded_triangles)

class CustomMesh(Mesh):
    """A custom mesh class for testing."""
    normals: np.ndarray = Field(..., description="Vertex normals")
    colors: Optional[np.ndarray] = Field(None, description="Vertex colors")
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list, description="Tags for the mesh")

class TestCustomMesh(unittest.TestCase):
    """Test custom Mesh subclass functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple mesh (a cube)
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
        
        self.normals = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.colors = np.array([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        self.mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals,
            colors=self.colors,
            material_name="test_material",
            tags=["test", "cube"]
        )
    
    def test_custom_mesh_attributes(self):
        """Test that the custom mesh attributes are set correctly."""
        self.assertEqual(self.mesh.vertex_count, len(self.vertices))
        self.assertEqual(self.mesh.index_count, len(self.indices))
        np.testing.assert_array_equal(self.mesh.normals, self.normals)
        np.testing.assert_array_equal(self.mesh.colors, self.colors)
        self.assertEqual(self.mesh.material_name, "test_material")
        self.assertEqual(self.mesh.tags, ["test", "cube"])
    
    def test_custom_mesh_encode_decode(self):
        """Test that the custom mesh can be encoded and decoded."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the mesh to a zip file
            self.mesh.save_to_zip(temp_path)
            
            # Load the mesh from the zip file
            loaded_mesh = CustomMesh.load_from_zip(temp_path)
            
            # Check that the loaded mesh has the correct attributes
            self.assertEqual(loaded_mesh.vertex_count, self.mesh.vertex_count)
            self.assertEqual(loaded_mesh.index_count, self.mesh.index_count)
            np.testing.assert_array_almost_equal(loaded_mesh.vertices, self.mesh.vertices)
            np.testing.assert_array_almost_equal(loaded_mesh.normals, self.mesh.normals)
            np.testing.assert_array_almost_equal(loaded_mesh.colors, self.mesh.colors)
            self.assertEqual(loaded_mesh.material_name, self.mesh.material_name)
            self.assertEqual(loaded_mesh.tags, self.mesh.tags)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()