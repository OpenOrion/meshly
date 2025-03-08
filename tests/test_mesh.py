"""
Tests for the Pydantic-based Mesh class.

This file contains tests to verify that the Pydantic-based Mesh class works correctly,
including inheritance, validation, and serialization/deserialization.
"""
import os
import tempfile
import numpy as np
import unittest
from typing import Optional, List, Dict, Any
from pydantic import Field, ValidationError

from pymeshoptimizer import Mesh


class TestPydanticMesh(unittest.TestCase):
    """Test Pydantic-based Mesh class functionality."""
    
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
    
    def test_mesh_creation(self):
        """Test that a Mesh can be created with vertices and indices."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        np.testing.assert_array_equal(mesh.vertices, self.vertices)
        np.testing.assert_array_equal(mesh.indices, self.indices)
    
    def test_mesh_validation(self):
        """Test that Mesh validation works correctly."""
        # Test that vertices are required
        with self.assertRaises(ValidationError):
            Mesh(indices=self.indices)
        
        # Test that indices are optional
        mesh = Mesh(vertices=self.vertices)
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, 0)
        self.assertIsNone(mesh.indices)
        
        # Test that vertices are converted to float32
        vertices_int = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1]
        ], dtype=np.int32)
        
        mesh = Mesh(vertices=vertices_int)
        self.assertEqual(mesh.vertices.dtype, np.float32)
        
        # Test that indices are converted to uint32
        indices_int = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)
        mesh = Mesh(vertices=self.vertices, indices=indices_int)
        self.assertEqual(mesh.indices.dtype, np.uint32)
    
    def test_mesh_optimization(self):
        """Test that mesh optimization methods work correctly."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        
        # Test optimize_vertex_cache
        mesh.optimize_vertex_cache()
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test optimize_overdraw
        mesh.optimize_overdraw()
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test optimize_vertex_fetch
        original_vertex_count = mesh.vertex_count
        mesh.optimize_vertex_fetch()
        self.assertLessEqual(mesh.vertex_count, original_vertex_count)
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test simplify
        original_index_count = mesh.index_count
        mesh.simplify(target_ratio=0.5)
        self.assertLessEqual(mesh.index_count, original_index_count)


class CustomMesh(Mesh):
    """A custom mesh class for testing."""
    normals: np.ndarray = Field(..., description="Vertex normals")
    colors: Optional[np.ndarray] = Field(None, description="Vertex colors")
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list, description="Tags for the mesh")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


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
    
    def test_custom_mesh_creation(self):
        """Test that a custom mesh can be created with additional attributes."""
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals,
            colors=self.colors,
            material_name="test_material",
            tags=["test", "cube"],
            properties={"shininess": 0.5, "reflectivity": 0.8}
        )
        
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        np.testing.assert_array_equal(mesh.vertices, self.vertices)
        np.testing.assert_array_equal(mesh.indices, self.indices)
        np.testing.assert_array_equal(mesh.normals, self.normals)
        np.testing.assert_array_equal(mesh.colors, self.colors)
        self.assertEqual(mesh.material_name, "test_material")
        self.assertEqual(mesh.tags, ["test", "cube"])
        self.assertEqual(mesh.properties, {"shininess": 0.5, "reflectivity": 0.8})
    
    def test_custom_mesh_validation(self):
        """Test that custom mesh validation works correctly."""
        # Test that normals are required
        with self.assertRaises(ValidationError):
            CustomMesh(
                vertices=self.vertices,
                indices=self.indices,
                colors=self.colors
            )
        
        # Test that colors are optional
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals
        )
        self.assertIsNone(mesh.colors)
        
        # Test default values
        self.assertEqual(mesh.material_name, "default")
        self.assertEqual(mesh.tags, [])
        self.assertEqual(mesh.properties, {})
    
    def test_custom_mesh_serialization(self):
        """Test that a custom mesh can be serialized and deserialized."""
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals,
            colors=self.colors,
            material_name="test_material",
            tags=["test", "cube"],
            properties={"shininess": 0.5, "reflectivity": 0.8}
        )
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the mesh to a zip file
            mesh.save_to_zip(temp_path)
            
            # Load the mesh from the zip file
            loaded_mesh = CustomMesh.load_from_zip(temp_path)
            
            # Check that the loaded mesh has the correct attributes
            self.assertEqual(loaded_mesh.vertex_count, mesh.vertex_count)
            self.assertEqual(loaded_mesh.index_count, mesh.index_count)
            np.testing.assert_array_almost_equal(loaded_mesh.vertices, mesh.vertices)
            np.testing.assert_array_almost_equal(loaded_mesh.normals, mesh.normals)
            np.testing.assert_array_almost_equal(loaded_mesh.colors, mesh.colors)
            self.assertEqual(loaded_mesh.material_name, mesh.material_name)
            self.assertEqual(loaded_mesh.tags, mesh.tags)
            self.assertEqual(loaded_mesh.properties, mesh.properties)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_custom_mesh_optimization(self):
        """Test that custom mesh optimization methods work correctly."""
        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals,
            colors=self.colors
        )
        
        # Test optimize_vertex_cache
        mesh.optimize_vertex_cache()
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test optimize_overdraw
        mesh.optimize_overdraw()
        self.assertEqual(mesh.vertex_count, len(self.vertices))
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test optimize_vertex_fetch
        original_vertex_count = mesh.vertex_count
        mesh.optimize_vertex_fetch()
        self.assertLessEqual(mesh.vertex_count, original_vertex_count)
        self.assertEqual(mesh.index_count, len(self.indices))
        
        # Test simplify
        original_index_count = mesh.index_count
        mesh.simplify(target_ratio=0.5)
        self.assertLessEqual(mesh.index_count, original_index_count)


if __name__ == '__main__':
    unittest.main()