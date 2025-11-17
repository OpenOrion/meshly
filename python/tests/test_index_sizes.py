"""
Tests for the index_sizes and cell_types functionality in the Mesh class.

This file contains tests to verify that the index_sizes and cell_types fields work correctly
for different polygon types and formats, including automatic inference,
validation, and preservation during serialization.
"""
import os
import tempfile
import numpy as np
import unittest
from typing import List
from pydantic import ValidationError

from meshly import Mesh, MeshUtils


class TestIndexSizes(unittest.TestCase):
    """Test index_sizes functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create vertices for a simple mesh
        self.vertices = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.5, 0.5, 1.0],  # 4
            [2.0, 0.0, 0.0],  # 5
            [2.0, 1.0, 0.0]   # 6
        ], dtype=np.float32)
    
    def test_triangular_indices_no_index_sizes(self):
        """Test triangular mesh without explicit index_sizes."""
        # Traditional triangular mesh format
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        mesh = Mesh(vertices=self.vertices, indices=indices)
        
        self.assertEqual(mesh.index_count, 6)
        self.assertEqual(mesh.polygon_count, 2)  # Now auto-infers triangles
        self.assertTrue(mesh.is_uniform_polygons)  # Auto-inferred uniform triangles
        np.testing.assert_array_equal(mesh.index_sizes, [3, 3])  # Auto-inferred triangle sizes
    
    def test_quad_indices_2d_array(self):
        """Test quad mesh using 2D numpy array with automatic index_sizes inference."""
        # 2D array format for uniform quads
        indices = np.array([
            [0, 1, 2, 3],  # First quad
            [1, 5, 6, 2]   # Second quad
        ], dtype=np.uint32)
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        
        self.assertEqual(mesh.index_count, 8)  # Flattened to 8 indices
        self.assertEqual(mesh.polygon_count, 2)  # 2 polygons
        self.assertTrue(mesh.is_uniform_polygons)  # All quads
        np.testing.assert_array_equal(mesh.index_sizes, [4, 4])
        
        # Check that we can get back the original structure
        polygon_indices = mesh.get_polygon_indices()
        np.testing.assert_array_equal(polygon_indices, indices)
    
    def test_mixed_polygons_list_of_lists(self):
        """Test mixed polygon mesh using list of lists with automatic index_sizes inference."""
        # Mixed polygon format: triangle, quad, pentagon
        indices = [
            [0, 1, 2],        # Triangle
            [1, 5, 6, 2],     # Quad
            [0, 1, 4, 3, 2]   # Pentagon
        ]
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        
        self.assertEqual(mesh.index_count, 12)  # 3 + 4 + 5 = 12
        self.assertEqual(mesh.polygon_count, 3)  # 3 polygons
        self.assertFalse(mesh.is_uniform_polygons)  # Mixed sizes
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 5])
        
        # Check that we can get back the original structure
        polygon_indices = mesh.get_polygon_indices()
        self.assertEqual(len(polygon_indices), 3)
        self.assertEqual(polygon_indices[0], [0, 1, 2])
        self.assertEqual(polygon_indices[1], [1, 5, 6, 2])
        self.assertEqual(polygon_indices[2], [0, 1, 4, 3, 2])
    
    def test_explicit_index_sizes_validation(self):
        """Test explicit index_sizes validation."""
        # Flat indices with explicit index_sizes
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 1, 4], dtype=np.uint32)
        explicit_sizes = np.array([3, 4, 3], dtype=np.uint32)  # Triangle, quad, triangle
        
        mesh = Mesh(
            vertices=self.vertices, 
            indices=flat_indices, 
            index_sizes=explicit_sizes
        )
        
        self.assertEqual(mesh.index_count, 10)
        self.assertEqual(mesh.polygon_count, 3)
        self.assertFalse(mesh.is_uniform_polygons)
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 3])
    
    def test_index_sizes_validation_mismatch(self):
        """Test that validation fails when index_sizes doesn't match indices."""
        # Indices sum to 10, but index_sizes sum to 8
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 1, 4], dtype=np.uint32)
        wrong_sizes = np.array([3, 5], dtype=np.uint32)  # Sum = 8, not 10
        
        with self.assertRaises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=flat_indices,
                index_sizes=wrong_sizes
            )
    
    def test_explicit_vs_inferred_index_sizes(self):
        """Test validation when explicit index_sizes conflicts with inferred structure."""
        # 2D array format that infers [4, 4]
        indices = np.array([
            [0, 1, 2, 3],
            [1, 5, 6, 2]
        ], dtype=np.uint32)
        
        # Explicit sizes that conflict with inferred structure
        conflicting_sizes = np.array([3, 5], dtype=np.uint32)
        
        with self.assertRaises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=indices,
                index_sizes=conflicting_sizes
            )
    
    def test_index_sizes_encoding_decoding(self):
        """Test that index_sizes is preserved during encoding/decoding."""
        # Create a mixed polygon mesh
        indices = [
            [0, 1, 2],        # Triangle
            [1, 5, 6, 2],     # Quad
            [0, 3, 4]         # Another triangle
        ]
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        original_index_sizes = mesh.index_sizes.copy()
        
        # Encode the mesh
        encoded_mesh = MeshUtils.encode(mesh)
        
        # Verify index_sizes is in the encoded arrays
        self.assertIn("index_sizes", encoded_mesh.arrays)
        
        # Decode the mesh
        decoded_mesh = MeshUtils.decode(Mesh, encoded_mesh)
        
        # Check that index_sizes is preserved
        np.testing.assert_array_equal(decoded_mesh.index_sizes, original_index_sizes)
        self.assertEqual(decoded_mesh.polygon_count, 3)
        self.assertFalse(decoded_mesh.is_uniform_polygons)
        
        # Check that polygon structure is preserved
        original_polygons = mesh.get_polygon_indices()
        decoded_polygons = decoded_mesh.get_polygon_indices()
        self.assertEqual(len(original_polygons), len(decoded_polygons))
        for orig, decoded in zip(original_polygons, decoded_polygons):
            self.assertEqual(orig, decoded)
    
    def test_index_sizes_serialization(self):
        """Test that index_sizes is preserved during ZIP serialization."""
        # Create a mixed polygon mesh
        indices = [
            [0, 1, 2, 3],     # Quad
            [1, 5, 6],        # Triangle
            [0, 3, 4, 2, 1]   # Pentagon
        ]
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        original_index_sizes = mesh.index_sizes.copy()
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the mesh to a zip file
            MeshUtils.save_to_zip(mesh, temp_path)
            
            # Load the mesh from the zip file
            loaded_mesh = MeshUtils.load_from_zip(Mesh, temp_path)
            
            # Check that index_sizes is preserved
            np.testing.assert_array_equal(loaded_mesh.index_sizes, original_index_sizes)
            self.assertEqual(loaded_mesh.polygon_count, 3)
            self.assertFalse(loaded_mesh.is_uniform_polygons)
            
            # Check that polygon structure is preserved
            original_polygons = mesh.get_polygon_indices()
            loaded_polygons = loaded_mesh.get_polygon_indices()
            self.assertEqual(len(original_polygons), len(loaded_polygons))
            for orig, loaded in zip(original_polygons, loaded_polygons):
                self.assertEqual(orig, loaded)
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_uniform_vs_mixed_polygons(self):
        """Test uniform vs mixed polygon detection."""
        # Uniform triangles
        triangle_indices = [
            [0, 1, 2],
            [1, 5, 6],
            [0, 3, 4]
        ]
        triangle_mesh = Mesh(vertices=self.vertices, indices=triangle_indices)
        self.assertTrue(triangle_mesh.is_uniform_polygons)
        np.testing.assert_array_equal(triangle_mesh.index_sizes, [3, 3, 3])
        
        # Uniform quads
        quad_indices = np.array([
            [0, 1, 2, 3],
            [1, 5, 6, 2]
        ], dtype=np.uint32)
        quad_mesh = Mesh(vertices=self.vertices, indices=quad_indices)
        self.assertTrue(quad_mesh.is_uniform_polygons)
        np.testing.assert_array_equal(quad_mesh.index_sizes, [4, 4])
        
        # Mixed polygons
        mixed_indices = [
            [0, 1, 2],        # Triangle
            [1, 5, 6, 2]      # Quad
        ]
        mixed_mesh = Mesh(vertices=self.vertices, indices=mixed_indices)
        self.assertFalse(mixed_mesh.is_uniform_polygons)
        np.testing.assert_array_equal(mixed_mesh.index_sizes, [3, 4])
    
    def test_polygon_reconstruction(self):
        """Test polygon reconstruction from flat indices and index_sizes."""
        # Create flat indices and sizes
        flat_indices = np.array([0, 1, 2, 1, 5, 6, 2, 0, 3, 4], dtype=np.uint32)
        sizes = np.array([3, 4, 3], dtype=np.uint32)
        
        mesh = Mesh(
            vertices=self.vertices,
            indices=flat_indices,
            index_sizes=sizes
        )
        
        # Test uniform polygon reconstruction (should return 2D array)
        uniform_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        uniform_mesh = Mesh(vertices=self.vertices, indices=uniform_indices)
        reconstructed_uniform = uniform_mesh.get_polygon_indices()
        self.assertIsInstance(reconstructed_uniform, np.ndarray)
        self.assertEqual(reconstructed_uniform.shape, (2, 3))
        np.testing.assert_array_equal(reconstructed_uniform, uniform_indices)
        
        # Test mixed polygon reconstruction (should return list of lists)
        mixed_polygons = mesh.get_polygon_indices()
        self.assertIsInstance(mixed_polygons, list)
        self.assertEqual(len(mixed_polygons), 3)
        self.assertEqual(mixed_polygons[0], [0, 1, 2])
        self.assertEqual(mixed_polygons[1], [1, 5, 6, 2])
        self.assertEqual(mixed_polygons[2], [0, 3, 4])


class TestIndexSizesIntegrity(unittest.TestCase):
    """Test index_sizes integrity during mesh operations."""
    
    def setUp(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.5, 0.5, 1.0]
        ], dtype=np.float32)
    
    def test_copy_preserves_index_sizes(self):
        """Test that copying a mesh preserves index_sizes."""
        # Create a mixed polygon mesh
        indices = [
            [0, 1, 2, 3],     # Quad
            [1, 5, 6],        # Triangle
            [4, 5, 6, 7, 2]   # Pentagon
        ]
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        copied_mesh = mesh.copy()
        
        # Check that index_sizes is preserved in the copy
        np.testing.assert_array_equal(copied_mesh.index_sizes, mesh.index_sizes)
        self.assertEqual(copied_mesh.polygon_count, mesh.polygon_count)
        self.assertEqual(copied_mesh.is_uniform_polygons, mesh.is_uniform_polygons)
        
        # Verify they are independent copies
        self.assertIsNot(copied_mesh.index_sizes, mesh.index_sizes)
        self.assertIsNot(copied_mesh.indices, mesh.indices)
    
    def test_optimization_with_index_sizes(self):
        """Test that mesh optimizations work correctly with index_sizes."""
        # Create a more complex mesh with mixed polygons
        vertices = []
        indices = []
        
        # Create a grid of quads and triangles
        for i in range(3):
            for j in range(3):
                base = len(vertices)
                # Add 4 vertices for a quad
                vertices.extend([
                    [j, i, 0], [j+1, i, 0], [j+1, i+1, 0], [j, i+1, 0]
                ])
                
                if (i + j) % 2 == 0:
                    # Create a quad
                    indices.append([base, base+1, base+2, base+3])
                else:
                    # Create two triangles
                    indices.extend([[base, base+1, base+2], [base, base+2, base+3]])
        
        mesh_vertices = np.array(vertices, dtype=np.float32)
        mesh = Mesh(vertices=mesh_vertices, indices=indices)
        
        original_polygon_count = mesh.polygon_count
        original_is_uniform = mesh.is_uniform_polygons
        
        # Test optimization methods return new meshes with preserved structure
        optimized_cache = MeshUtils.optimize_vertex_cache(mesh)
        optimized_overdraw = MeshUtils.optimize_overdraw(mesh)
        
        # Check that index_sizes structure is preserved in optimized meshes
        self.assertEqual(optimized_cache.polygon_count, original_polygon_count)
        self.assertEqual(optimized_cache.is_uniform_polygons, original_is_uniform)
        self.assertEqual(optimized_overdraw.polygon_count, original_polygon_count)
        self.assertEqual(optimized_overdraw.is_uniform_polygons, original_is_uniform)
        
        # Original mesh should be unchanged
        self.assertEqual(mesh.polygon_count, original_polygon_count)
        self.assertEqual(mesh.is_uniform_polygons, original_is_uniform)


class TestCellTypes(unittest.TestCase):
    """Test cell_types functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.5, 0.5, 1.0]
        ], dtype=np.float32)
    
    def test_cell_types_auto_inference(self):
        """Test automatic inference of cell_types from index_sizes."""
        # Mixed polygon mesh
        indices = [
            [0, 1],           # Line (2 vertices)
            [0, 1, 2],        # Triangle (3 vertices)
            [0, 1, 2, 3],     # Quad (4 vertices)
            [0, 1, 2, 3, 4]   # Pentagon (5 vertices)
        ]
        
        mesh = Mesh(vertices=self.vertices, indices=indices)
        
        # Check that cell_types are automatically inferred
        expected_cell_types = [3, 5, 9, 14]  # Line, Triangle, Quad, Pentagon
        np.testing.assert_array_equal(mesh.cell_types, expected_cell_types)
        np.testing.assert_array_equal(mesh.index_sizes, [2, 3, 4, 5])
    
    def test_cell_types_explicit(self):
        """Test explicit cell_types validation."""
        # Create a mesh with explicit cell_types
        indices = [
            [0, 1, 2],        # Triangle
            [1, 2, 3, 4]      # Quad
        ]
        explicit_cell_types = [5, 9]  # Triangle, Quad
        
        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )
        
        np.testing.assert_array_equal(mesh.cell_types, [5, 9])
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4])
    
    def test_cell_types_length_mismatch(self):
        """Test that validation fails when cell_types length doesn't match index_sizes."""
        indices = [
            [0, 1, 2],        # Triangle
            [1, 2, 3, 4]      # Quad
        ]
        wrong_cell_types = [5]  # Only one cell type for two polygons
        
        with self.assertRaises(ValidationError):
            Mesh(
                vertices=self.vertices,
                indices=indices,
                cell_types=wrong_cell_types
            )
    
    def test_cell_types_encoding_decoding(self):
        """Test that cell_types is preserved during encoding/decoding."""
        # Create a mesh with explicit cell_types
        indices = [
            [0, 1, 2],        # Triangle
            [1, 2, 3, 4],     # Quad
            [0, 4, 5]         # Another triangle
        ]
        explicit_cell_types = [5, 9, 5]  # Triangle, Quad, Triangle
        
        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )
        
        # Encode the mesh
        encoded_mesh = MeshUtils.encode(mesh)
        
        # Verify cell_types is in the encoded arrays
        self.assertIn("cell_types", encoded_mesh.arrays)
        
        # Decode the mesh
        decoded_mesh = MeshUtils.decode(Mesh, encoded_mesh)
        
        # Check that cell_types is preserved
        np.testing.assert_array_equal(decoded_mesh.cell_types, explicit_cell_types)
        np.testing.assert_array_equal(decoded_mesh.index_sizes, [3, 4, 3])
    
    def test_cell_types_serialization(self):
        """Test that cell_types is preserved during ZIP serialization."""
        # Create a mesh with mixed polygons and explicit cell_types
        indices = [
            [0],              # Vertex
            [0, 1],           # Line
            [0, 1, 2],        # Triangle
            [1, 2, 3, 4]      # Quad
        ]
        explicit_cell_types = [1, 3, 5, 9]  # Vertex, Line, Triangle, Quad
        
        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the mesh to a zip file
            MeshUtils.save_to_zip(mesh, temp_path)
            
            # Load the mesh from the zip file
            loaded_mesh = MeshUtils.load_from_zip(Mesh, temp_path)
            
            # Check that cell_types is preserved
            np.testing.assert_array_equal(loaded_mesh.cell_types, explicit_cell_types)
            np.testing.assert_array_equal(loaded_mesh.index_sizes, [1, 2, 3, 4])
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_cell_types_copy_preservation(self):
        """Test that cell_types is preserved during mesh copying."""
        indices = [
            [0, 1, 2, 3],     # Quad
            [1, 2, 4]         # Triangle
        ]
        explicit_cell_types = [9, 5]  # Quad, Triangle
        
        mesh = Mesh(
            vertices=self.vertices,
            indices=indices,
            cell_types=explicit_cell_types
        )
        
        # Copy the mesh
        copied_mesh = mesh.copy()
        
        # Check that cell_types is preserved in the copy
        np.testing.assert_array_equal(copied_mesh.cell_types, mesh.cell_types)
        np.testing.assert_array_equal(copied_mesh.index_sizes, mesh.index_sizes)
        
        # Verify they are independent copies
        self.assertIsNot(copied_mesh.cell_types, mesh.cell_types)
        self.assertIsNot(copied_mesh.index_sizes, mesh.index_sizes)
    
    def test_cell_types_vtk_inference(self):
        """Test VTK cell type inference for various polygon sizes."""
        test_cases = [
            ([0], [1]),                           # Vertex
            ([0, 1], [3]),                        # Line
            ([0, 1, 2], [5]),                     # Triangle
            ([0, 1, 2, 3], [9]),                  # Quad
            ([0, 1, 2, 3, 4], [14]),              # Pyramid
            ([0, 1, 2, 3, 4, 5], [13]),           # Wedge
            ([0, 1, 2, 3, 4, 5, 6, 7], [12]),     # Hexahedron
            ([0, 1, 2, 3, 4, 5, 6, 7, 0], [7])   # Generic polygon (9 vertices)
        ]
        
        for indices_list, expected_cell_types in test_cases:
            with self.subTest(polygon_size=len(indices_list)):
                mesh = Mesh(vertices=self.vertices, indices=[indices_list])
                np.testing.assert_array_equal(mesh.cell_types, expected_cell_types)


if __name__ == '__main__':
    unittest.main()