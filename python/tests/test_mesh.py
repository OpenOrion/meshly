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

from meshly import Mesh, MeshUtils
from meshly.cell_types import VTKCellType


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
        optimized_mesh = MeshUtils.optimize_vertex_cache(mesh)
        self.assertEqual(optimized_mesh.vertex_count, len(self.vertices))
        self.assertEqual(optimized_mesh.index_count, len(self.indices))

        # Test optimize_overdraw
        overdraw_mesh = MeshUtils.optimize_overdraw(mesh)
        self.assertEqual(overdraw_mesh.vertex_count, len(self.vertices))
        self.assertEqual(overdraw_mesh.index_count, len(self.indices))

        # Test optimize_vertex_fetch
        original_vertex_count = mesh.vertex_count
        fetch_mesh = MeshUtils.optimize_vertex_fetch(mesh)
        self.assertLessEqual(fetch_mesh.vertex_count, original_vertex_count)
        self.assertEqual(fetch_mesh.index_count, len(self.indices))

        # Test simplify
        original_index_count = mesh.index_count
        simplified_mesh = MeshUtils.simplify(mesh, target_ratio=0.5)
        self.assertLessEqual(simplified_mesh.index_count, original_index_count)

    def test_mesh_polygon_support(self):
        """Test mesh polygon support with index_sizes."""
        # Test 2D array input (quad mesh)
        quad_indices = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ], dtype=np.uint32)

        quad_mesh = Mesh(vertices=self.vertices, indices=quad_indices)
        self.assertEqual(quad_mesh.polygon_count, 2)
        self.assertTrue(quad_mesh.is_uniform_polygons)
        np.testing.assert_array_equal(quad_mesh.index_sizes, [4, 4])

        # Test list of lists input (mixed polygons)
        mixed_indices = [
            [0, 1, 2],        # Triangle
            [3, 4, 5, 6]      # Quad
        ]

        mixed_mesh = Mesh(vertices=self.vertices, indices=mixed_indices)
        self.assertEqual(mixed_mesh.polygon_count, 2)
        self.assertFalse(mixed_mesh.is_uniform_polygons)
        np.testing.assert_array_equal(mixed_mesh.index_sizes, [3, 4])

        # Test polygon reconstruction
        reconstructed = mixed_mesh.get_polygon_indices()
        self.assertIsInstance(reconstructed, list)
        self.assertEqual(len(reconstructed), 2)
        self.assertEqual(reconstructed[0], [0, 1, 2])
        self.assertEqual(reconstructed[1], [3, 4, 5, 6])

    def test_mesh_copy(self):
        """Test mesh copying functionality."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        copied_mesh = mesh.copy()

        # Verify the copy has the same data
        self.assertEqual(copied_mesh.vertex_count, mesh.vertex_count)
        self.assertEqual(copied_mesh.index_count, mesh.index_count)
        np.testing.assert_array_equal(copied_mesh.vertices, mesh.vertices)
        np.testing.assert_array_equal(copied_mesh.indices, mesh.indices)

        # Verify they are independent copies
        self.assertIsNot(copied_mesh.vertices, mesh.vertices)
        self.assertIsNot(copied_mesh.indices, mesh.indices)

        # Modify copy and ensure original is unchanged
        copied_mesh.vertices[0, 0] = 999.0
        self.assertNotEqual(copied_mesh.vertices[0, 0], mesh.vertices[0, 0])


class CustomMesh(Mesh):
    """A custom mesh class for testing."""
    normals: np.ndarray = Field(..., description="Vertex normals")
    colors: Optional[np.ndarray] = Field(None, description="Vertex colors")
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list,
                            description="Tags for the mesh")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional properties")


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
        self.assertEqual(mesh.properties, {
                         "shininess": 0.5, "reflectivity": 0.8})

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
            MeshUtils.save_to_zip(mesh, temp_path)

            # Load the mesh from the zip file
            loaded_mesh = MeshUtils.load_from_zip(CustomMesh, temp_path)

            # Check that the loaded mesh has the correct attributes
            self.assertEqual(loaded_mesh.vertex_count, mesh.vertex_count)
            self.assertEqual(loaded_mesh.index_count, mesh.index_count)
            np.testing.assert_array_almost_equal(
                loaded_mesh.vertices, mesh.vertices)
            np.testing.assert_array_almost_equal(
                loaded_mesh.normals, mesh.normals)
            np.testing.assert_array_almost_equal(
                loaded_mesh.colors, mesh.colors)
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
        optimized_mesh = MeshUtils.optimize_vertex_cache(mesh)
        self.assertEqual(optimized_mesh.vertex_count, len(self.vertices))
        self.assertEqual(optimized_mesh.index_count, len(self.indices))
        self.assertIsInstance(optimized_mesh, CustomMesh)

        # Test optimize_overdraw
        overdraw_mesh = MeshUtils.optimize_overdraw(mesh)
        self.assertEqual(overdraw_mesh.vertex_count, len(self.vertices))
        self.assertEqual(overdraw_mesh.index_count, len(self.indices))
        self.assertIsInstance(overdraw_mesh, CustomMesh)

        # Test optimize_vertex_fetch
        original_vertex_count = mesh.vertex_count
        fetch_mesh = MeshUtils.optimize_vertex_fetch(mesh)
        self.assertLessEqual(fetch_mesh.vertex_count, original_vertex_count)
        self.assertEqual(fetch_mesh.index_count, len(self.indices))
        self.assertIsInstance(fetch_mesh, CustomMesh)

        # Test simplify
        original_index_count = mesh.index_count
        simplified_mesh = MeshUtils.simplify(mesh, target_ratio=0.5)
        self.assertLessEqual(simplified_mesh.index_count, original_index_count)
        self.assertIsInstance(simplified_mesh, CustomMesh)

    def test_custom_mesh_with_polygons(self):
        """Test custom mesh with polygon support using index_sizes."""
        # Create a custom mesh with mixed polygons
        mixed_indices = [
            [0, 1, 2],        # Triangle
            [2, 3, 4, 5],     # Quad
            [5, 6, 7, 0, 1]   # Pentagon
        ]

        mesh = CustomMesh(
            vertices=self.vertices,
            indices=mixed_indices,
            normals=self.normals,
            colors=self.colors,
            material_name="mixed_polygon_material",
            tags=["mixed", "polygons"],
            properties={"type": "mixed_mesh"}
        )

        # Check polygon properties
        self.assertEqual(mesh.polygon_count, 3)
        self.assertFalse(mesh.is_uniform_polygons)
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 5])

        # Check that polygon structure is preserved
        reconstructed = mesh.get_polygon_indices()
        self.assertIsInstance(reconstructed, list)
        self.assertEqual(len(reconstructed), 3)
        self.assertEqual(reconstructed[0], [0, 1, 2])
        self.assertEqual(reconstructed[1], [2, 3, 4, 5])
        self.assertEqual(reconstructed[2], [5, 6, 7, 0, 1])

    def test_custom_mesh_copy_with_index_sizes(self):
        """Test copying custom mesh with index_sizes."""
        # Create a quad mesh
        quad_indices = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ], dtype=np.uint32)

        mesh = CustomMesh(
            vertices=self.vertices,
            indices=quad_indices,
            normals=self.normals,
            colors=self.colors,
            material_name="quad_material"
        )

        # Copy the mesh
        copied_mesh = mesh.copy()

        # Verify polygon data is preserved
        self.assertEqual(copied_mesh.polygon_count, mesh.polygon_count)
        self.assertEqual(copied_mesh.is_uniform_polygons,
                         mesh.is_uniform_polygons)
        np.testing.assert_array_equal(
            copied_mesh.index_sizes, mesh.index_sizes)

        # Verify custom attributes are preserved
        self.assertEqual(copied_mesh.material_name, mesh.material_name)
        np.testing.assert_array_equal(copied_mesh.normals, mesh.normals)
        np.testing.assert_array_equal(copied_mesh.colors, mesh.colors)

        # Verify independence
        self.assertIsNot(copied_mesh.index_sizes, mesh.index_sizes)
        self.assertIsNot(copied_mesh.normals, mesh.normals)


class TestMeshMarkers(unittest.TestCase):
    """Test mesh marker functionality."""

    def setUp(self):
        """Set up test data."""
        # Create a simple mesh
        self.vertices = np.array([
            [0.0, 0.0, 0.0],    # vertex 0
            [1.0, 0.0, 0.0],    # vertex 1
            [1.0, 1.0, 0.0],    # vertex 2
            [0.0, 1.0, 0.0],    # vertex 3
        ], dtype=np.float32)

        self.indices = np.array(
            [0, 1, 2, 0, 2, 3], dtype=np.uint32)  # Two triangles

    def test_marker_creation_from_lists(self):
        """Test that markers can be created from list of lists format."""
        markers = {
            "boundary": [
                [0, 1],  # line from vertex 0 to 1
                [1, 2],  # line from vertex 1 to 2
                [2, 3],  # line from vertex 2 to 3
                [3, 0],  # line from vertex 3 to 0
            ],
            "corners": [
                [0, 1, 2],  # triangle corner
            ]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        # Check that markers were converted to flattened structure
        self.assertIn("boundary", mesh.markers)
        self.assertIn("corners", mesh.markers)

        # Check boundary marker
        expected_boundary_indices = [0, 1, 1, 2, 2, 3, 3, 0]
        np.testing.assert_array_equal(
            mesh.markers["boundary"], expected_boundary_indices)
        np.testing.assert_array_equal(
            mesh.marker_sizes["boundary"], [2, 2, 2, 2])
        np.testing.assert_array_equal(mesh.marker_cell_types["boundary"], [
                                      3, 3, 3, 3])  # VTK_LINE

        # Check corner marker
        np.testing.assert_array_equal(mesh.markers["corners"], [0, 1, 2])
        np.testing.assert_array_equal(mesh.marker_sizes["corners"], [3])
        np.testing.assert_array_equal(
            mesh.marker_cell_types["corners"], [5])  # VTK_TRIANGLE

    def test_marker_reconstruction(self):
        """Test that markers can be reconstructed from flattened structure."""
        original_markers = {
            "edges": [
                [0, 1],
                [1, 2],
            ],
            "faces": [
                [0, 1, 2, 3],  # quad
            ]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=original_markers,
            dim=2
        )

        # Reconstruct markers
        reconstructed = mesh.get_reconstructed_markers()

        # Verify reconstruction matches original
        self.assertEqual(reconstructed["edges"], original_markers["edges"])
        self.assertEqual(reconstructed["faces"], original_markers["faces"])

    def test_marker_type_detection(self):
        """Test that marker types are correctly detected based on element size."""
        markers = {
            "lines": [[0, 1], [1, 2]],
            "triangles": [[0, 1, 2]],
            "quads": [[0, 1, 2, 3]],
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        # Check VTK types and sizes
        np.testing.assert_array_equal(mesh.marker_sizes["lines"], [2, 2])
        np.testing.assert_array_equal(mesh.marker_sizes["triangles"], [3])
        np.testing.assert_array_equal(mesh.marker_sizes["quads"], [4])
        np.testing.assert_array_equal(
            mesh.marker_cell_types["lines"], [3, 3])  # VTK_LINE
        np.testing.assert_array_equal(
            mesh.marker_cell_types["triangles"], [5])  # VTK_TRIANGLE
        np.testing.assert_array_equal(
            mesh.marker_cell_types["quads"], [9])  # VTK_QUAD

    def test_marker_large_size(self):
        """Test that markers with large element sizes are now supported."""
        markers = {
            # 7-element polygon (now supported)
            "large_polygon": [[0, 1, 2, 3, 4, 5, 6]],
        }

        # This should now work without error
        mesh = Mesh(
            vertices=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0]
            ], dtype=np.float32),
            indices=self.indices,
            markers=markers,
            dim=2
        )

        # Verify the large polygon was processed correctly
        self.assertIn("large_polygon", mesh.markers)
        np.testing.assert_array_equal(mesh.marker_sizes["large_polygon"], [7])
        np.testing.assert_array_equal(
            mesh.marker_cell_types["large_polygon"], [7])  # VTK_POLYGON

    def test_marker_serialization(self):
        """Test that markers are preserved during serialization."""
        markers = {
            "boundary": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "center": [[0, 1, 2]],
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Save the mesh to a zip file
            MeshUtils.save_to_zip(mesh, temp_path)

            # Load the mesh from the zip file
            loaded_mesh = MeshUtils.load_from_zip(Mesh, temp_path)

            # Check that marker data is preserved
            self.assertIn("boundary", loaded_mesh.markers)
            self.assertIn("center", loaded_mesh.markers)

            # Reconstruct and verify
            reconstructed = loaded_mesh.get_reconstructed_markers()
            self.assertEqual(reconstructed["boundary"], markers["boundary"])
            self.assertEqual(reconstructed["center"], markers["center"])
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_marker_auto_sizes(self):
        """Test that marker_sizes is automatically calculated from marker_cell_types."""
        # Define marker data as a flattened array
        marker_data = np.array([0, 1, 2, 3], dtype=np.uint32)
        
        # Define marker cell types (2 lines: VTK_LINE=3)
        marker_cell_types = np.array([VTKCellType.VTK_LINE, VTKCellType.VTK_LINE], dtype=np.uint8)
        
        # Create mesh with markers and cell types, but without marker_sizes
        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers={'boundary': marker_data},
            marker_cell_types={'boundary': marker_cell_types}
            # Note: marker_sizes is not provided - should be calculated automatically
        )
        
        # Check that marker_sizes was automatically calculated
        self.assertIn('boundary', mesh.marker_sizes)
        expected_sizes = np.array([2, 2], dtype=np.uint32)  # Two lines, each with 2 vertices
        np.testing.assert_array_equal(mesh.marker_sizes['boundary'], expected_sizes)

    def test_marker_auto_sizes_mixed(self):
        """Test automatic size calculation with mixed cell types."""
        # Create extended vertices for more complex test
        extended_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]
        ], dtype=np.float32)
        
        # Define marker data: vertex (1 node), line (2 nodes), triangle (3 nodes)
        marker_data = np.array([0, 1, 2, 0, 1, 4], dtype=np.uint32)
        
        # Define mixed cell types: vertex, line, triangle
        marker_cell_types = np.array([
            VTKCellType.VTK_VERTEX,    # 1 vertex
            VTKCellType.VTK_LINE,      # 2 vertices
            VTKCellType.VTK_TRIANGLE   # 3 vertices
        ], dtype=np.uint8)
        
        # Create mesh with markers and cell types, but without marker_sizes
        mesh = Mesh(
            vertices=extended_vertices,
            indices=self.indices,
            markers={'mixed': marker_data},
            marker_cell_types={'mixed': marker_cell_types}
        )
        
        # Check that marker_sizes was automatically calculated
        self.assertIn('mixed', mesh.marker_sizes)
        expected_sizes = np.array([1, 2, 3], dtype=np.uint32)
        np.testing.assert_array_equal(mesh.marker_sizes['mixed'], expected_sizes)

    def test_marker_manual_sizes_preserved(self):
        """Test that manually provided marker_sizes is preserved."""
        # Define marker data
        marker_data = np.array([0, 1, 1, 2], dtype=np.uint32)
        
        # Define cell types and manual sizes
        marker_cell_types = np.array([VTKCellType.VTK_LINE, VTKCellType.VTK_LINE], dtype=np.uint8)
        marker_sizes = np.array([2, 2], dtype=np.uint32)
        
        # Create mesh with both marker_cell_types and marker_sizes provided
        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers={'boundary': marker_data},
            marker_cell_types={'boundary': marker_cell_types},
            marker_sizes={'boundary': marker_sizes}
        )
        
        # Check that the manually provided marker_sizes was preserved
        self.assertIn('boundary', mesh.marker_sizes)
        np.testing.assert_array_equal(mesh.marker_sizes['boundary'], marker_sizes)


if __name__ == '__main__':
    unittest.main()
