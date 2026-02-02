"""
Tests for the Pydantic-based Mesh class.

This file contains tests to verify that the Pydantic-based Mesh class works correctly,
including inheritance, validation, and serialization/deserialization.
"""
import os
import tempfile
import numpy as np
import pytest
from typing import Optional, List, Dict, Any
from pydantic import Field, ValidationError

from meshly import Mesh, Array
from meshly.cell_types import VTKCellType


class TestPydanticMesh:
    """Test Pydantic-based Mesh class functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
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
            0, 1, 2, 2, 3, 0,
            1, 5, 6, 6, 2, 1,
            5, 4, 7, 7, 6, 5,
            4, 0, 3, 3, 7, 4,
            3, 2, 6, 6, 7, 3,
            4, 5, 1, 1, 0, 4
        ], dtype=np.uint32)

    def test_mesh_creation(self):
        """Test that a Mesh can be created with vertices and indices."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        assert mesh.vertex_count == len(self.vertices)
        assert mesh.index_count == len(self.indices)
        np.testing.assert_array_equal(mesh.vertices, self.vertices)
        np.testing.assert_array_equal(mesh.indices, self.indices)

    def test_mesh_validation(self):
        """Test that Mesh validation works correctly."""
        # Test that vertices are required
        with pytest.raises(ValidationError):
            Mesh(indices=self.indices)

        # Test that indices are optional
        mesh = Mesh(vertices=self.vertices)
        assert mesh.vertex_count == len(self.vertices)
        assert mesh.index_count == 0
        assert mesh.indices is None

        # Test that vertices are converted to float32
        vertices_int = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1]
        ], dtype=np.int32)

        mesh = Mesh(vertices=vertices_int)
        assert mesh.vertices.dtype == np.float32

        # Test that indices are converted to uint32
        indices_int = np.array([0, 1, 2, 2, 3, 0], dtype=np.int32)
        mesh = Mesh(vertices=self.vertices, indices=indices_int)
        assert mesh.indices.dtype == np.uint32

    def test_mesh_optimization(self):
        """Test that mesh optimization methods work correctly."""
        mesh = Mesh(vertices=self.vertices, indices=self.indices)

        # Test optimize_vertex_cache
        optimized_mesh = mesh.optimize_vertex_cache()
        assert optimized_mesh.vertex_count == len(self.vertices)
        assert optimized_mesh.index_count == len(self.indices)

        # Test optimize_overdraw
        overdraw_mesh = mesh.optimize_overdraw()
        assert overdraw_mesh.vertex_count == len(self.vertices)
        assert overdraw_mesh.index_count == len(self.indices)

        # Test optimize_vertex_fetch
        original_vertex_count = mesh.vertex_count
        fetch_mesh = mesh.optimize_vertex_fetch()
        assert fetch_mesh.vertex_count <= original_vertex_count
        assert fetch_mesh.index_count == len(self.indices)

        # Test simplify
        original_index_count = mesh.index_count
        simplified_mesh = mesh.simplify(target_ratio=0.5)
        assert simplified_mesh.index_count <= original_index_count

    def test_mesh_polygon_support(self):
        """Test mesh polygon support with index_sizes."""
        # Test 2D array input (quad mesh)
        quad_indices = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ], dtype=np.uint32)

        quad_mesh = Mesh(vertices=self.vertices, indices=quad_indices)
        assert quad_mesh.polygon_count == 2
        assert quad_mesh.is_uniform_polygons
        np.testing.assert_array_equal(quad_mesh.index_sizes, [4, 4])

        # Test list of lists input (mixed polygons)
        mixed_indices = [
            [0, 1, 2],        # Triangle
            [3, 4, 5, 6]      # Quad
        ]

        mixed_mesh = Mesh(vertices=self.vertices, indices=mixed_indices)
        assert mixed_mesh.polygon_count == 2
        assert not mixed_mesh.is_uniform_polygons
        np.testing.assert_array_equal(mixed_mesh.index_sizes, [3, 4])

        # Test polygon reconstruction
        reconstructed = mixed_mesh.get_polygon_indices()
        assert isinstance(reconstructed, list)
        assert len(reconstructed) == 2
        assert reconstructed[0] == [0, 1, 2]
        assert reconstructed[1] == [3, 4, 5, 6]

    def test_mesh_copy(self):
        """Test mesh copying functionality."""
        # Create a simple mesh
        mesh = Mesh(vertices=self.vertices, indices=self.indices)
        copied_mesh = mesh.model_copy(deep=True)

        # Verify the copy has the same data
        assert copied_mesh.vertex_count == mesh.vertex_count
        assert copied_mesh.index_count == mesh.index_count
        np.testing.assert_array_equal(copied_mesh.vertices, mesh.vertices)
        np.testing.assert_array_equal(copied_mesh.indices, mesh.indices)

        # Verify they are independent copies
        assert copied_mesh.vertices is not mesh.vertices
        assert copied_mesh.indices is not mesh.indices

        # Modify copy and ensure original is unchanged
        copied_mesh.vertices[0, 0] = 999.0
        assert copied_mesh.vertices[0, 0] != mesh.vertices[0, 0]


class CustomMesh(Mesh):
    """A custom mesh class for testing."""
    normals: Array = Field(..., description="Vertex normals")
    colors: Optional[Array] = Field(None, description="Vertex colors")
    material_name: str = Field("default", description="Material name")
    tags: List[str] = Field(default_factory=list, description="Tags for the mesh")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class TestCustomMesh:
    """Test custom Mesh subclass functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
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
            0, 1, 2, 2, 3, 0,
            1, 5, 6, 6, 2, 1,
            5, 4, 7, 7, 6, 5,
            4, 0, 3, 3, 7, 4,
            3, 2, 6, 6, 7, 3,
            4, 5, 1, 1, 0, 4
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

        assert mesh.vertex_count == len(self.vertices)
        assert mesh.index_count == len(self.indices)
        np.testing.assert_array_equal(mesh.vertices, self.vertices)
        np.testing.assert_array_equal(mesh.indices, self.indices)
        np.testing.assert_array_equal(mesh.normals, self.normals)
        np.testing.assert_array_equal(mesh.colors, self.colors)
        assert mesh.material_name == "test_material"
        assert mesh.tags == ["test", "cube"]
        assert mesh.properties == {"shininess": 0.5, "reflectivity": 0.8}

    def test_custom_mesh_validation(self):
        """Test that custom mesh validation works correctly."""
        with pytest.raises(ValidationError):
            CustomMesh(
                vertices=self.vertices,
                indices=self.indices,
                colors=self.colors
            )

        mesh = CustomMesh(
            vertices=self.vertices,
            indices=self.indices,
            normals=self.normals
        )
        assert mesh.colors is None

        assert mesh.material_name == "default"
        assert mesh.tags == []
        assert mesh.properties == {}

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

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = CustomMesh.load_from_zip(temp_path)

            assert loaded_mesh.vertex_count == mesh.vertex_count
            assert loaded_mesh.index_count == mesh.index_count
            np.testing.assert_array_almost_equal(loaded_mesh.vertices, mesh.vertices)
            np.testing.assert_array_almost_equal(loaded_mesh.normals, mesh.normals)
            np.testing.assert_array_almost_equal(loaded_mesh.colors, mesh.colors)
            assert loaded_mesh.material_name == mesh.material_name
            assert loaded_mesh.tags == mesh.tags
            assert loaded_mesh.properties == mesh.properties
        finally:
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

        optimized_mesh = mesh.optimize_vertex_cache()
        assert optimized_mesh.vertex_count == len(self.vertices)
        assert optimized_mesh.index_count == len(self.indices)
        assert isinstance(optimized_mesh, CustomMesh)

        overdraw_mesh = mesh.optimize_overdraw()
        assert overdraw_mesh.vertex_count == len(self.vertices)
        assert overdraw_mesh.index_count == len(self.indices)
        assert isinstance(overdraw_mesh, CustomMesh)

        original_vertex_count = mesh.vertex_count
        fetch_mesh = mesh.optimize_vertex_fetch()
        assert fetch_mesh.vertex_count <= original_vertex_count
        assert fetch_mesh.index_count == len(self.indices)
        assert isinstance(fetch_mesh, CustomMesh)

        original_index_count = mesh.index_count
        simplified_mesh = mesh.simplify(target_ratio=0.5)
        assert simplified_mesh.index_count <= original_index_count
        assert isinstance(simplified_mesh, CustomMesh)

    def test_custom_mesh_with_polygons(self):
        """Test custom mesh with polygon support using index_sizes."""
        # Test with mixed polygon types: triangle, quad, pentagon
        mixed_indices = [[0, 1, 2], [2, 3, 4, 5], [5, 6, 7, 0, 1]]

        mesh = CustomMesh(
            vertices=self.vertices,
            indices=mixed_indices,
            normals=self.normals,
            colors=self.colors,
            material_name="mixed_polygon_material",
            tags=["mixed", "polygons"],
            properties={"type": "mixed_mesh"}
        )

        assert mesh.polygon_count == 3
        assert not mesh.is_uniform_polygons
        np.testing.assert_array_equal(mesh.index_sizes, [3, 4, 5])

        reconstructed = mesh.get_polygon_indices()
        assert isinstance(reconstructed, list)
        assert len(reconstructed) == 3
        assert reconstructed[0] == [0, 1, 2]
        assert reconstructed[1] == [2, 3, 4, 5]
        assert reconstructed[2] == [5, 6, 7, 0, 1]

    def test_custom_mesh_copy_with_index_sizes(self):
        """Test copying custom mesh with index_sizes."""
        quad_indices = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint32)

        mesh = CustomMesh(
            vertices=self.vertices,
            indices=quad_indices,
            normals=self.normals,
            colors=self.colors,
            material_name="quad_material"
        )

        copied_mesh = mesh.model_copy(deep=True)

        assert copied_mesh.polygon_count == mesh.polygon_count
        assert copied_mesh.is_uniform_polygons == mesh.is_uniform_polygons
        np.testing.assert_array_equal(copied_mesh.index_sizes, mesh.index_sizes)

        assert copied_mesh.material_name == mesh.material_name
        np.testing.assert_array_equal(copied_mesh.normals, mesh.normals)
        np.testing.assert_array_equal(copied_mesh.colors, mesh.colors)

        assert copied_mesh.index_sizes is not mesh.index_sizes
        assert copied_mesh.normals is not mesh.normals


class TestMeshMarkers:
    """Test mesh marker functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test data."""
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        self.indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    def test_marker_creation_from_lists(self):
        """Test that markers can be created from list of lists format."""
        markers = {
            "boundary": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "corners": [[0, 1, 2]]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        assert "boundary" in mesh.markers
        assert "corners" in mesh.markers

        expected_boundary_indices = [0, 1, 1, 2, 2, 3, 3, 0]
        np.testing.assert_array_equal(mesh.markers["boundary"], expected_boundary_indices)
        np.testing.assert_array_equal(mesh.marker_sizes["boundary"], [2, 2, 2, 2])
        np.testing.assert_array_equal(mesh.marker_cell_types["boundary"], [3, 3, 3, 3])

        np.testing.assert_array_equal(mesh.markers["corners"], [0, 1, 2])
        np.testing.assert_array_equal(mesh.marker_sizes["corners"], [3])
        np.testing.assert_array_equal(mesh.marker_cell_types["corners"], [5])

    def test_marker_reconstruction(self):
        """Test that markers can be reconstructed from flattened structure."""
        original_markers = {
            "edges": [[0, 1], [1, 2]],
            "faces": [[0, 1, 2, 3]]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=original_markers,
            dim=2
        )

        reconstructed = mesh.get_reconstructed_markers()

        assert reconstructed["edges"] == original_markers["edges"]
        assert reconstructed["faces"] == original_markers["faces"]

    def test_marker_type_detection(self):
        """Test that marker types are correctly detected based on element size."""
        markers = {
            "lines": [[0, 1], [1, 2]],
            "triangles": [[0, 1, 2]],
            "quads": [[0, 1, 2, 3]]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        np.testing.assert_array_equal(mesh.marker_sizes["lines"], [2, 2])
        np.testing.assert_array_equal(mesh.marker_sizes["triangles"], [3])
        np.testing.assert_array_equal(mesh.marker_sizes["quads"], [4])
        np.testing.assert_array_equal(mesh.marker_cell_types["lines"], [3, 3])
        np.testing.assert_array_equal(mesh.marker_cell_types["triangles"], [5])
        np.testing.assert_array_equal(mesh.marker_cell_types["quads"], [9])

    def test_marker_large_size(self):
        """Test that markers with large element sizes are now supported."""
        markers = {"large_polygon": [[0, 1, 2, 3, 4, 5, 6]]}

        mesh = Mesh(
            vertices=np.array([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]
            ], dtype=np.float32),
            indices=self.indices,
            markers=markers,
            dim=2
        )

        assert "large_polygon" in mesh.markers
        np.testing.assert_array_equal(mesh.marker_sizes["large_polygon"], [7])
        np.testing.assert_array_equal(mesh.marker_cell_types["large_polygon"], [7])

    def test_marker_serialization(self):
        """Test that markers are preserved during serialization."""
        markers = {
            "boundary": [[0, 1], [1, 2], [2, 3], [3, 0]],
            "center": [[0, 1, 2]]
        }

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers=markers,
            dim=2
        )

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            mesh.save_to_zip(temp_path)
            loaded_mesh = Mesh.load_from_zip(temp_path)

            assert "boundary" in loaded_mesh.markers
            assert "center" in loaded_mesh.markers

            reconstructed = loaded_mesh.get_reconstructed_markers()
            assert reconstructed["boundary"] == markers["boundary"]
            assert reconstructed["center"] == markers["center"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_marker_auto_sizes(self):
        """Test that marker_sizes is automatically calculated from marker_cell_types."""
        marker_data = np.array([0, 1, 2, 3], dtype=np.uint32)
        marker_cell_types = np.array([VTKCellType.VTK_LINE, VTKCellType.VTK_LINE], dtype=np.uint8)

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers={'boundary': marker_data},
            marker_cell_types={'boundary': marker_cell_types}
        )

        assert 'boundary' in mesh.marker_sizes
        expected_sizes = np.array([2, 2], dtype=np.uint32)
        np.testing.assert_array_equal(mesh.marker_sizes['boundary'], expected_sizes)

    def test_marker_auto_sizes_mixed(self):
        """Test automatic size calculation with mixed cell types."""
        extended_vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0], [0.5, 0.5, 0.0]
        ], dtype=np.float32)

        marker_data = np.array([0, 1, 2, 0, 1, 4], dtype=np.uint32)
        marker_cell_types = np.array([
            VTKCellType.VTK_VERTEX,
            VTKCellType.VTK_LINE,
            VTKCellType.VTK_TRIANGLE
        ], dtype=np.uint8)

        mesh = Mesh(
            vertices=extended_vertices,
            indices=self.indices,
            markers={'mixed': marker_data},
            marker_cell_types={'mixed': marker_cell_types}
        )

        assert 'mixed' in mesh.marker_sizes
        expected_sizes = np.array([1, 2, 3], dtype=np.uint32)
        np.testing.assert_array_equal(mesh.marker_sizes['mixed'], expected_sizes)

    def test_marker_manual_sizes_preserved(self):
        """Test that manually provided marker_sizes is preserved."""
        marker_data = np.array([0, 1, 1, 2], dtype=np.uint32)
        marker_cell_types = np.array([VTKCellType.VTK_LINE, VTKCellType.VTK_LINE], dtype=np.uint8)
        marker_sizes = np.array([2, 2], dtype=np.uint32)

        mesh = Mesh(
            vertices=self.vertices,
            indices=self.indices,
            markers={'boundary': marker_data},
            marker_cell_types={'boundary': marker_cell_types},
            marker_sizes={'boundary': marker_sizes}
        )

        assert 'boundary' in mesh.marker_sizes
        np.testing.assert_array_equal(mesh.marker_sizes['boundary'], marker_sizes)
