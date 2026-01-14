"""
Tests for mesh triangulation functionality.

This file tests the MeshUtils.triangulate method with various polygon types.
"""
import numpy as np
import pytest

from meshly import Mesh
from meshly.cell_types import VTKCellType


class TestMeshTriangulation:
    """Test mesh triangulation with mixed polygon types."""

    def test_triangulate_already_triangles(self):
        """Test that triangulating an already-triangulated mesh returns a copy."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 1, 3, 4], dtype=np.uint32)
        index_sizes = np.array([3, 3], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        assert tri_mesh.polygon_count == 2
        np.testing.assert_array_equal(tri_mesh.indices, indices)
        np.testing.assert_array_equal(tri_mesh.index_sizes, np.array([3, 3]))
        np.testing.assert_array_equal(
            tri_mesh.cell_types,
            np.array([VTKCellType.VTK_TRIANGLE, VTKCellType.VTK_TRIANGLE])
        )

    def test_triangulate_simple_quad(self):
        """Test triangulating a single quad."""
        # Create a quad mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3], dtype=np.uint32)
        index_sizes = np.array([4], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        # Should have 2 triangles (quad splits into 2 triangles)
        assert tri_mesh.polygon_count == 2
        assert tri_mesh.index_count == 6
        
        # Check triangles: [0,1,2] and [0,2,3]
        expected_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        np.testing.assert_array_equal(tri_mesh.indices, expected_indices)
        np.testing.assert_array_equal(tri_mesh.index_sizes, np.array([3, 3]))
        np.testing.assert_array_equal(
            tri_mesh.cell_types,
            np.array([VTKCellType.VTK_TRIANGLE, VTKCellType.VTK_TRIANGLE])
        )

    def test_triangulate_pentagon(self):
        """Test triangulating a pentagon."""
        # Create a pentagon mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.9, 0.0],
            [0.5, 1.5, 0.0],
            [-0.5, 0.9, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        index_sizes = np.array([5], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        # Pentagon should produce 3 triangles (n-2 = 5-2 = 3)
        assert tri_mesh.polygon_count == 3
        assert tri_mesh.index_count == 9
        
        # Check triangles: [0,1,2], [0,2,3], [0,3,4]
        expected_indices = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4], dtype=np.uint32)
        np.testing.assert_array_equal(tri_mesh.indices, expected_indices)
        np.testing.assert_array_equal(tri_mesh.index_sizes, np.array([3, 3, 3]))

    def test_triangulate_mixed_polygons(self):
        """Test triangulating a complex mesh with mixed polygon types."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.5, 0.9, 0.0], [4.5, 1.5, 0.0], [3.5, 0.9, 0.0],
            [6.0, 0.0, 0.0], [7.0, 0.0, 0.0], [7.5, 0.866, 0.0], [7.0, 1.732, 0.0],
            [6.0, 1.732, 0.0], [5.5, 0.866, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            3, 4, 5, 6,
            7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17,
        ], dtype=np.uint32)
        
        index_sizes = np.array([3, 4, 5, 6], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        
        assert mesh.polygon_count == 4
        assert mesh.index_count == 18
        
        tri_mesh = mesh.triangulate()
        
        expected_triangle_count = 10
        expected_index_count = expected_triangle_count * 3
        
        assert tri_mesh.polygon_count == expected_triangle_count
        assert tri_mesh.index_count == expected_index_count
        
        np.testing.assert_array_equal(
            tri_mesh.index_sizes,
            np.full(expected_triangle_count, 3, dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            tri_mesh.cell_types,
            np.full(expected_triangle_count, VTKCellType.VTK_TRIANGLE, dtype=np.uint32)
        )
        
        quad_triangles = tri_mesh.indices[3:9]
        expected_quad_triangles = np.array([3, 4, 5, 3, 5, 6], dtype=np.uint32)
        np.testing.assert_array_equal(quad_triangles, expected_quad_triangles)

    def test_triangulate_preserves_vertices(self):
        """Test that triangulation preserves all vertices unchanged."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3], dtype=np.uint32)
        index_sizes = np.array([4], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        np.testing.assert_array_equal(tri_mesh.vertices, vertices)
        assert tri_mesh.vertex_count == mesh.vertex_count

    def test_triangulate_preserves_markers(self):
        """Test that triangulation preserves marker data."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3], dtype=np.uint32)
        index_sizes = np.array([4], dtype=np.uint32)
        
        markers = {"boundary": np.array([0, 1, 1, 2], dtype=np.uint32)}
        marker_sizes = {"boundary": np.array([2, 2], dtype=np.uint32)}
        marker_cell_types = {"boundary": np.array(
            [VTKCellType.VTK_LINE, VTKCellType.VTK_LINE], dtype=np.uint32)}
        
        mesh = Mesh(
            vertices=vertices,
            indices=indices,
            index_sizes=index_sizes,
            markers=markers,
            marker_sizes=marker_sizes,
            marker_cell_types=marker_cell_types,
        )
        
        tri_mesh = mesh.triangulate()
        
        assert "boundary" in tri_mesh.markers
        np.testing.assert_array_equal(tri_mesh.markers["boundary"], markers["boundary"])
        np.testing.assert_array_equal(tri_mesh.marker_sizes["boundary"], marker_sizes["boundary"])
        np.testing.assert_array_equal(
            tri_mesh.marker_cell_types["boundary"], marker_cell_types["boundary"])

    def test_triangulate_no_indices_raises_error(self):
        """Test that triangulating a mesh without indices raises an error."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ], dtype=np.float32)
        
        mesh = Mesh(vertices=vertices)
        
        with pytest.raises(ValueError, match="indices"):
            mesh.triangulate()

    def test_triangulate_complex_mesh_with_different_sizes(self):
        """Test triangulation with a variety of polygon sizes including large polygons."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 1.0, 0.0],
            [4.0, 0.5, 0.0], [4.5, 0.0, 0.0], [5.0, 0.2, 0.0], [5.2, 0.7, 0.0],
            [5.0, 1.2, 0.0], [4.5, 1.4, 0.0], [4.0, 1.1, 0.0],
            [6.0, 0.5, 0.0], [6.4, 0.1, 0.0], [6.9, 0.1, 0.0], [7.3, 0.5, 0.0],
            [7.3, 1.0, 0.0], [6.9, 1.4, 0.0], [6.4, 1.4, 0.0], [6.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2,
            3, 4, 5, 6,
            7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21,
        ], dtype=np.uint32)
        
        index_sizes = np.array([3, 4, 7, 8], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        expected_triangles = 14
        
        assert tri_mesh.polygon_count == expected_triangles
        assert tri_mesh.index_count == expected_triangles * 3
        
        np.testing.assert_array_equal(
            tri_mesh.index_sizes,
            np.full(expected_triangles, 3, dtype=np.uint32)
        )
        np.testing.assert_array_equal(
            tri_mesh.cell_types,
            np.full(expected_triangles, VTKCellType.VTK_TRIANGLE, dtype=np.uint32)
        )
        
        assert np.all(tri_mesh.indices < len(vertices))
        assert np.all(tri_mesh.indices >= 0)

    def test_triangulate_fan_pattern_correctness(self):
        """Test that fan triangulation creates correct triangles."""
        # Create a simple pentagon to verify the fan pattern
        vertices = np.array([
            [0.0, 0.0, 0.0],  # vertex 0 (pivot)
            [1.0, 0.0, 0.0],  # vertex 1
            [1.5, 0.9, 0.0],  # vertex 2
            [0.5, 1.5, 0.0],  # vertex 3
            [-0.5, 0.9, 0.0], # vertex 4
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        index_sizes = np.array([5], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes)
        tri_mesh = mesh.triangulate()
        
        # Expected fan triangulation from vertex 0:
        # Triangle 1: [0, 1, 2]
        # Triangle 2: [0, 2, 3]
        # Triangle 3: [0, 3, 4]
        expected_indices = np.array([
            0, 1, 2,  # First triangle
            0, 2, 3,  # Second triangle
            0, 3, 4,  # Third triangle
        ], dtype=np.uint32)
        
        np.testing.assert_array_equal(tri_mesh.indices, expected_indices)


class TestVolumeTriangulation:
    """Test triangulation of 3D volume cells (hexahedra, tetrahedra, etc.)."""

    def test_triangulate_hexahedron(self):
        """Test triangulating a single hexahedron produces 12 triangles."""
        # Create a unit cube hexahedron (VTK vertex ordering)
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0: bottom-front-left
            [1.0, 0.0, 0.0],  # 1: bottom-front-right
            [1.0, 1.0, 0.0],  # 2: bottom-back-right
            [0.0, 1.0, 0.0],  # 3: bottom-back-left
            [0.0, 0.0, 1.0],  # 4: top-front-left
            [1.0, 0.0, 1.0],  # 5: top-front-right
            [1.0, 1.0, 1.0],  # 6: top-back-right
            [0.0, 1.0, 1.0],  # 7: top-back-left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
        index_sizes = np.array([8], dtype=np.uint32)
        cell_types = np.array([VTKCellType.VTK_HEXAHEDRON], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        # Hexahedron has 6 quad faces, each becomes 2 triangles = 12 triangles
        assert tri_mesh.polygon_count == 12
        assert np.all(tri_mesh.cell_types == VTKCellType.VTK_TRIANGLE)
        assert np.all(tri_mesh.index_sizes == 3)

    def test_triangulate_tetrahedron(self):
        """Test triangulating a tetrahedron produces 4 triangles."""
        # Create a simple tetrahedron
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0: base front
            [1.0, 0.0, 0.0],  # 1: base right
            [0.5, 1.0, 0.0],  # 2: base back
            [0.5, 0.5, 1.0],  # 3: apex
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3], dtype=np.uint32)
        index_sizes = np.array([4], dtype=np.uint32)
        cell_types = np.array([VTKCellType.VTK_TETRA], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        # Tetrahedron has 4 triangle faces
        assert tri_mesh.polygon_count == 4
        assert np.all(tri_mesh.cell_types == VTKCellType.VTK_TRIANGLE)

    def test_triangulate_wedge(self):
        """Test triangulating a wedge produces 8 triangles."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
        index_sizes = np.array([6], dtype=np.uint32)
        cell_types = np.array([VTKCellType.VTK_WEDGE], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        assert tri_mesh.polygon_count == 8
        assert np.all(tri_mesh.cell_types == VTKCellType.VTK_TRIANGLE)

    def test_triangulate_pyramid(self):
        """Test triangulating a pyramid produces 6 triangles."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        index_sizes = np.array([5], dtype=np.uint32)
        cell_types = np.array([VTKCellType.VTK_PYRAMID], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        assert tri_mesh.polygon_count == 6
        assert np.all(tri_mesh.cell_types == VTKCellType.VTK_TRIANGLE)

    def test_planar_detected_as_polygon(self):
        """Test that planar cells with volume cell types are treated as polygons."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.9, 0.0],
            [0.5, 1.5, 0.0],
            [-0.5, 0.9, 0.0],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
        index_sizes = np.array([5], dtype=np.uint32)
        cell_types = np.array([VTKCellType.VTK_PYRAMID], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        assert tri_mesh.polygon_count == 3

    def test_multiple_hexahedra(self):
        """Test triangulating multiple hexahedra."""
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0],
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2, 3, 4, 5, 6, 7,
            4, 5, 6, 7, 8, 9, 10, 11,
        ], dtype=np.uint32)
        index_sizes = np.array([8, 8], dtype=np.uint32)
        cell_types = np.array(
            [VTKCellType.VTK_HEXAHEDRON, VTKCellType.VTK_HEXAHEDRON], dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, indices=indices, index_sizes=index_sizes, cell_types=cell_types)
        tri_mesh = mesh.triangulate()
        
        assert tri_mesh.polygon_count == 24
