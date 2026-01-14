"""Unit tests for cell_types module edge topology functions."""

import pytest
import numpy as np
from meshly import CellTypeUtils, VTKCellType


class TestEdgeTopology:
    """Tests for VTK cell type edge topology."""

    def test_get_edge_topology_hexahedron(self):
        """Hexahedron should have 12 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_HEXAHEDRON)
        assert len(edges) == 12
        assert edges.shape == (12, 2)

    def test_get_edge_topology_tetrahedron(self):
        """Tetrahedron should have 6 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TETRA)
        assert len(edges) == 6
        assert edges.shape == (6, 2)

    def test_get_edge_topology_wedge(self):
        """Wedge/prism should have 9 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_WEDGE)
        assert len(edges) == 9

    def test_get_edge_topology_pyramid(self):
        """Pyramid should have 8 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_PYRAMID)
        assert len(edges) == 8

    def test_get_edge_topology_triangle(self):
        """Triangle should have 3 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_TRIANGLE)
        assert len(edges) == 3

    def test_get_edge_topology_quad(self):
        """Quad should have 4 edges."""
        edges = CellTypeUtils.get_edge_topology(VTKCellType.VTK_QUAD)
        assert len(edges) == 4

    def test_get_edge_topology_unknown(self):
        """Unknown cell type should return empty array."""
        edges = CellTypeUtils.get_edge_topology(999)
        assert len(edges) == 0
        assert edges.shape == (0, 2)


class TestGetCellEdges:
    """Tests for get_cell_edges with global vertex indices."""

    def test_hexahedron_edges(self):
        """Test hex edges with global vertex indices."""
        # Hexahedron (cube) with arbitrary global vertex indices
        vertices = [10, 20, 30, 40, 50, 60, 70, 80]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_HEXAHEDRON)
        
        assert len(edges) == 12
        assert edges.shape == (12, 2)
        # All edges should be (min, max) ordered
        for u, v in edges:
            assert u < v
        
        # Check specific edges exist (as sorted tuples)
        edge_set = {tuple(e) for e in edges}
        assert (10, 20) in edge_set
        assert (10, 50) in edge_set

    def test_triangle_edges(self):
        """Test triangle edges."""
        vertices = [5, 10, 15]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_TRIANGLE)
        
        assert len(edges) == 3
        edge_set = {tuple(e) for e in edges}
        assert (5, 10) in edge_set
        assert (10, 15) in edge_set
        assert (5, 15) in edge_set

    def test_tetrahedron_edges(self):
        """Test tetrahedron edges."""
        vertices = [0, 1, 2, 3]
        edges = CellTypeUtils.get_cell_edges(vertices, VTKCellType.VTK_TETRA)
        
        assert len(edges) == 6
        expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        edge_set = {tuple(e) for e in edges}
        assert edge_set == expected

    def test_unknown_type_as_polygon(self):
        """Unknown type should be treated as polygon."""
        vertices = [0, 1, 2, 3, 4]
        edges = CellTypeUtils.get_cell_edges(vertices, 999)
        
        assert len(edges) == 5
        edge_set = {tuple(e) for e in edges}
        assert (0, 1) in edge_set
        assert (1, 2) in edge_set
        assert (2, 3) in edge_set
        assert (3, 4) in edge_set
        assert (0, 4) in edge_set


class TestGetEdgesFromElementSize:
    """Tests for inferring edges from element vertex count."""

    def test_8_vertices_is_hex(self):
        """8 vertices should be treated as hexahedron."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(8)))
        assert len(edges) == 12

    def test_6_vertices_is_wedge(self):
        """6 vertices should be treated as wedge."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(6)))
        assert len(edges) == 9

    def test_5_vertices_is_pyramid(self):
        """5 vertices should be treated as pyramid."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(5)))
        assert len(edges) == 8

    def test_4_vertices_is_quad(self):
        """4 vertices should be treated as quad (not tetra)."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(4)))
        assert len(edges) == 4

    def test_3_vertices_is_triangle(self):
        """3 vertices should be treated as triangle."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(3)))
        assert len(edges) == 3

    def test_2_vertices_is_line(self):
        """2 vertices should be treated as line."""
        edges = CellTypeUtils.get_edges_from_element_size([0, 1])
        assert len(edges) == 1
        assert np.array_equal(edges[0], [0, 1])

    def test_unknown_size_as_polygon(self):
        """Unknown size (e.g., 7) should be treated as polygon."""
        edges = CellTypeUtils.get_edges_from_element_size(list(range(7)))
        assert len(edges) == 7
